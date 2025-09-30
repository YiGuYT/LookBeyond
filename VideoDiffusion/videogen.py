# Full hybrid interpolation script (single file)
# - Defines a unified FrameInterpolationWithNoiseInjectionPipeline that supports BOTH
#   spatial weights (poses_current/pose_source/pose_target) and ray-map conditioning
#   (pose_embeddings + linear_fuser).
# - Provides a runnable CLI that prepares inputs and runs inference.

import os
import argparse
import copy
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Union
from einops import rearrange

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils import load_image, export_to_video
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.utils import logging
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,
    tensor2vid,
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
)
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

# Optional: attention control hooks (only used if available)
try:
    from attn_ctrl.attention_control import (
        AttentionStore,
        register_temporal_self_attention_control,
        register_temporal_self_attention_flip_control,
    )
    ATTENTION_CTRL_OK = True
except Exception:
    ATTENTION_CTRL_OK = False

# Ray-map utilities (Plücker rays)
from camctrl.pose_adaptor import CameraPoseEncoder
from dataset.realestate10K import ray_condition
from dataset.realestate10K import Camera

logger = logging.get_logger(__name__)


# ============= Unified Pipeline ============= #
class FrameInterpolationWithNoiseInjectionPipeline(DiffusionPipeline):
    """Unified pipeline that supports spatial weights and/or ray-map conditioning."""

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.ori_unet = copy.deepcopy(unet)

    # -------- encoders -------- #
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, _, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, 1, -1)

        if do_classifier_free_guidance:
            negative = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative, image_embeddings])
        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        if do_classifier_free_guidance:
            negative = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative, image_latents])
        return image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected = self.unet.add_embedding.linear_1.in_features
        if expected != passed:
            raise ValueError(
                f"Added time embed dim mismatch: expected {expected}, got {passed}."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            kwargs = {"num_frames": num_frames_in} if accepts_num_frames else {}
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        return frames.float()

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(f"`image` must be Tensor, PIL.Image or list[PIL.Image], not {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("`height` and `width` must be divisible by 8")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("generator list length must match batch size")
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents * self.scheduler.init_noise_sigma

    # -------- guidance + spatial weighting -------- #
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def angular_distance(self, q_a, q_b):
        q_a = torch.nn.functional.normalize(q_a, dim=-1)
        q_b = torch.nn.functional.normalize(q_b, dim=-1)
        dot = torch.sum(q_a * q_b, dim=-1).clamp(-1.0, 1.0)
        return 2.0 * torch.acos(torch.abs(dot))

    def pose_similarity(self, p_current, p_source, p_target, sigma_q=1.0, sigma_t=1.0):
        q_current, t_current = p_current
        q_source, t_source = p_source
        q_target, t_target = p_target
        w_q_source = torch.exp(-self.angular_distance(q_current, q_source) / sigma_q)
        w_q_target = torch.exp(-self.angular_distance(q_current, q_target) / sigma_q)
        w_t_source = torch.exp(-torch.norm(t_current - t_source, dim=-1) / sigma_t)
        w_t_target = torch.exp(-torch.norm(t_current - t_target, dim=-1) / sigma_t)
        w_source = w_q_source * w_t_source
        w_target = w_q_target * w_t_target
        w_sum = w_source + w_target + 1e-8
        return w_source / w_sum, w_target / w_sum

    def matrix_to_translation_quaternion(self, matrix):
        translation = matrix[:, :3, 3]
        rotation_matrix = matrix[:, :3, :3]
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        return translation, quaternion

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        B = rotation_matrix.shape[0]
        q = torch.zeros((B, 4), device=rotation_matrix.device, dtype=rotation_matrix.dtype)
        R = rotation_matrix
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        for i in range(B):
            if trace[i] > 0:
                s = torch.sqrt(trace[i] + 1.0) * 2
                qw = 0.25 * s
                qx = (R[i, 2, 1] - R[i, 1, 2]) / s
                qy = (R[i, 0, 2] - R[i, 2, 0]) / s
                qz = (R[i, 1, 0] - R[i, 0, 1]) / s
            else:
                if R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
                    s = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                    qw = (R[i, 2, 1] - R[i, 1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[i, 0, 1] + R[i, 1, 0]) / s
                    qz = (R[i, 0, 2] + R[i, 2, 0]) / s
                elif R[i, 1, 1] > R[i, 2, 2]:
                    s = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                    qw = (R[i, 0, 2] - R[i, 2, 0]) / s
                    qx = (R[i, 0, 1] + R[i, 1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[i, 1, 2] + R[i, 2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                    qw = (R[i, 1, 0] - R[i, 0, 1]) / s
                    qx = (R[i, 0, 2] + R[i, 2, 0]) / s
                    qy = (R[i, 1, 2] + R[i, 2, 1]) / s
                    qz = 0.25 * s
            q[i] = torch.tensor([qw, qx, qy, qz], device=rotation_matrix.device, dtype=rotation_matrix.dtype)
        return q

    # -------- multidiffusion step -------- #
    @torch.no_grad()
    def multidiffusion_step(
        self,
        latents,
        t,
        image1_embeddings,
        image2_embeddings,
        image1_latents,
        image2_latents,
        added_time_ids,
        avg_weight,
    ):
        latents1 = latents
        latents2 = torch.flip(latents, (1,))
        in1 = torch.cat([latents1] * 2) if self.do_classifier_free_guidance else latents1
        in1 = self.scheduler.scale_model_input(in1, t)
        in2 = torch.cat([latents2] * 2) if self.do_classifier_free_guidance else latents2
        in2 = self.scheduler.scale_model_input(in2, t)

        in1 = torch.cat([in1, image1_latents], dim=2)
        in2 = torch.cat([in2, image2_latents], dim=2)

        n1 = self.ori_unet(in1, t, encoder_hidden_states=image1_embeddings, added_time_ids=added_time_ids, return_dict=False)[0]
        n2 = self.unet(in2, t, encoder_hidden_states=image2_embeddings, added_time_ids=added_time_ids, return_dict=False)[0]

        if self.do_classifier_free_guidance:
            u1, c1 = n1.chunk(2)
            n1 = u1 + self.guidance_scale * (c1 - u1)
            u2, c2 = n2.chunk(2)
            n2 = u2 + self.guidance_scale * (c2 - u2)

        n2 = torch.flip(n2, (1,))
        if isinstance(avg_weight, torch.Tensor):
            return avg_weight * n1 + (1 - avg_weight) * n2
        return 0.5 * (n1 + n2)

    # -------- call -------- #
    @torch.no_grad()
    def __call__(
        self,
        image1: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image2: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        # spatial weights (optional)
        poses_current: Optional[torch.Tensor] = None,  # [B,T,4,4]
        pose_source: Optional[torch.Tensor] = None,    # [B,4,4]
        pose_target: Optional[torch.Tensor] = None,    # [B,4,4]
        # ray-map (optional)
        pose_embeddings: Optional[torch.FloatTensor] = None,  # [B,T,4,H,W]
        linear_fuser: Optional[nn.Module] = None,
        # common params
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        weighted_average: bool = False,
        noise_injection_steps: int = 0,
        noise_injection_ratio: float = 0.0,
        return_dict: bool = True,
    ):
        # defaults
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # checks
        self.check_inputs(image1, height, width)
        self.check_inputs(image2, height, width)

        # batch + device
        if isinstance(image1, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image1, list):
            batch_size = len(image1)
        else:
            batch_size = image1.shape[0]
        device = self._execution_device

        # init guidance
        self._guidance_scale = max_guidance_scale

        # CLIP enc
        emb1 = self._encode_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        emb2 = self._encode_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # preprocess for VAE
        fps = fps - 1
        im1 = self.image_processor.preprocess(image1, height=height, width=width).to(device)
        im2 = self.image_processor.preprocess(image2, height=height, width=width).to(device)
        noise = randn_tensor(im1.shape, generator=generator, device=im1.device, dtype=im1.dtype)
        im1 = im1 + noise_aug_strength * noise
        im2 = im2 + noise_aug_strength * noise

        needs_up = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_up:
            self.vae.to(dtype=torch.float32)

        lat1 = self._encode_vae_image(im1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        lat1 = lat1.to(emb1.dtype)
        latents1 = lat1.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        lat2 = self._encode_vae_image(im2, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        lat2 = lat2.to(emb2.dtype)
        latents2 = lat2.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # ray-map fuse
        if (pose_embeddings is not None) and (linear_fuser is not None):
            pe = pose_embeddings.to(latents1.dtype)
            if pe.shape != latents1.shape:
                raise ValueError(f"pose_embeddings shape {pe.shape} must equal {latents1.shape}")
            def fuse(L, P):
                L = L + P
                L = L.permute(0, 1, 3, 4, 2).contiguous()  # [B,T,H,W,C]
                L = linear_fuser(L)
                return L.permute(0, 1, 4, 2, 3).contiguous()
            latents1 = fuse(latents1, pe)
            latents2 = fuse(latents2, torch.flip(pe, (1,)))

        if needs_up:
            self.vae.to(dtype=torch.float16)

        # time ids
        add_t = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, emb1.dtype, batch_size, num_videos_per_prompt, self.do_classifier_free_guidance
        ).to(device)

        # timesteps + init latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_ch = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_ch,
            height,
            width,
            emb1.dtype,
            device,
            generator,
            latents,
        )

        # guidance schedule
        if (poses_current is not None) and (pose_source is not None) and (pose_target is not None):
            B, T = poses_current.shape[:2]
            pc_flat = poses_current.view(B * T, 4, 4)
            t_curr, q_curr = self.matrix_to_translation_quaternion(pc_flat)
            t_curr = t_curr.view(B, T, -1)
            q_curr = q_curr.view(B, T, -1)
            t_src, q_src = self.matrix_to_translation_quaternion(pose_source)
            t_tgt, q_tgt = self.matrix_to_translation_quaternion(pose_target)
            t_src = t_src.unsqueeze(1).expand(-1, T, -1)
            q_src = q_src.unsqueeze(1).expand(-1, T, -1)
            t_tgt = t_tgt.unsqueeze(1).expand(-1, T, -1)
            q_tgt = q_tgt.unsqueeze(1).expand(-1, T, -1)
            w_s, w_t = self.pose_similarity((q_curr, t_curr), (q_src, t_src), (q_tgt, t_tgt), 1.0, 1.0)
            g = w_s * min_guidance_scale + w_t * max_guidance_scale
            g = _append_dims(g.to(device, latents.dtype), latents.ndim)
            if weighted_average:
                self._guidance_scale = g
                avg_w = _append_dims(w_s.to(device, latents.dtype), latents.ndim)
            else:
                self._guidance_scale = (g + torch.flip(g, (1,))) * 0.5
                avg_w = 0.5
        else:
            g = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            g = g.to(device, latents.dtype)
            g = g.repeat(batch_size * num_videos_per_prompt, 1)
            g = _append_dims(g, latents.ndim)
            if weighted_average:
                self._guidance_scale = g
                w = torch.linspace(1, 0, num_frames).unsqueeze(0).to(device, latents.dtype)
                avg_w = _append_dims(w.repeat(batch_size * num_videos_per_prompt, 1), latents.ndim)
            else:
                self._guidance_scale = (g + torch.flip(g, (1,))) * 0.5
                avg_w = 0.5

        # denoise
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.ori_unet = self.ori_unet.to(device)
        inj_thresh = int(num_inference_steps * noise_injection_ratio)

        with self.progress_bar(total=num_inference_steps) as bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.multidiffusion_step(latents, t, emb1, emb2, latents1, latents2, add_t, avg_w)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if i < inj_thresh and noise_injection_steps > 0:
                    sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_tm1 = self.scheduler.sigmas[self.scheduler.step_index + 1]
                    sigma = torch.sqrt(sigma_t ** 2 - sigma_tm1 ** 2)
                    for _ in range(noise_injection_steps):
                        extra = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype) * sigma
                        latents = latents + extra
                        noise_pred = self.multidiffusion_step(latents, t, emb1, emb2, latents1, latents2, add_t, avg_w)
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                self.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    bar.update()

        # decode
        if output_type != "latent":
            if needs_up:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


# ============= Helper functions for CLI ============= #

def crop_square_center(img: PIL.Image.Image) -> PIL.Image.Image:
    W, H = img.size
    if W < H:
        left, right = 0, W
        top = int(np.ceil((H - W) / 2.0))
        bottom = top + W
    else:
        left = int(np.ceil((W - H) / 2.0))
        right = left + H
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))


def load_camera_parameters_matrix_posefile(pose_file: str):
    pose_dict = {}
    with open(pose_file, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        frame_id = int(tokens[0])
        r00, r01, r02 = map(float, tokens[7:10])
        r10, r11, r12 = map(float, tokens[10:13])
        r20, r21, r22 = map(float, tokens[13:16])
        tx, ty, tz = map(float, tokens[16:19])
        R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float32)
        t = np.array([tx, ty, tz], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        pose_dict[frame_id] = c2w
    return pose_dict


def load_camera_parameters_camera_objects(pose_file: str):
    pose_dict = {}
    with open(pose_file, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        entries = [float(x) for x in line.strip().split()]
        frame_id = int(entries[0])
        pose_dict[frame_id] = Camera(entries)
    return pose_dict


def get_frame_id_from_image_path(image_path: str) -> int:
    return int(os.path.splitext(os.path.basename(image_path))[0])


def select_between_frames(sorted_keys, start_frame_id: int, end_frame_id: int, num_frames: int):
    if start_frame_id > end_frame_id:
        start_frame_id, end_frame_id = end_frame_id, start_frame_id
    frame_ids = [fid for fid in sorted_keys if start_frame_id <= fid <= end_frame_id]
    if len(frame_ids) < num_frames:
        raise ValueError(
            f"Not enough frames between frame IDs {start_frame_id} and {end_frame_id} to select {num_frames} frames."
        )
    idx = np.linspace(0, len(frame_ids) - 1, num_frames, dtype=int)
    return [frame_ids[i] for i in idx]


def compute_plucker_embeddings_from_camera_dict(
    cam_dict: dict,
    start_frame_id: int,
    end_frame_id: int,
    num_frames: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    selected_ids = select_between_frames(sorted(cam_dict.keys()), start_frame_id, end_frame_id, num_frames)
    selected_cams = [cam_dict[fid] for fid in selected_ids]
    intrinsics = np.asarray(
        [[cam.fx * W, cam.fy * H, cam.cx * W, cam.cy * H] for cam in selected_cams], dtype=np.float32
    )
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0)  # [1, T, 4]
    c2w_poses = np.array([cam.c2w_mat for cam in selected_cams], dtype=np.float32)
    c2w = torch.from_numpy(c2w_poses).unsqueeze(0)  # [1, T, 4, 4]
    plucker = ray_condition(intrinsics.to(device), c2w.to(device), H, W, device=device)[0]  # [T, H, W, 6]
    return plucker.permute(0, 3, 1, 2).contiguous()  # [T, 6, H, W]


# ============= CLI main ============= #

def main(args):
    torch_dtype = torch.float16 if args.precision == "fp16" else torch.float32

    # scheduler + pipeline
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = FrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        variant=args.precision,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir,
    )

    # merge weights (delta or full)
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir, subfolder="unet", torch_dtype=torch_dtype, cache_dir=args.cache_dir
    )
    assert finetuned_unet.config.num_frames == args.num_frames
    state_dict = pipe.unet.state_dict()

    if args.delta_mode == "temporal_attn_delta":
        ori_unet = UNetSpatioTemporalConditionModel.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="unet", variant=args.precision, torch_dtype=torch_dtype, cache_dir=args.cache_dir
        )
        finetuned_state = finetuned_unet.state_dict()
        ori_state = ori_unet.state_dict()
        for name, param in finetuned_state.items():
            if (
                "temporal_transformer_blocks.0.attn1.to_v" in name
                or "temporal_transformer_blocks.0.attn1.to_out.0" in name
            ):
                delta_w = param - ori_state[name]
                state_dict[name] = state_dict[name] + delta_w
    else:
        # full replace
        state_dict.update(finetuned_unet.state_dict())

    pipe.unet.load_state_dict(state_dict)

    # optional attention control (if installed)
    if ATTENTION_CTRL_OK:
        controller_ref = AttentionStore(); register_temporal_self_attention_control(pipe.ori_unet, controller_ref)
        controller = AttentionStore(); register_temporal_self_attention_flip_control(pipe.unet, controller, controller_ref)

    device = torch.device(args.device)
    pipe = pipe.to(device)

    # RNG
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # load images
    frame1 = load_image(args.frame1_path)
    frame2 = load_image(args.frame2_path)
    if args.square_crop:
        frame1 = crop_square_center(frame1)
        frame2 = crop_square_center(frame2)
    frame1 = frame1.resize((args.width, args.height))
    frame2 = frame2.resize((args.width, args.height))

    # spatial poses (optional)
    poses_current = pose_source = pose_target = None
    if not args.skip_spatial_weight:
        pose_dict_mats = load_camera_parameters_matrix_posefile(args.pose_file)
        fid1 = get_frame_id_from_image_path(args.frame1_path)
        fid2 = get_frame_id_from_image_path(args.frame2_path)
        sel_ids = select_between_frames(sorted(pose_dict_mats.keys()), fid1, fid2, args.num_frames)
        c2w_mats = np.stack([pose_dict_mats[f] for f in sel_ids], axis=0).astype(np.float32)
        poses_current = torch.from_numpy(c2w_mats).unsqueeze(0).to(device)  # [1,T,4,4]
        pose_source = torch.from_numpy(pose_dict_mats[fid1]).unsqueeze(0).to(device)  # [1,4,4]
        pose_target = torch.from_numpy(pose_dict_mats[fid2]).unsqueeze(0).to(device)  # [1,4,4]

    # ray-map (optional)
    pose_embeddings = None
    linear_fuser = None
    if not args.skip_raymap:
        # Load pose encoder + reducers
        pose_encoder = CameraPoseEncoder(); pose_encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_2_state_dict.pth"), map_location="cpu"))
        pose_encoder.eval().requires_grad_(False).to(device)
        channel_reducer = nn.Conv2d(in_channels=320, out_channels=4, kernel_size=1)
        channel_reducer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_1_state_dict.pth"), map_location="cpu"))
        channel_reducer.eval().requires_grad_(False).to(device)
        linear_fuser = nn.Linear(4, 4)
        linear_fuser.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_3_state_dict.pth"), map_location="cpu"))
        linear_fuser.eval().requires_grad_(False).to(device)
        if torch_dtype == torch.float16:
            linear_fuser = linear_fuser.half()

        # build plücker -> pose_embeddings
        H, W = args.height, args.width
        cam_dict = load_camera_parameters_camera_objects(args.pose_file)
        fid1 = get_frame_id_from_image_path(args.frame1_path)
        fid2 = get_frame_id_from_image_path(args.frame2_path)
        plucker = compute_plucker_embeddings_from_camera_dict(cam_dict, fid1, fid2, args.num_frames, H, W, device)
        # run encoder
        with torch.no_grad():
            pe_in = plucker.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1,T,6,H,W]
            feat320 = pose_encoder(pe_in)                        # [(B*T),320,H,W]
            raymap_reduced = channel_reducer(feat320)            # [(B*T),4,H,W]
        pose_embeddings = rearrange(raymap_reduced, '(b f) c h w -> b f c h w', b=1).to(device)

    # run pipeline
    result = pipe(
        image1=frame1, image2=frame2,
        height=args.height, width=args.width,
        num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
        generator=generator,
        weighted_average=args.weighted_average,
        noise_injection_steps=args.noise_injection_steps,
        noise_injection_ratio=args.noise_injection_ratio,
        # spatial
        poses_current=poses_current, pose_source=pose_source, pose_target=pose_target,
        # ray-map
        pose_embeddings=pose_embeddings, linear_fuser=linear_fuser,
    ).frames[0]

    # save
    if args.out_path.endswith('.gif'):
        result[0].save(args.out_path, save_all=True, append_images=result[1:], duration=142, loop=0)
    else:
        export_to_video(result, args.out_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full hybrid (spatial + ray-map) frame interpolation")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--frame1_path", type=str, required=True)
    parser.add_argument("--frame2_path", type=str, required=True)
    parser.add_argument("--pose_file", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=7)

    parser.add_argument("--weighted_average", action='store_true')
    parser.add_argument("--noise_injection_steps", type=int, default=0)
    parser.add_argument("--noise_injection_ratio", type=float, default=0.5)

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--precision", type=str, choices=['fp16','fp32'], default='fp16')
    parser.add_argument("--delta_mode", type=str, choices=['temporal_attn_delta','full'], default='temporal_attn_delta')
    parser.add_argument("--square_crop", action='store_true')

    parser.add_argument("--skip_spatial_weight", action='store_true')
    parser.add_argument("--skip_raymap", action='store_true')

    parser.add_argument("--cache_dir", type=str, default='hub')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    os.environ.setdefault('HF_HOME', 'cache')
    os.environ.setdefault('HF_HUB_CACHE', args.cache_dir)

    main(args)
