import os
from glob import glob
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from packaging import version as pver

class Camera(object):
    def __init__(self, entry):
        """
        Parses a single line from the annotation file and initializes camera parameters.

        Args:
            entry (list of float): List of floats parsed from a line in the annotation file.
        """
        # Entry[0] is frame ID (ignored here)
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # Skip entries[5:7] as they are zeros or unknown
        w2c_mat_flat = entry[7:19]  # Should be 12 elements for 3x4 matrix
        w2c_mat = np.array(w2c_mat_flat).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device):
    """
    Computes Plücker embeddings for rays based on camera intrinsics and poses.

    Args:
        K (torch.Tensor): Camera intrinsics tensor of shape [B, V, 4].
        c2w (torch.Tensor): Camera-to-world poses tensor of shape [B, V, 4, 4].
        H (int): Height of the images.
        W (int): Width of the images.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        torch.Tensor: Plücker embeddings of shape [B, V, H, W, 6].
    """
    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, V, H, W, 6)                        # B, V, H, W, 6
    return plucker

class StableVideoDataset(Dataset):
    def __init__(self, 
        video_data_dir, 
        annotations_dir,
        max_num_videos=None,
        frame_height=540, frame_width=960, num_frames=14,
        is_reverse_video=True,
        random_seed=42,
        double_sampling_rate=False,
    ):  
        """
        Initializes the StableVideoDataset.

        Args:
            video_data_dir (str): Path to the directory containing video frames.
            annotations_dir (str): Path to the directory containing annotation files.
            max_num_videos (int, optional): Maximum number of videos to use.
            frame_height (int): Height to resize frames to.
            frame_width (int): Width to resize frames to.
            num_frames (int): Number of frames to sample per video.
            is_reverse_video (bool): Whether to reverse the video frames.
            random_seed (int): Seed for random operations.
            double_sampling_rate (bool): Whether to double the sampling rate.
        """
        self.video_data_dir = video_data_dir
        self.annotations_dir = annotations_dir
        video_names = sorted([video for video in os.listdir(video_data_dir) 
                    if os.path.isdir(os.path.join(video_data_dir, video))])
        
        self.length = min(len(video_names), max_num_videos) if max_num_videos is not None else len(video_names)
        if double_sampling_rate:
            self.sample_frames = num_frames*2-1
            self.sample_stride = 2
        else:
            self.sample_frames = num_frames
            self.sample_stride = 1
        self.video_names = []
        for video in video_names[:self.length]:
            video_dir = os.path.join(self.video_data_dir, video)
            num_frames_in_video = len(glob(os.path.join(video_dir, '*.png')))
            if num_frames_in_video >= self.sample_frames * self.sample_stride:
                self.video_names.append(video)
            else:
                self.length -= 1
                print(f"Skipping video {video} due to insufficient frames.")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pixel_transforms = transforms.Compose([
            transforms.Resize((self.frame_height, self.frame_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.is_reverse_video=is_reverse_video
        np.random.seed(random_seed)

        # Load pose files for each video
        self.pose_files = {}
        for video_name in self.video_names:
            # Assuming the annotations are in files named after the video
            # e.g., annotations_dir/video_name.txt
            pose_file = os.path.join(annotations_dir, f'{video_name}.txt')
            if os.path.exists(pose_file):
                self.pose_files[video_name] = pose_file
            else:
                print(f"No pose file found for video {video_name}")
                self.pose_files[video_name] = None

    def load_cameras(self, video_name, frame_ids):
        """
        Loads camera parameters from the pose file for the given frame_ids.

        Args:
            video_name (str): Name of the video.
            frame_ids (list of int): List of frame IDs.

        Returns:
            list of Camera: List of Camera objects corresponding to the frame IDs.
        """
        pose_file = self.pose_files.get(video_name, None)
        if pose_file is None:
            raise ValueError(f"No pose file available for video {video_name}")

        # Load the pose file
        with open(pose_file, 'r') as f:
            lines = f.readlines()
        # Skip the first line (e.g., header or metadata)
        lines = lines[1:]
        # Build a dictionary mapping frame_id to line entries
        pose_dict = {}
        for line in lines:
            tokens = line.strip().split()
            frame_id = tokens[0]
            entries = [float(x) for x in tokens]
            pose_dict[frame_id] = entries

        # Get the camera parameters for the required frame_ids
        cam_params = []
        for frame_id in frame_ids:
            frame_id_str = str(frame_id)
            if frame_id_str not in pose_dict:
                raise ValueError(f"Frame ID {frame_id_str} not found in pose file for video {video_name}")
            entries = pose_dict[frame_id_str]
            cam_param = Camera(entries)
            cam_params.append(cam_param)
        return cam_params

    def get_batch(self, idx):
        """
        Retrieves a batch of data for a given index.

        Args:
            idx (int): Index of the dataset entry.

        Returns:
            tuple: (pixel_values, plucker_embedding)
        """
        video_name = self.video_names[idx]
        video_dir = os.path.join(self.video_data_dir, video_name)
        # List of image files in the video directory
        video_frame_paths = sorted(glob(os.path.join(video_dir, '*.png')))
        # Extract frame IDs from file names
        frame_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in video_frame_paths]

        # Ensure enough frames are available
        if len(frame_ids) < self.sample_frames * self.sample_stride:
            raise ValueError(f"Not enough frames in video {video_name}")

        start_idx = np.random.randint(len(frame_ids) - self.sample_frames * self.sample_stride + 1)
        selected_indices = list(range(start_idx, start_idx + self.sample_frames * self.sample_stride, self.sample_stride))
        selected_frame_ids = [frame_ids[i] for i in selected_indices]
        selected_frame_paths = [video_frame_paths[i] for i in selected_indices]
        video_frames = [np.asarray(Image.open(frame_path).convert('RGB')).astype(np.float32)/255.0 for frame_path in selected_frame_paths]
        video_frames = np.stack(video_frames, axis=0)
        pixel_values = torch.from_numpy(video_frames.transpose(0, 3, 1, 2))

        # Load camera parameters
        cam_params = self.load_cameras(video_name, selected_frame_ids)

        # Prepare camera intrinsics and poses
        intrinsics = np.asarray([
            [cam_param.fx * self.frame_width,
             cam_param.fy * self.frame_height,
             cam_param.cx * self.frame_width,
             cam_param.cy * self.frame_height]
            for cam_param in cam_params
        ], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics).unsqueeze(0)  # [1, n_frame, 4]

        c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses).unsqueeze(0)  # [1, n_frame, 4, 4]

        # Compute Plücker embeddings
        plucker_embedding = ray_condition(
            intrinsics, c2w, self.frame_height, self.frame_width, device=pixel_values.device
        )[0].permute(0, 3, 1, 2).contiguous()  # [n_frame, 6, H, W]

        return pixel_values, plucker_embedding

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the dataset entry.

        Returns:
            dict: Dictionary containing 'pixel_values', 'plucker_embedding', and 'conditions'.
        """
        while True:
            try:
                pixel_values, plucker_embedding = self.get_batch(idx)
                break

            except Exception as e:
                print(f"Error loading index {idx}: {e}. Selecting a new random index.")
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        conditions = pixel_values[-1]
        if self.is_reverse_video:
            pixel_values = torch.flip(pixel_values, (0,))
            plucker_embedding = torch.flip(plucker_embedding, (0,))

        sample = {
            'pixel_values': pixel_values,           # [n_frame, C, H, W]
            'plucker_embedding': plucker_embedding, # [n_frame, 6, H, W]
            'conditions': conditions                # [C, H, W]
        }
        return sample
