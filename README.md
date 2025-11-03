# Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion

<p align="center">
  <img src="assets/small.gif" width="40%" style="margin: 0 1%;" />
</p>

## Method Structure

Our method is organized into **three stages**, with corresponding reference implementation in this repository:

1. **Panorama Generation** — [`PanoDiT/`](PanoDiT)  
   Generate panoramic views using a panoramic diffusion transformer.

2. **Panorama → Perspective Conversion** — [`Pano2Perspect/`](Pano2Perspect)  
   Convert equirectangular panoramas into perspective images for subsequent processing.

3. **Video Generation from Perspective** — [`VideoDiffusion/`](VideoDiffusion)  
   Generate temporally consistent video sequences conditioned on the perspective images.

---

## Data Preparation

The training process is divided into two stages based on dataset domains:

- **Stage 1** is trained using the [Matterport3D](https://niessner.github.io/Matterport/) dataset.  
- **Stage 2** is trained using the [RealEstate10K](https://google.github.io/realestate10k/) dataset.  

---

## Environment

The corresponding environments are stored in the `requirements.txt` files located in the two repositories:

- [`PanoDiT/requirements.txt`](PanoDiT/requirements.txt)  
- [`VideoDiffusion/requirements.txt`](VideoDiffusion/requirements.txt)

Please refer to each for detailed dependency setup.

---

## Training

We follow a two-stage training pipeline aligned with the dataset domains:

### Stage 1 — Panorama Generation (Matterport3D)

Train the panoramic diffusion transformer using both **image features** and **text embeddings**.

**Feature extraction**
Use `extract_features.py` and `extract_text_features.py` beforehand to accelerate training.

```bash
python train_pano_with_text.py \
  --feature-path /path/to/features \
  --results-dir ./results_pano_text \
  --model DiT-XL/2 \
  --image-size 256 \
  --global-batch-size 256 \
  --num-workers 4 \
  --log-every 100 \
  --ckpt-every 50000
```

### Stage 2 — Video Generation (RealEstate10K)
Train the video diffusion model on RealEstate10K, optionally conditioning on generated perspective frames produced from Stage 1 panoramas.

```bash
cd VideoDiffusion
python train_video_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
  --variant fp16 \
  --train-data-dir /path/to/RealEstate10K/frames \
  --train-annotations-dir /path/to/RealEstate10K/annotations \
  --output_dir ./runs/video_lora \
  --num_train_epochs 10 \
  --max_train_steps 200000 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 1000 \
  --mixed_precision fp16 \
  --dataloader_num_workers 8 \
  --num_frames 14 \
  --checkpointing_steps 5000 \
  --checkpoints_total_limit 5 \
  --enable_xformers_memory_efficient_attention \
  --allow_tf32 \
  --seed 42
```

---

## Sampling / Inference

### 1) Generate a Panorama
```bash
python sample_with_text.py \
  --model DiT-XL/2 \
  --image-path ./features/sample_000001.npy \
  --text-path ./features/sample_000001_text.npy \
  --ckpt ./checkpoints/pano.ckpt \
  --vae ema \
  --image-size 512 \
  --num-sampling-steps 1000 \
  --output-folder ./outputs/pano \
  --device cuda:0
```

---

### 2) Convert Panorama → Perspective Frames
```bash
python panorama2cube.py \
  --input ../PanoDiT/outputs/pano/*.png \
  --output ./outputs/perspective_frames \
```

---

### 3) Generate Video from Perspective Frames
```bash
python videogen.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
  --checkpoint_dir ./runs/video_lora \
  --frame1_path ../pano2perspect/outputs/perspective/sample_000001/000000.png \
  --frame2_path ../pano2perspect/outputs/perspective/sample_000001/000011.png \
  --pose_file ./metadata/pose.txt \
  --out_path ./outputs/video/sample_000001.mp4 \
  --num_frames 14 \
  --num_inference_steps 50 \
  --fps 24 \
  --device cuda:0
```

---

## Acknowledgements

- **PanoDiT** is adapted from [FastDiT](https://github.com/chuanyangjin/fast-DiT).  
- **Video Diffusion** is adapted from both [svd_keyframe_interpolation](https://github.com/jeanne-wang/svd_keyframe_interpolation) and [CameraCtrl](https://hehao13.github.io/projects-CameraCtrl/).

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{kang2025look,
  title={Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion},
  author={Kang, Xueyang and Xiang, Zhengkang and Zhang, Zezheng and Khoshelham, Kourosh},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={9375--9384},
  year={2025}
}
}
