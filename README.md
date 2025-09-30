# Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion

## Method Structure

Our method is organized into **three stages**, with corresponding reference implementation in this repository:

1. **Panorama Generation** — [`PanoDiT/`](PanoDiT)  
   Generate panoramic views using a panoramic diffusion transformer.

2. **Panorama → Perspective Conversion** — [`pano2perspect/`](pano2perspect)  
   Convert equirectangular panoramas into perspective images for subsequent processing.

3. **Video Generation from Perspective** — [`VideoDiffusion/`](VideoDiffusion)  
   Generate temporally consistent video sequences conditioned on the perspective images.

---

## Acknowledgements

- **PanoDiT** is adapted from [FastDiT](https://github.com/chuanyangjin/fast-DiT).  
- **Video Diffusion** is adapted from both [svd_keyframe_interpolation](https://github.com/jeanne-wang/svd_keyframe_interpolation) and [CameraCtrl](https://hehao13.github.io/projects-CameraCtrl/).

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{kang2025look,
  title   = {Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion},
  author  = {Kang, Xueyang and Xiang, Zhengkang and Zhang, Zezheng and Khoshelham, Kourosh},
  journal = {arXiv preprint arXiv:2509.00843},
  year    = {2025}
}
