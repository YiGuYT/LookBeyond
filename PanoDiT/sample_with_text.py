import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion_warp import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models.DiT_with_text import DiT_models
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def process_image(image_path, text_path, model, diffusion, vae, device, output_folder, image_size):
    # Load image and text features
    image_features = torch.from_numpy(np.load(image_path)).to(device)
    text_features = torch.from_numpy(np.load(text_path)).to(device)
    width = image_features.shape[3]
    shift_amount = width // 4
    image_features = torch.roll(image_features, shifts=-shift_amount, dims=3)
    # Desired patch size (nxn)
    n = 32
    # Calculate the start and end indices for the center patch
    start_h = (image_features.size(2) - n) // 2
    start_w = (image_features.size(3) - n) // 2
    end_h = start_h + n
    end_w = start_w + n

    # Create a mask with the same size as the original tensor
    mask = torch.zeros_like(image_features, dtype=torch.bool)
    mask[:, :, start_h:end_h, start_w:end_w] = True

    # Pad the mask from [1, 4, n, n] to [1, 4, 64, 64]
    pad_size = (0, 0, 16, 16)
    padded_mask = F.pad(mask, pad=pad_size, mode='constant', value=False)

    latent_size = image_size // 8
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    z[padded_mask] = image_features[mask]
    model_kwargs = dict(text=text_features.unsqueeze(0))  # Ensure correct shape for text input

    # Sample images
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, padded_mask, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    width = samples.shape[3]
    shift_amount = width // 4
    samples = torch.roll(samples, shifts=shift_amount, dims=3)
    samples = samples[:, :, 16:48, :]
    samples = vae.decode(samples / 0.18215).sample

    # Save the image with the same name as the feature file but with .png extension
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.npy', '.png'))
    save_image(samples, output_path, nrow=8, normalize=True, value_range=(-1, 1))

def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", cache_dir="hub").to(device)

    # Process the individual image and text feature files
    image_path = args.image_path
    text_path = args.text_path

    # Check if the image and text feature files exist
    if os.path.exists(image_path) and os.path.exists(text_path):
        process_image(image_path, text_path, model, diffusion, vae, device, args.output_folder, args.image_size)
    else:
        if not os.path.exists(image_path):
            print(f"Image feature file not found: {image_path}")
        if not os.path.exists(text_path):
            print(f"Text feature file not found: {text_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-path", required=True, type=str, help="Path to image feature .npy file")
    parser.add_argument("--text-path", required=True, type=str, help="Path to text feature .npy file")
    parser.add_argument("--output-folder", type=str, default="output_images", help="Directory to save output images")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify CUDA device (e.g., 'cuda:0', 'cuda:1')")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model)")
    args = parser.parse_args()
    main(args)
