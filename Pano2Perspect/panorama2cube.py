import os
import cv2 
import lib.Equirec2Perspec as E2P
import glob
import argparse
import numpy as np

def images_to_video(images_folder, output_video_path, fps=30):
    # Get list of image files in the folder, sorted alphabetically
    images = sorted([
        img for img in os.listdir(images_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    if not images:
        raise ValueError("No images found in the specified folder.")

    # Path to the first image to get the size
    first_image_path = os.path.join(images_folder, images[0])
    frame = cv2.imread(first_image_path)

    if frame is None:
        raise ValueError(f"Could not read the first image: {first_image_path}")

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(images_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: {image_path} could not be read and is skipped.")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")


def panorama2cube(input_dir, output_dir, cube_size=512, crop_size=512):
    """
    Convert equirectangular panoramas to cube faces.
    After generating each 256x256 face, we crop it to 225x225 by center cropping.
    """
    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a sorted list of all image file paths in the input directory
    all_images = sorted(glob.glob(os.path.join(input_dir, '*.*')))

    print(f"Found {len(all_images)} images:")
    for img_path in all_images:
        print(f" - {img_path}")

    for img_path in all_images:
        # Extract the base name of the image (e.g., 'image1.png')
        base_name = os.path.basename(img_path)
        # Remove the file extension to get the image name (e.g., 'image1')
        image_name = os.path.splitext(base_name)[0]

        # Load the equirectangular image
        equ = E2P.Equirectangular(img_path)

        # Define the output directory for this image
        image_output_dir = os.path.join(output_dir, image_name)
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        # Define start and end angles in degrees
        theta_start = 180  # Right 30% of the panorama
        theta_end   = -180 # Left 30% of the panorama

        # Generate 50 angles between theta_start and theta_end
        angles = np.linspace(theta_start, theta_end, 12)

        images = []
        for idx, angle in enumerate(angles):
            # Generate the 256x256 perspective image
            img_256 = equ.GetPerspective(90, angle, 0, cube_size, cube_size)

            # ---- Crop from 256x256 down to crop_size x crop_size (default 225) ----
            if crop_size > 0 and crop_size < cube_size:
                start = (cube_size - crop_size) // 2
                end   = start + crop_size
                img_cropped = img_256[start:end, start:end]
            else:
                # If crop_size is not valid or zero, keep original
                img_cropped = img_256
            # -----------------------------------------------------------------------
            images.append(img_cropped)

            # Define the output file path with zero-padded index
            output_filename = f"{idx:05d}.png"
            output_path = os.path.join(image_output_dir, output_filename)
            cv2.imwrite(output_path, img_cropped)

        # Optionally, create a video from the generated images
        video_filename = f"{image_name}.mp4"
        video_path = os.path.join(image_output_dir, video_filename)
        images_to_video(image_output_dir, video_path)

    print("Panorama to cube conversion (with optional cropping) completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # We only care about "panorama2cube" for this example
    parser.add_argument('--input',  type=str, default='./panorama', help="Input directory path.")
    parser.add_argument('--output', type=str, default='./output',   help="Output directory path.")
    # Let user specify the final face size (225 by default)
    parser.add_argument('--crop_size', type=int, default=1024, 
                        help="Final output face size (width and height).")

    config = parser.parse_args()

    # Run panorama2cube with the user-specified (or default) crop_size
    panorama2cube(config.input, config.output, cube_size=1024, crop_size=0)
