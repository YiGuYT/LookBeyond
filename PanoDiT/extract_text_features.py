import torch
import os
import numpy as np
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import open_clip
import torch.distributed as dist

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class CustomTextDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the text files.
        """
        self.root_dir = root_dir
        self.text_paths = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.txt')):
                    self.text_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.text_paths)

    def __getitem__(self, idx):
        text_path = self.text_paths[idx]
        with open(text_path, "r") as file:
            text = file.readline().strip()
        return text, text_path


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Extracts text features and saves them as .npy files.
    """
    assert torch.cuda.is_available(), "This script requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load the OpenCLIP model and tokenizer
    model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model.eval()
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')

    # Setup data:
    dataset = CustomTextDataset(args.data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one text file at a time
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Use tqdm for progress bar
    train_steps = 0
    progress_bar = tqdm(loader, desc="Processing", unit="batch")

    for text, path in progress_bar:
        text = tokenizer([text[0]]).to(device)
        
        with torch.no_grad():
            # Encode text to get features
            text_features = model.encode_text(text)
            text_features = text_features.detach().cpu().numpy()

        # Save the features as .npy files
        save_directory = '/'.join(path[0].split("/")[:-2]) + "/sampled_feature_text"
        text_id = path[0].split("/")[-1].split(".")[0]
        os.makedirs(save_directory, exist_ok=True)
        np.save(os.path.join(save_directory, text_id + ".npy"), text_features)

        train_steps += 1
        progress_bar.set_postfix({"Train Steps": train_steps})


if __name__ == "__main__":
    # Default args here will extract text features and save them
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the directory containing text files.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
