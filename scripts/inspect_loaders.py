import torch
import os
import glob
import sys
import matplotlib
import random
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

###########################################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)


class CovarianceBlockDataset(Dataset):
    def __init__(self, dataset_root, planet_folder=None, block_pattern='cov_label_block_*.pt', max_T=31):
        if planet_folder is not None:
            root_dir = os.path.join(dataset_root, 'spdnet', planet_folder)
        else:
            root_dir = os.path.join(dataset_root, 'spdnet')

        self.block_files = sorted(glob.glob(os.path.join(root_dir, '**', block_pattern), recursive=True))
        self.max_dim = 4 * max_T

        if len(self.block_files) == 0:
            raise ValueError(f"No files found in {root_dir} with pattern {block_pattern}")

        print(f"Loaded {len(self.block_files)} covariance blocks from {root_dir}")

    def __len__(self):
        return len(self.block_files)

    def __getitem__(self, idx):
        block_file = self.block_files[idx]
        block = torch.load(block_file, map_location='cpu')

        cov_block = block['covariance']
        label_block = block['labels']

        H, W, dim, _ = cov_block.shape

        if dim < self.max_dim:
            pad_diff = self.max_dim - dim
            cov_block = F.pad(cov_block, (0, pad_diff, 0, pad_diff), value=0)

        cov_block = cov_block.reshape(-1, self.max_dim, self.max_dim)
        label_block = label_block.reshape(-1)

        return cov_block.float(), label_block.long()

########################################
# Dataloaders
########################################
def get_dataloaders(dataset, batch_size=1, val_split=0.1, test_split=0.1, num_workers=2):
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))

    return train_loader, val_loader, test_loader
################################################################################

def display_multiple_cov_and_label_blocks(test_loader, n_blocks=6, pixel_coords=(0, 0)):
    """
    Display multiple label blocks and the corresponding covariance matrix heatmap
    for a specific pixel in each block.

    Args:
        test_loader (DataLoader): Test dataloader yielding (cov_blocks, label_blocks).
        n_blocks (int): Number of blocks to visualize.
        pixel_coords (tuple): (row, col) coordinate of the pixel in the block for which to show the covariance matrix.
    """
    # Define color mapping (RGB values normalized [0, 1])
    color_mapping = {
        0: (127/255, 127/255, 127/255),  # impervious surface
        1: (189/255, 189/255, 34/255),   # agriculture
        2: (51/255, 204/255, 51/255),    # forest & other vegetation
        3: (0/255, 0/255, 153/255),      # wetlands
        4: (153/255, 102/255, 51/255),   # soil
        5: (51/255, 153/255, 255/255),   # water
        6: (153/255, 204/255, 204/255),  # snow & ice
    }

    fig, axs = plt.subplots(n_blocks, 2, figsize=(12, 6 * n_blocks))

    if n_blocks == 1:
        axs = np.expand_dims(axs, axis=0)

    row_idx, col_idx = pixel_coords
    flat_pixel_idx = row_idx * 64 + col_idx  # Convert 2D to 1D index

    print(f"Showing covariance matrix for pixel coordinates: {pixel_coords} (flattened index: {flat_pixel_idx})")

    count = 0
    for cov_blocks, label_blocks in test_loader:
        if count >= n_blocks:
            break

        label_block = label_blocks[0]          # Shape: (64 * 64,)
        cov_block = cov_blocks[0]              # Shape: (64 * 64, 124, 124)

        # Reshape label block into 64x64
        label_block_2d = label_block.view(64, 64).cpu().numpy()

        # Create color label map
        color_label_map = np.zeros((64, 64, 3))
        for class_index, color in color_mapping.items():
            mask = label_block_2d == class_index
            color_label_map[mask] = color

        # Extract the covariance matrix at the specified pixel
        cov_matrix = cov_block[flat_pixel_idx].cpu().numpy()

        # Plot Covariance Matrix Heatmap
        im0 = axs[count, 0].imshow(cov_matrix, cmap='viridis')
        axs[count, 0].set_title(f'Covariance Matrix\nBlock {count + 1} | Pixel {pixel_coords}')
        axs[count, 0].axis('off')
        fig.colorbar(im0, ax=axs[count, 0], fraction=0.046, pad=0.04)

        # Plot Label Block
        axs[count, 1].imshow(color_label_map)
        axs[count, 1].set_title(f'Label Block {count + 1}')
        axs[count, 1].axis('off')

        count += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_root = "/media/thibault/DynEarthNet/datasets"
    planet_folder = "planet.13N"

    # Load dataset and dataloader
    spdnet_dataset = CovarianceBlockDataset(dataset_root, planet_folder=planet_folder)
    train_loader, val_loader, test_loader = get_dataloaders(
        spdnet_dataset, batch_size=1, val_split=0.1, test_split=0.1, num_workers=2
    )

    # Count label blocks in the dataloader
    num_blocks = len(train_loader.dataset)
    print(f"\nTotal number of label blocks in the train dataloader: {num_blocks}\n")

    # Display the first 4 blocks (each row: label block + covariance matrix for pixel (0, 0))
    display_multiple_cov_and_label_blocks(train_loader, n_blocks=4, pixel_coords=(0, 0))
