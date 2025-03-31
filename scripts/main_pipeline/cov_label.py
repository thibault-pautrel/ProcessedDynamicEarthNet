import os
import re
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc

###########################################
# CONFIGURATION
###########################################
RAW_ROOT = "/media/thibault/DynEarthNet/subsampled_data"
DATASET_ROOT = "/media/thibault/DynEarthNet/subsampled_data/datasets"
# Use the planet.* folders from the unet subdirectory only
UNET_DATASET_ROOT = os.path.join(DATASET_ROOT, "unet")
WINDOW_SIZE = 11
PAD = WINDOW_SIZE // 2  # 5
NOISE_STD = 1e-6
SPD_EPS = 1e-5
ALPHA = 1e-4

###########################################
# SPD Covariance Matrix Function
###########################################
def make_spd(matrix, eps=SPD_EPS):
    """
    Ensures a matrix is Symmetric Positive Definite (SPD) by eigenvalue clamping.

    Args:
        matrix (torch.Tensor): Input square matrix of shape [..., C, C].
        eps (float): Minimum eigenvalue threshold to enforce SPD condition.

    Returns:
        torch.Tensor: SPD matrix of the same shape [..., C, C].
    """
    matrix_sym = 0.5 * (matrix + matrix.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(matrix_sym)
    eigvals_clamped = torch.clamp(eigvals, min=eps)
    fixed_matrix = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1)
    fixed_matrix = 0.5 * (fixed_matrix + fixed_matrix.transpose(-2, -1))
    return fixed_matrix

###########################################
# Full Tile-Month Processor
###########################################
def process_one_month_tile(planet, tile_id, month, alpha=ALPHA):
    """
    Processes a single (planet, tile, month) dataset and computes per-pixel SPD covariance matrices
    from a local spatial window in the feature tensor.

    Loads:
        - features: Tensor of shape [H, W, C]
        - labels: Tensor of shape [H, W]

    For each pixel (excluding border padding), computes a covariance matrix over a 
    WINDOW_SIZE x WINDOW_SIZE spatial patch resulting in:
        - covariance tensor: [H - 2*PAD, W - 2*PAD, C, C]
        - labels: [H - 2*PAD, W - 2*PAD]

    Saves:
        A dictionary with "covariance" and "labels" tensors to output path.

    Args:
        planet (str): Planet folder name, e.g. "planet.10N".
        tile_id (str): Tile ID, e.g. "1700_3100_13".
        month (str): Month in "YYYY-MM" format.
        alpha (float): Diagonal regularization factor for SPD.
    """
    input_file = os.path.join(DATASET_ROOT, "unet", planet, tile_id, month, f"pixel_dataset_{month}.pt")
    if not os.path.exists(input_file):
        print(f"[SKIP] Missing {input_file}")
        return

    out_dir = os.path.join(DATASET_ROOT, "spdnet_monthly", planet, tile_id, month)
    out_file = os.path.join(out_dir, f"cov_label_{month}.pt")
    if os.path.exists(out_file):
        print(f"[SKIP] Already processed {out_file}")
        return

    print(f"[PROCESSING] {planet}/{tile_id}/{month}")
    os.makedirs(out_dir, exist_ok=True)

    data = torch.load(input_file)
    features = data['features']  # shape [H, W, C]
    labels = data['labels']      # shape [H, W]

    # Rearrange features for unfolding
    C = features.shape[-1]
    features = features.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    # Pad so each pixel has a full neighborhood
    features_padded = F.pad(features, (PAD, PAD, PAD, PAD), mode='reflect')

    # Compute windows all at once
    with torch.no_grad():
        features_padded = features_padded.double()
        windows = features_padded.unfold(2, WINDOW_SIZE, 1).unfold(3, WINDOW_SIZE, 1)  
        # windows shape: [1, C, H_padded, W_padded, WINDOW_SIZE, WINDOW_SIZE]

        windows = (
            windows.squeeze(0)
                   .permute(1, 2, 0, 3, 4) 
                   .reshape(windows.shape[2], windows.shape[3], C, -1)
                   .permute(0, 1, 3, 2)
        )
        # Final shape: [H, W, WINDOW_SIZE*WINDOW_SIZE, C]

        # Center and add small noise
        windows_centered = windows - windows.mean(dim=2, keepdim=True)
        windows_centered += NOISE_STD * torch.randn_like(windows_centered)
        windows_centered = windows_centered.double()
        # Covariance for each pixel
        cov_tensor = torch.matmul(
            windows_centered.transpose(-1, -2), windows_centered
        ) / (WINDOW_SIZE * WINDOW_SIZE - 1)
        C = cov_tensor.shape[-1]
        cov_tensor = cov_tensor.double()
        
        eyeC = torch.eye(C, dtype=cov_tensor.dtype, device=cov_tensor.device)
        cov_tensor = cov_tensor + alpha * eyeC.view(1, 1, C, C)

        for i in range(cov_tensor.shape[0]):
            for j in range(cov_tensor.shape[1]):
                cov_tensor[i, j] = make_spd(cov_tensor[i, j])

    cov_tensor = cov_tensor.float()
    # Crop labels to remove edges
    labels_cropped = labels[PAD:-PAD, PAD:-PAD].contiguous()
    # Crop edges from cov_tensor to match the shape of labels
    cov_tensor_cropped = cov_tensor[PAD:-PAD, PAD:-PAD, :, :].contiguous()

    # Save the covariance and labels dictionary
    torch.save(
        {
            "covariance": cov_tensor_cropped,      # [H - 2*PAD, W - 2*PAD, C, C]
            "labels": labels_cropped,              # [H - 2*PAD, W - 2*PAD]
        },
        out_file
    )

    print(f"[SAVED] {out_file}")
    gc.collect()

###########################################
# Processing all planet directories in unet
###########################################
def process_all_planets(selected_planet=None, selected_tile=None, selected_month=None):
    """
    Iterates over all planet folders and tiles located in the unet directory and processes each month 
    using `process_one_month_tile()`. Supports optional filtering by planet, tile ID, or month.

    Args:
        selected_planet (str, optional): If set, only this planet is processed.
        selected_tile (str, optional): If set, only this tile is processed.
        selected_month (str, optional): If set, only this month is processed.
    """
    # Get planet folders exclusively from the unet subdirectory
    planet_dirs = glob.glob(os.path.join(UNET_DATASET_ROOT, "planet.*"))
    months = [f"2018-{m:02d}" for m in range(1, 13)] + [f"2019-{m:02d}" for m in range(1, 13)]

    for planet_path in planet_dirs:
        planet = os.path.basename(planet_path)
        if selected_planet and planet != selected_planet:
            continue

        tile_dirs = glob.glob(os.path.join(planet_path, "*_*_*"))
        for tile_path in tile_dirs:
            tile_id = os.path.basename(tile_path)
            if selected_tile and tile_id != selected_tile:
                continue

            for month in tqdm(months, desc=f"{planet}/{tile_id} months"):
                if selected_month and month != selected_month:
                    continue
                process_one_month_tile(planet, tile_id, month)

if __name__ == "__main__":
    process_all_planets()
