import os
import glob
import re
import gc
import torch
import torch.nn.functional as F

#######################################
# CONFIGURATION
#######################################
WINDOW_SIZE = 21
BLOCK_SIZE = 64
SUBSAMPLE = 128  # None for full resolution
LABEL_FILE = 'raster.pt'
NOISE_STD = 1e-4  #Gaussian noise added to avoid ill-conditioning
SPD_EPS=1e-5 # Minimum eigenvalue threshold for SPD check

#######################################
def convert_labels_to_class_indices(label_tensor):
    """
    Converts a multi-channel label tensor into a single-channel tensor of class indices.

    Args:
        label_tensor (torch.Tensor): A tensor of shape (C, H, W), where C is the number of classes.
                                     Each channel is expected to have binary values (0 or 255).

    Returns:
        torch.Tensor: A 2D tensor of shape (H, W) where each pixel contains the class index
                      corresponding to the channel with the highest value.
    """
    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    return class_indices

def is_spd(matrix, eps=SPD_EPS):
    """
    Checks whether a given matrix is Symmetric Positive Definite (SPD).

    Args:
        matrix (torch.Tensor): A square matrix of shape (D, D).
        eps (float): Minimum threshold for eigenvalues to consider the matrix SPD.

    Returns:
        bool: True if the matrix is symmetric and all eigenvalues are greater than eps, False otherwise.
    """
    # Check symmetry
    symmetric = torch.allclose(matrix, matrix.transpose(-2, -1), atol=1e-6)
    if not symmetric:
        return False

    # Check positive definiteness (eigenvalues > eps)
    eigvals = torch.linalg.eigvalsh(matrix)
    positive_definite = torch.all(eigvals > eps)
    return positive_definite

def make_spd(matrix, eps=SPD_EPS):
    """
    Forces a matrix to be Symmetric Positive Definite (SPD) by:
    - Symmetrizing it
    - Clamping eigenvalues below a given threshold

    Args:
        matrix (torch.Tensor): A square matrix of shape (D, D).
        eps (float): Minimum threshold for eigenvalues to ensure positive definiteness.

    Returns:
        torch.Tensor: A symmetrized SPD matrix of shape (D, D).
    """
    # Symmetrize the matrix
    matrix_sym = 0.5 * (matrix + matrix.transpose(-2, -1))

    # Eigen decomposition and clamp eigenvalues
    eigvals, eigvecs = torch.linalg.eigh(matrix_sym)
    eigvals_clamped = torch.clamp(eigvals, min=eps)

    # Reconstruct SPD matrix
    fixed_matrix = (eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1))
    return fixed_matrix


def process_blocks_and_labels(month_folder, dataset_root, planet, tile_id, month, label_file='raster.pt', block_size=64, window_size=21, subsample=None):
    """
    Processes raw satellite data and labels for a single month into blocks of covariance matrices and labels.

    This function:
    - Loads raw time-series data (T temporal frames) and label raster.
    - Subsamples the data spatially if specified.
    - Splits the data into blocks (default 64x64 pixels per block).
    - Computes covariance matrices for each pixel in the block, using a sliding window of `window_size`.
    - Adds noise for numerical stability and enforces SPD properties on covariance matrices.
    - Saves the processed covariance matrices and label blocks as `.pt` files.

    Args:
        month_folder (str): Path to the directory containing time-series data and labels for one month.
        dataset_root (str): Root directory where the processed SPDNet data should be stored.
        planet (str): Identifier for the planet dataset (e.g., 'planet.13N').
        tile_id (str): Identifier for the tile within the planet dataset.
        month (str): Month string in 'YYYY-MM' format.
        label_file (str): Filename of the label raster within the labels directory. Default is 'raster.pt'.
        block_size (int): Spatial size of each block (default is 64).
        window_size (int): Size of the sliding window used for covariance computation (default is 21).
        subsample (int or None): Spatial subsampling factor for H and W dimensions (default is None for no subsampling).

    Raises:
        RuntimeError: If data files or label files are missing for the specified month.
    """
    pad = window_size // 2
    pt_files = sorted(glob.glob(os.path.join(month_folder, 'data_*.pt')))
    T = len(pt_files)
    
    if T == 0:
        raise RuntimeError(f"No data_*.pt files found in {month_folder}")

    example = torch.load(pt_files[0]).permute(1, 2, 0)
    H, W, B = example.shape
    H_sub, W_sub = (H, W) if subsample is None else (min(H, subsample), min(W, subsample))

    label_path = os.path.join(month_folder, 'labels', label_file)
    if not os.path.isfile(label_path):
        raise RuntimeError(f"Missing label file: {label_path}")

    raw_labels = torch.load(label_path)
    label_indices = convert_labels_to_class_indices(raw_labels)
    label_indices = label_indices[:H_sub, :W_sub]

    output_folder = os.path.join(dataset_root, 'spdnet', planet, tile_id, month)
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, H_sub, block_size):
        i_end = min(i + block_size, H_sub)
        for j in range(0, W_sub, block_size):
            j_end = min(j + block_size, W_sub)

            out_file = os.path.join(output_folder, f'cov_label_block_{i}_{j}.pt')
            if os.path.isfile(out_file):
                print(f"[Skipping] Already processed block: {out_file}")
                continue

            print(f"Processing block ({i}:{i_end}, {j}:{j_end})...")

            block_h = i_end - i
            block_w = j_end - j

            block_data = torch.zeros((T, block_h + 2 * pad, block_w + 2 * pad, B), dtype=torch.float64)

            try:
                for t, f in enumerate(pt_files):
                    data = torch.load(f)
                    padded_data = F.pad(data, (pad, pad, pad, pad), mode='reflect')
                    padded_data = padded_data.permute(1, 2, 0)
                    block_data[t] = padded_data[i:i_end + 2 * pad, j:j_end + 2 * pad, :]

                block_data_stacked = block_data.permute(1, 2, 0, 3).reshape(block_h + 2 * pad, block_w + 2 * pad, T * B)

                windows = block_data_stacked.unfold(0, window_size, 1).unfold(1, window_size, 1)
                windows = windows.contiguous().view(block_h, block_w, -1, T * B)

                windows_centered = windows - windows.mean(dim=2, keepdim=True)

                # Add noise before covariance computation
                noise = NOISE_STD * torch.randn_like(windows_centered)
                windows_centered_noisy = windows_centered + noise

                cov_matrices = torch.matmul(windows_centered_noisy.transpose(-1, -2), windows_centered_noisy)
                cov_matrices /= (windows.shape[2] - 1)
                
                # Validate SPD property and fix if necessary (vectorized)
                block_cov = cov_matrices.reshape(-1, cov_matrices.shape[-2], cov_matrices.shape[-1])  # Shape: (N, D, D)

                # Symmetrize all matrices
                block_cov = 0.5 * (block_cov + block_cov.transpose(-2, -1))

                # Eigen-decomposition (batched)
                eigvals, eigvecs = torch.linalg.eigh(block_cov)  # Shapes: (N, D), (N, D, D)

                # Find matrices with problematic eigenvalues
                non_spd_mask = eigvals.min(dim=1).values <= SPD_EPS
                n_fixed = non_spd_mask.sum().item()

                if n_fixed > 0:
                    print(f"[Fixed] {n_fixed} matrices in block ({i}:{i_end}, {j}:{j_end})")

                    # Clamp eigenvalues below threshold
                    eigvals_clamped = torch.clamp(eigvals, min=SPD_EPS)

                    # Reconstruct SPD matrices: eigvecs @ diag(clamped eigvals) @ eigvecs.T
                    block_cov_fixed = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1)

                    # Reshape back to original block shape
                    cov_matrices = block_cov_fixed.view(block_h, block_w, cov_matrices.shape[-2], cov_matrices.shape[-1])

                label_block = label_indices[i:i_end, j:j_end]

                torch.save({'covariance': cov_matrices, 'labels': label_block}, out_file)
                print(f"[Saved] {out_file}")

            except Exception as e:
                print(f"[Error] Failed block ({i},{j}): {e}")

            del block_data, block_data_stacked, windows, windows_centered, windows_centered_noisy, cov_matrices, label_block
            gc.collect()

def process_time_series_dir(time_series_dir, dataset_root, label_file='raster.pt', subsample=None, block_size=64, window_size=21):
    """
    Processes an entire time series directory containing monthly satellite data folders.

    This function:
    - Iterates through all months available in a tile directory.
    - For each month, invokes `process_blocks_and_labels()` to process data and labels.
    - Saves processed data into the specified dataset root directory, under `spdnet/{planet}/{tile}/{month}`.

    Args:
        time_series_dir (str): Path to a directory containing multiple monthly folders for a given tile.
                               Example: '/path/to/planet.13N/1700_3100_13'
        dataset_root (str): Root directory where the processed SPDNet data should be stored.
        label_file (str): Filename of the label raster within the labels directory. Default is 'raster.pt'.
        subsample (int or None): Spatial subsampling factor for H and W dimensions (default is None for no subsampling).
        block_size (int): Spatial size of each block (default is 64).
        window_size (int): Size of the sliding window used for covariance computation (default is 21).

    Warnings:
        Will print a warning if no valid month folders are found within `time_series_dir`.
    """

    month_regex = re.compile(r'^\d{4}-\d{2}$')
    
    # Parse planet and tile 
    path_parts = time_series_dir.strip('/').split('/')
    planet = path_parts[-2]   # e.g., 'planet.13N'
    tile_id = path_parts[-1]  # e.g., '1700_3100_13'

    months = sorted(
        d for d in os.listdir(time_series_dir)
        if month_regex.match(d) and os.path.isdir(os.path.join(time_series_dir, d))
    )

    if not months:
        print(f"[WARNING] No monthly folders found in {time_series_dir}")
        return

    for month in months:
        month_folder = os.path.join(time_series_dir, month)
        print(f"\n==> Processing {month_folder}")

        try:
            process_blocks_and_labels(
                month_folder=month_folder,
                dataset_root=dataset_root,
                planet=planet,
                tile_id=tile_id,
                month=month,
                label_file=label_file,
                block_size=block_size,
                window_size=window_size,
                subsample=subsample
            )
        except Exception as e:
            print(f"[Error] Month {month_folder}: {e}")

if __name__ == '__main__':
    # Input raw tile folder (adjust as needed)
    time_series_dir = '/media/thibault/DynEarthNet/planet.15N/2006_3280_13'
    
    # Output dataset folder (base folder for dataset storage)
    output_dataset_dir = '/media/thibault/DynEarthNet/datasets'

    process_time_series_dir(
        time_series_dir=time_series_dir,
        dataset_root=output_dataset_dir,
        label_file=LABEL_FILE,
        subsample=SUBSAMPLE,
        block_size=BLOCK_SIZE,
        window_size=WINDOW_SIZE
    )
