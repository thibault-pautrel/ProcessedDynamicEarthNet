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
SUBSAMPLE = 128  # or None for full resolution
LABEL_FILE = 'raster.pt'
NOISE_STD = 1e-4  # Gaussian noise added to avoid ill-conditioning
SPD_EPS = 1e-5    # Minimum eigenvalue threshold for SPD check

#######################################
def convert_labels_to_class_indices(label_tensor):
    """
    Converts a multi-channel label tensor into a single-channel tensor of class indices.
    """
    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    return class_indices

def is_spd(matrix, eps=SPD_EPS):
    symmetric = torch.allclose(matrix, matrix.transpose(-2, -1), atol=1e-6)
    if not symmetric:
        return False
    eigvals = torch.linalg.eigvalsh(matrix)
    return torch.all(eigvals > eps)

def make_spd(matrix, eps=SPD_EPS):
    matrix_sym = 0.5 * (matrix + matrix.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(matrix_sym)
    eigvals_clamped = torch.clamp(eigvals, min=eps)
    fixed_matrix = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1)
    return fixed_matrix

def process_blocks_and_labels(month_folder, dataset_root, planet, tile_id, month, label_file='raster.pt', block_size=64, window_size=21, subsample=None):
    """
    Processes raw satellite data and labels for a single month into blocks of covariance matrices and labels.
    """
    pad = window_size // 2
    pt_files = sorted(glob.glob(os.path.join(month_folder, 'data_*.pt')))
    T = len(pt_files)
    if T == 0:
        raise RuntimeError(f"No data_*.pt files found in {month_folder}")

    example = torch.load(pt_files[0]).permute(1, 2, 0)  # shape: (H, W, B)
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

            print(f"Processing block ({i}:{i_end}, {j}:{j_end}) in month {month}...")
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
                noise = NOISE_STD * torch.randn_like(windows_centered)
                windows_centered_noisy = windows_centered + noise

                cov_matrices = torch.matmul(windows_centered_noisy.transpose(-1, -2), windows_centered_noisy)
                cov_matrices /= (windows.shape[2] - 1)
                
                block_cov = cov_matrices.reshape(-1, cov_matrices.shape[-2], cov_matrices.shape[-1])
                block_cov = 0.5 * (block_cov + block_cov.transpose(-2, -1))
                eigvals, eigvecs = torch.linalg.eigh(block_cov)
                non_spd_mask = eigvals.min(dim=1).values <= SPD_EPS
                n_fixed = non_spd_mask.sum().item()
                if n_fixed > 0:
                    print(f"[Fixed] {n_fixed} matrices in block ({i}:{i_end}, {j}:{j_end})")
                    eigvals_clamped = torch.clamp(eigvals, min=SPD_EPS)
                    block_cov_fixed = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-2, -1)
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
    For each month, if the output folder already exists and contains block files, that month is skipped.
    """
    month_regex = re.compile(r'^\d{4}-\d{2}$')
    path_parts = time_series_dir.strip('/').split(os.sep)
    planet = path_parts[-2]   # e.g., 'planet.13N'
    tile_id = path_parts[-1]  # e.g., '1700_3100_13'

    months = sorted(d for d in os.listdir(time_series_dir)
                    if month_regex.match(d) and os.path.isdir(os.path.join(time_series_dir, d)))
    if not months:
        print(f"[WARNING] No monthly folders found in {time_series_dir}")
        return

    for month in months:
        month_folder = os.path.join(time_series_dir, month)
        output_folder = os.path.join(dataset_root, 'spdnet', planet, tile_id, month)
        # If output_folder exists and has at least one block file, assume month is processed
        if os.path.isdir(output_folder) and len(glob.glob(os.path.join(output_folder, "cov_label_block_*.pt"))) > 0:
            print(f"[SKIP] {output_folder} already processed. Skipping month {month}.")
            continue

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

def process_all_tiles_cov(raw_root_base, dataset_root, label_file='raster.pt', subsample=None, block_size=64, window_size=21):
    """
    Iterates over all planet.* directories within raw_root_base, and for each planet,
    processes all tile directories (matching "*_*_*"). For each tile, it calls the time-series
    processing function. If data has already been computed for a month, that month is skipped.
    """
    planet_dirs = sorted(glob.glob(os.path.join(raw_root_base, "planet.*")))
    if not planet_dirs:
        print(f"No planet.* directories found in {raw_root_base}")
        return

    for planet_dir in planet_dirs:
        tile_dirs = sorted(glob.glob(os.path.join(planet_dir, "*_*_*")))
        if not tile_dirs:
            print(f"No tile directories found in {planet_dir}")
            continue
        print(f"\n[Planet] Processing {planet_dir}: found {len(tile_dirs)} tile(s).")
        for tile_dir in tile_dirs:
            print(f"\n[Tile] Processing tile directory: {tile_dir}")
            try:
                process_time_series_dir(
                    time_series_dir=tile_dir,
                    dataset_root=dataset_root,
                    label_file=label_file,
                    subsample=subsample,
                    block_size=block_size,
                    window_size=window_size
                )
            except Exception as e:
                print(f"[Error] Failed to process tile {tile_dir}: {e}")

if __name__ == '__main__':
    # raw_root_base: the base folder containing all planet.* directories.
    raw_root_base = "/media/thibault/DynEarthNet"
    
    # dataset_root: base folder for processed SPDNet data.
    dataset_root = "/media/thibault/DynEarthNet/datasets"

    process_all_tiles_cov(
        raw_root_base=raw_root_base,
        dataset_root=dataset_root,
        label_file=LABEL_FILE,
        subsample=SUBSAMPLE,
        block_size=BLOCK_SIZE,
        window_size=WINDOW_SIZE
    )
