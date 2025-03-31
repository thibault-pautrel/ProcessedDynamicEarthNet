import os
import glob
import re
import torch
import shutil  # Added to support directory deletion

def build_month_tensor(month_folder):
    """
    Stacks all daily `.pt` feature files (named `data_*.pt`) in a given month directory.

    Each file is expected to be a tensor of shape [4, H, W] (4 bands, height, width).
    The resulting stacked tensor has shape [H, W, 4 * N], where N is the number of days.

    Args:
        month_folder (str): Path to the folder containing daily data_*.pt files.

    Returns:
        stacked_data (torch.Tensor): Stacked tensor of shape [H, W, 4 * N].
        H_sub (int): Height of the cropped area (same as input H).
        W_sub (int): Width of the cropped area (same as input W).
    """

    daily_files = sorted(glob.glob(os.path.join(month_folder, "data_*.pt")))
    if not daily_files:
        raise FileNotFoundError(f"No 'data_*.pt' files found in {month_folder}")

    example = torch.load(daily_files[0])  # expected shape: [4, H, W]
    _, H, W = example.shape

    H_sub = H
    W_sub = W

    stacked_data = torch.zeros((H_sub, W_sub, 4 * len(daily_files)), dtype=torch.float32)
    for t, file in enumerate(daily_files):
        data = torch.load(file)  # shape [4, H, W]
        data_cropped = data[:, :H_sub, :W_sub]
        stacked_data[:, :, t * 4:(t + 1) * 4] = data_cropped.permute(1, 2, 0)

    return stacked_data, H_sub, W_sub

def convert_labels_to_class_indices(label_tensor, H_sub, W_sub):
    """
    Converts a label tensor (multi-class one-hot encoded, shape [C, H, W]) into a 2D tensor of class indices.

    Binary thresholding (value > 127) is applied first, followed by `argmax` over class dimension.

    Args:
        label_tensor (torch.Tensor): Tensor of shape [C, H, W] with raw label values.
        H_sub (int): Desired height crop.
        W_sub (int): Desired width crop.

    Returns:
        torch.Tensor: Tensor of shape [H_sub, W_sub] with integer class indices.
    """

    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    return class_indices[:H_sub, :W_sub]

def build_pixel_datasets_for_all_months(root_dir, output_dataset_dir):
    """
    Processes a tile directory by iterating through each month and building a stacked pixel dataset.

    For each month with N days:
    - Loads daily features: shape [4, H, W] ➝ stacked as [H, W, 4 * N].
    - Loads label raster: shape [C, H, W] ➝ converted to class indices [H, W].

    Saves result as a dictionary:
        {"features": [H, W, 4 * N], "labels": [H, W]}.

    Args:
        root_dir (str): Path to tile folder containing month subfolders.
        output_dataset_dir (str): Path where processed monthly pixel datasets will be saved.
    """

    month_regex = re.compile(r'^\d{4}-\d{2}$')
    
    path_parts = root_dir.strip('/').split(os.sep)
    planet = path_parts[-2]   # e.g., 'planet.13N'
    tile_id = path_parts[-1]  # e.g., '1700_3100_13'

    for entry in sorted(os.listdir(root_dir)):
        month_folder = os.path.join(root_dir, entry)
        if not (os.path.isdir(month_folder) and month_regex.match(entry)):
            continue

        out_folder = os.path.join(output_dataset_dir, 'unet', planet, tile_id, entry)
        os.makedirs(out_folder, exist_ok=True)
        output_file = os.path.join(out_folder, f"pixel_dataset_{entry}.pt")

        if os.path.isfile(output_file):
            print(f"[SKIP] {output_file} already exists. Skipping month {entry}.")
            continue

        print(f"\n[Processing] {month_folder}")
        try:
            features, H_sub, W_sub = build_month_tensor(month_folder)

            label_file = os.path.join(month_folder, "labels", "raster.pt")
            if not os.path.isfile(label_file):
                print(f"[Warning] Label file missing: {label_file}, skipping.")
                continue

            raw_labels = torch.load(label_file)
            labels = convert_labels_to_class_indices(raw_labels, H_sub, W_sub)

            torch.save({"features": features, "labels": labels}, output_file)
            print(f"[Saved] {output_file}")

        except Exception as e:
            print(f"[Error] Failed to process {month_folder}: {e}")

def process_all_tiles(raw_root_base, output_dataset_dir):
    """
    Iterates through all planet.* and tile directories, and for each, processes all months.

    If a month's dataset is already processed and saved, it will be skipped.

    Args:
        raw_root_base (str): Root directory containing all planet.* directories.
        output_dataset_dir (str): Path to save all processed datasets.
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
                build_pixel_datasets_for_all_months(tile_dir, output_dataset_dir)
            except Exception as e:
                print(f"[Error] Failed to process tile {tile_dir}: {e}")

if __name__ == "__main__":
    raw_root_base = "/media/thibault/DynEarthNet/subsampled_data"
    output_dataset_dir = "/media/thibault/DynEarthNet/subsampled_data/datasets"

    print(f"Saving datasets to: {output_dataset_dir}")
    process_all_tiles(raw_root_base, output_dataset_dir)

    # After processing, delete all planet.* directories from subsampled_data
    planet_dirs = sorted(glob.glob(os.path.join(raw_root_base, "planet.*")))
    for planet_dir in planet_dirs:
        try:
            shutil.rmtree(planet_dir)
            print(f"[DELETED] {planet_dir}")
        except Exception as e:
            print(f"[ERROR] Could not delete {planet_dir}: {e}")
