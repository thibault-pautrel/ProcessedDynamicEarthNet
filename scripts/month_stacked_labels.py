import os
import glob
import re
import torch

SUBSAMPLE = 128  # None for full resolution

def build_month_tensor(month_folder, subsample=None):
    daily_files = sorted(glob.glob(os.path.join(month_folder, "data_*.pt")))
    if not daily_files:
        raise FileNotFoundError(f"No 'data_*.pt' files found in {month_folder}")

    example = torch.load(daily_files[0])  # expected shape: [4, H, W]
    _, H, W = example.shape

    H_sub = min(H, subsample) if subsample else H
    W_sub = min(W, subsample) if subsample else W

    stacked_data = torch.zeros((H_sub, W_sub, 4 * len(daily_files)), dtype=torch.float32)
    for t, file in enumerate(daily_files):
        data = torch.load(file)  # shape [4, H, W]
        data_cropped = data[:, :H_sub, :W_sub]
        stacked_data[:, :, t * 4:(t + 1) * 4] = data_cropped.permute(1, 2, 0)

    return stacked_data, H_sub, W_sub

def convert_labels_to_class_indices(label_tensor, H_sub, W_sub):
    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    return class_indices[:H_sub, :W_sub]

def build_pixel_datasets_for_all_months(root_dir, output_dataset_dir, subsample=None):
    """
    For a given tile directory (root_dir), iterates over all month folders (e.g. "2018-01") 
    and builds the pixel dataset for that month, saving the output to output_dataset_dir.
    If an output file for a month already exists, processing is skipped.
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
            features, H_sub, W_sub = build_month_tensor(month_folder, subsample=subsample)

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

def process_all_tiles(raw_root_base, output_dataset_dir, subsample=None):
    """
    Iterates over all planet.* directories within raw_root_base, and for each planet,
    processes all tile folders (matching the "*_*_*" pattern). For each tile, if the monthly
    dataset has already been computed, that month is skipped.
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
                build_pixel_datasets_for_all_months(tile_dir, output_dataset_dir, subsample=subsample)
            except Exception as e:
                print(f"[Error] Failed to process tile {tile_dir}: {e}")

if __name__ == "__main__":
    raw_root_base = "/media/thibault/DynEarthNet"
    output_dataset_dir = "/media/thibault/DynEarthNet/datasets_subsampled"

    print(f"Saving datasets to: {output_dataset_dir}")
    print(f"Subsampling enabled: {SUBSAMPLE if SUBSAMPLE else 'None (full resolution)'}")

    process_all_tiles(raw_root_base, output_dataset_dir, subsample=SUBSAMPLE)
