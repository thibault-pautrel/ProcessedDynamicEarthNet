import os
import glob
import gc
import torch
import rasterio
import numpy as np
import h5py
from datetime import datetime

###############################################################################
# Configurable parameters for memory management
###############################################################################
MAX_TIFS_PER_MONTH = 40
GC_AFTER_EACH_MONTH = True
###############################################################################

def create_h5_file(file_path, data_array, label_array):
    """
    Creates an HDF5 file at the specified path containing:
      - 'data': Satellite features with shape (n_times, H, W, n_features)
      - 'labels': Corresponding pixel-wise class labels with shape (H, W)
    
    Data is chunked for efficient I/O operations.

    Args:
        file_path (str): Output path for the HDF5 file.
        data_array (np.ndarray): 4D array of satellite image features.
        label_array (np.ndarray): 2D array of remapped class labels.
    """
    n_times, H, W, n_features = data_array.shape
    data_chunk_shape = (n_times, 32, 32, n_features)
    labels_chunk_shape = (32, 32)

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("labels", data=label_array, chunks=labels_chunk_shape, dtype=label_array.dtype)
        hf.create_dataset("data", data=data_array, chunks=data_chunk_shape, dtype=data_array.dtype)

    print(f"[HDF5] Created {file_path} with chunk shapes {data_chunk_shape} & {labels_chunk_shape}")


def load_tif_as_tensor(filepath):
    """
    Loads a multi-band GeoTIFF file as a PyTorch tensor.

    Args:
        filepath (str): Path to the .tif file.

    Returns:
        torch.Tensor: Tensor of shape [bands, height, width]
    """
    with rasterio.open(filepath) as src:
        img = src.read()
    return torch.tensor(img)


def build_month_tensor_from_tifs(tif_files):
    """
    Stacks a sequence of daily 4-band TIFs into a monthly tensor.

    Args:
        tif_files (List[str]): Sorted list of daily TIF file paths.

    Returns:
        torch.Tensor: Tensor of shape [n_times, H, W, 4]
    """
    tif_files = sorted(tif_files)
    n_times = len(tif_files)

    example = load_tif_as_tensor(tif_files[0])  # shape [4, H, W]
    bands, H, W = example.shape
    assert bands == 4, "Expected 4 bands per day"

    stacked_data = torch.zeros((n_times, H, W, bands), dtype=torch.float32)
    stacked_data[0] = example.permute(1, 2, 0).to(torch.float32)

    for t, tif_path in enumerate(tif_files[1:], start=1):
        arr = load_tif_as_tensor(tif_path).permute(1, 2, 0).to(torch.float32)
        stacked_data[t] = arr
        del arr
    del example
    return stacked_data


def convert_labels_to_class_indices(label_tensor, H, W):
    """
    Converts a multi-band label tensor into a 2D class index map using
    predefined remapping logic (from 7-class to 4-class).

    Args:
        label_tensor (torch.Tensor): Raw label tensor [bands, H, W].
        H (int): Target height.
        W (int): Target width.

    Returns:
        torch.Tensor: Class indices, shape (H, W)
    """
    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    class_indices = class_indices[:H, :W]

    remapped = torch.full_like(class_indices, -1)
    remapped[class_indices == 0] = 0
    remapped[(class_indices == 1) | (class_indices == 3) | (class_indices == 4) | (class_indices == 6)] = 1
    remapped[class_indices == 2] = 2
    remapped[class_indices == 5] = 3

    if (remapped == -1).any():
        raise ValueError("Some label values were not properly mapped.")
    return remapped


def parse_tiles_from_txt(txt_file):
    """
    Extracts unique tile IDs from a text file listing paths and metadata.

    Args:
        txt_file (str): Path to train/val/test split .txt file.

    Returns:
        Set[str]: A set of tile IDs found in the file.
    """


    tiles = set()
    if not os.path.exists(txt_file):
        print(f"[WARN] {txt_file} not found, skipping.")
        return tiles
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pf_sr_path = parts[0]
            tile_id = os.path.basename(os.path.dirname(pf_sr_path))
            tiles.add(tile_id)
    return tiles


def select_months(month_keys):
    """
    Selects a sparse subset of months to avoid memory overload. This version:
    - Takes every 6th month from sorted list
    - Limits to first 4 entries

    Args:
        month_keys (Iterable[str]): Available month keys.

    Returns:
        List[str]: Subset of months to process.
    """
    return sorted(month_keys)[::][:]


def main():
    raw_root = "/media/thibault/DynEarthNet/raw_data"
    label_base = os.path.join(raw_root, "labels")
    output_base = "/media/thibault/DynEarthNet/datasets/"

    train_tiles = parse_tiles_from_txt(os.path.join(raw_root, "train.txt"))
    val_tiles = parse_tiles_from_txt(os.path.join(raw_root, "val.txt"))
    test_tiles = parse_tiles_from_txt(os.path.join(raw_root, "test.txt"))

    tile_to_split = {}
    for tid in train_tiles:
        tile_to_split[tid] = "train"
    for tid in val_tiles:
        tile_to_split[tid] = "val"
    for tid in test_tiles:
        tile_to_split[tid] = "test"

    for item in os.listdir(raw_root):
        if item.startswith("planet."):
            for dirpath, _, _ in os.walk(os.path.join(raw_root, item)):
                if os.path.basename(dirpath) == "PF-SR":
                    time_series_id = os.path.basename(os.path.dirname(dirpath))
                    if time_series_id not in tile_to_split:
                        continue

                    split_name = tile_to_split[time_series_id]
                    planet_dir = next((p for p in dirpath.split(os.sep) if p.startswith("planet.")), None)
                    planet_id = planet_dir.split('.')[-1] if planet_dir else "unknown"
                    label_folder_name = f"{time_series_id}_{planet_id}"

                    raster_label_base = os.path.join(label_base, label_folder_name, "Labels", "Raster")
                    raster_tile_folder = None
                    if os.path.exists(raster_label_base):
                        folders = os.listdir(raster_label_base)
                        if folders:
                            raster_tile_folder = os.path.join(raster_label_base, folders[0])

                    print(f"\n[PROCESSING TILE] {time_series_id}")
                    daily_files = sorted(glob.glob(os.path.join(dirpath, "*.tif")))
                    monthly_files = {}
                    for file in daily_files:
                        try:
                            dt = datetime.strptime(os.path.basename(file).split('.')[0], "%Y-%m-%d")
                            key = dt.strftime("%Y-%m")
                            monthly_files.setdefault(key, []).append(file)
                        except:
                            continue

                    for month in select_months(monthly_files.keys()):
                        print(f"  [MONTH] {month}")

                        out_dir = os.path.join(output_base, split_name, time_series_id)
                        os.makedirs(out_dir, exist_ok=True)
                        h5_path = os.path.join(out_dir, f"{month}.h5")

                        if os.path.isfile(h5_path):
                            print(f"    [SKIP] Already exists: {h5_path}")
                            continue

                        tif_paths = sorted(monthly_files[month])
                        if len(tif_paths) > MAX_TIFS_PER_MONTH:
                            print(f"    [SKIP] Too many files in {month}")
                            continue


                        try:
                            stacked_data = build_month_tensor_from_tifs(tif_paths)
                            n_times, H, W, _ = stacked_data.shape

                            label_candidates = []
                            if raster_tile_folder:
                                pattern1 = os.path.join(raster_tile_folder, f"*{month.replace('-', '_')}_01.tif")
                                pattern2 = os.path.join(raster_tile_folder, f"*{month}-01.tif")
                                label_candidates = glob.glob(pattern1) + glob.glob(pattern2)

                            if not label_candidates:
                                print(f"    [SKIP] No label for {month}")
                                del stacked_data
                                continue

                            label_tensor = load_tif_as_tensor(label_candidates[0]).to(torch.float32)
                            final_labels = convert_labels_to_class_indices(label_tensor, H, W)

                            out_dir = os.path.join(output_base, split_name, time_series_id)
                            os.makedirs(out_dir, exist_ok=True)
                            h5_path = os.path.join(out_dir, f"{month}.h5")

                            create_h5_file(
                                h5_path,
                                data_array=stacked_data.numpy(),
                                label_array=final_labels.numpy(),
                            )

                            del stacked_data, final_labels, label_tensor

                        except Exception as e:
                            print(f"    [ERROR] Failed to process {month}: {e}")

                        if GC_AFTER_EACH_MONTH:
                            gc.collect()

if __name__ == "__main__":
    main()
