import os
import glob
import gc
import torch
import rasterio
from datetime import datetime

###############################################################################
# Configurable parameters for memory management
###############################################################################
MAX_TIFS_PER_MONTH = 40  # skip months that have more than 40 daily TIFs
USE_HALF_PRECISION = False  # store stacked data in float16 instead of float32
GC_AFTER_EACH_MONTH = True  # force garbage collection after each month
###############################################################################

def load_tif_as_tensor(filepath):
    with rasterio.open(filepath) as src:
        img = src.read()
    # img shape: [bands, height, width]
    return torch.tensor(img)  # default float64 if it has floats, but we’ll cast later

def build_month_tensor_from_tifs(tif_files):
    """
    Stack daily TIFs into a single [H, W, 4*D] Tensor.
    Potentially store in half precision to reduce memory usage.
    """
    tif_files = sorted(tif_files)
    print(f"  [INFO] Stacking {len(tif_files)} daily files...")

    # Load the first day's data to see shape
    example = load_tif_as_tensor(tif_files[0])  # shape [4, H, W]
    bands, H, W = example.shape

    # Decide on dtype
    dtype = torch.float16 if USE_HALF_PRECISION else torch.float32
    stacked_data = torch.zeros((H, W, bands * len(tif_files)), dtype=dtype)

    # Insert first day's data
    stacked_data[:, :, 0:4] = example.permute(1, 2, 0).to(dtype)

    # Insert subsequent days
    for t, tif_path in enumerate(tif_files[1:], start=1):
        data = load_tif_as_tensor(tif_path)
        data = data.permute(1, 2, 0).to(dtype)
        stacked_data[:, :, t*4:(t+1)*4] = data
        del data  # free right away
    del example
    return stacked_data, H, W

def convert_labels_to_class_indices(label_tensor, H, W):
    """
    Example label remapping logic.  
    Adjust as needed if your classes differ.
    """
    label_tensor = (label_tensor > 127).long()  # 0 or 1
    class_indices = torch.argmax(label_tensor, dim=0)
    class_indices = class_indices[:H, :W]

    remapped = torch.full_like(class_indices, -1)
    remapped[class_indices == 0] = 0
    remapped[(class_indices == 1) | (class_indices == 3) | 
             (class_indices == 4) | (class_indices == 6)] = 1
    remapped[class_indices == 2] = 2
    remapped[class_indices == 5] = 3

    if (remapped == -1).any():
        raise ValueError("Some label values were not properly mapped.")
    return remapped

def select_months(month_keys):
    """
    Returns a subset of months to process, skipping 3 out of 4, 
    and limiting to first 6 unique months. 
    Adjust as needed to limit memory usage further.
    """
    return sorted(month_keys)[::6][:4]

def parse_tiles_from_txt(txt_file):
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

def process_tile_series(series_path, label_base_path, output_base, tile_to_split):
    time_series_id = os.path.basename(os.path.dirname(series_path))
    if time_series_id not in tile_to_split:
        print(f"[SKIP] {time_series_id} not in any split. Skipping.")
        return

    split_name = tile_to_split[time_series_id]
    planet_dir = next((p for p in series_path.split(os.sep) if p.startswith("planet.")), None)
    planet_id = planet_dir.split('.')[-1] if planet_dir else "unknown"
    label_folder_name = f"{time_series_id}_{planet_id}"

    raster_label_base = os.path.join(label_base_path, label_folder_name, "Labels", "Raster")
    raster_tile_folder = None
    if os.path.exists(raster_label_base):
        folders = os.listdir(raster_label_base)
        if folders:
            raster_tile_folder = os.path.join(raster_label_base, folders[0])

    out_dir = os.path.join(output_base, split_name, time_series_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[PROCESSING TILE] {series_path}")
    daily_files = sorted(glob.glob(os.path.join(series_path, "*.tif")))
    if not daily_files:
        print("  [WARNING] No .tif files found. Skipping.")
        return

    # Group daily files by year-month
    monthly_files = {}
    for file in daily_files:
        basename = os.path.basename(file)
        try:
            dt = datetime.strptime(basename.split('.')[0], "%Y-%m-%d")
            key = dt.strftime("%Y-%m")
            monthly_files.setdefault(key, []).append((dt, file))
        except:
            continue

    # Pick which months to process
    chosen_months = select_months(monthly_files)
    for month in chosen_months:
        print(f"  [MONTH] {month}")
        if month not in monthly_files:
            print(f"    [WARNING] No data for {month}. Skipping.")
            continue

        # Construct output path for this month
        save_dir = os.path.join(out_dir, month)
        save_path = os.path.join(save_dir, f"pixel_dataset_{month}.pt")

        # If it's already on disk, skip
        if os.path.isfile(save_path):
            print(f"    [SKIP] Already computed: {save_path}")
            continue

        # Extract the sorted TIF paths for that month
        monthly_data_sorted = sorted(monthly_files[month], key=lambda x: x[0])
        tif_paths = [f[1] for f in monthly_data_sorted]
        if len(tif_paths) > MAX_TIFS_PER_MONTH:
            print(f"    [SKIP] Month {month} has {len(tif_paths)} daily TIFs > {MAX_TIFS_PER_MONTH}. Skipping to save memory.")
            continue

        try:
            features, H, W = build_month_tensor_from_tifs(tif_paths)

            # Attempt to find a matching label file
            label_candidates = []
            if raster_tile_folder:
                pattern1 = os.path.join(raster_tile_folder, f"*{month.replace('-', '_')}_01.tif")
                pattern2 = os.path.join(raster_tile_folder, f"*{month}-01.tif")
                label_candidates = glob.glob(pattern1) + glob.glob(pattern2)

            if not label_candidates:
                print(f"    [WARNING] No label file found for {month}. Skipping.")
                # Clean up memory
                del features
                gc.collect()
                continue

            label_tensor = load_tif_as_tensor(label_candidates[0])
            label_tensor = label_tensor.to(torch.float32)  # or half if desired
            labels = convert_labels_to_class_indices(label_tensor, H, W)
            del label_tensor

            os.makedirs(save_dir, exist_ok=True)  # ensure parent folder
            torch.save({"features": features, "labels": labels}, save_path)
            print(f"    [SAVED] {save_path}")

            # Clean up
            del features
            del labels

        except Exception as e:
            print(f"    [ERROR] Failed to process {month}: {e}")

        # Force garbage collection after finishing each month
        if GC_AFTER_EACH_MONTH:
            gc.collect()

def main():
    raw_root = "/media/thibault/DynEarthNet/raw_data"
    label_base = os.path.join(raw_root, "labels")
    output_base = "/media/thibault/DynEarthNet/full_data/datasets"

    # Build tile_id -> split mapping
    train_tiles = parse_tiles_from_txt(os.path.join(raw_root, "train.txt"))
    val_tiles   = parse_tiles_from_txt(os.path.join(raw_root, "val.txt"))
    test_tiles  = parse_tiles_from_txt(os.path.join(raw_root, "test.txt"))

    tile_to_split = {}
    for tid in train_tiles:
        tile_to_split[tid] = "train"
    for tid in val_tiles:
        tile_to_split[tid] = "val"
    for tid in test_tiles:
        tile_to_split[tid] = "test"

    print(f"[START] Processing from raw root: {raw_root}")

    # Walk the planet.* directories
    for item in os.listdir(raw_root):
        if item.startswith("planet."):
            for dirpath, _, _ in os.walk(os.path.join(raw_root, item)):
                if os.path.basename(dirpath) == "PF-SR":
                    process_tile_series(
                        dirpath, 
                        label_base, 
                        output_base, 
                        tile_to_split
                    )
                    # After each tile, we can also run gc
                    gc.collect()

if __name__ == "__main__":
    main()
