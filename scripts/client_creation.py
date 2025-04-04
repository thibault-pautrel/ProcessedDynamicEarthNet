import os
import glob
import gc
import shutil
import torch
import rasterio
from datetime import datetime
from collections import defaultdict, Counter

###############################################################################
# Configurable parameters and paths
###############################################################################
RAW_ROOT = "/media/thibault/DynEarthNet/raw_data"
LABEL_ROOT = os.path.join(RAW_ROOT, "labels")

FEDERATED_BASE = "/media/thibault/DynEarthNet/federated/datasets"  # final output
FULLDATA_BASE = "/media/thibault/DynEarthNet/full_data/datasets"   # existing val/test from train_val_test_datasets_creation.py

TRAIN_TXT = os.path.join(RAW_ROOT, "train.txt")
VAL_DIR   = os.path.join(FULLDATA_BASE, "val")
TEST_DIR  = os.path.join(FULLDATA_BASE, "test")

MAX_TIFS_PER_MONTH = 40     
USE_HALF_PRECISION = False  
GC_AFTER_EACH_MONTH = True  

###############################################################################
# Functions from train_val_test_datasets_creation.py
###############################################################################
def load_tif_as_tensor(filepath):
    with rasterio.open(filepath) as src:
        img = src.read()
    return torch.tensor(img)

def build_month_tensor_from_tifs(tif_files):
    tif_files = sorted(tif_files)
    print(f"  [INFO] Stacking {len(tif_files)} daily files...")

    example = load_tif_as_tensor(tif_files[0])  # shape [4, H, W]
    bands, H, W = example.shape

    dtype = torch.float16 if USE_HALF_PRECISION else torch.float32
    stacked_data = torch.zeros((H, W, bands * len(tif_files)), dtype=dtype)

    stacked_data[:, :, 0:4] = example.permute(1, 2, 0).to(dtype)

    for t, tif_path in enumerate(tif_files[1:], start=1):
        data = load_tif_as_tensor(tif_path).permute(1, 2, 0).to(dtype)
        stacked_data[:, :, t*4:(t+1)*4] = data
        del data
    del example
    return stacked_data, H, W

def convert_labels_to_class_indices(label_tensor, H, W):
    label_tensor = (label_tensor > 127).long()
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
    return sorted(month_keys)[::6][:4]

###############################################################################
# Train tile ID parsing (fix so it matches PF-SR scanning)
###############################################################################
def parse_train_tiles(train_txt):
    """
    Reads train.txt and extracts tile IDs, normalizing them the same way 
    we do when scanning planet.* directories.
    """
    train_tiles = set()
    if not os.path.exists(train_txt):
        print(f"[WARNING] {train_txt} not found.")
        return train_tiles

    with open(train_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pf_sr_path = parts[0]
            tile_folder = os.path.basename(os.path.dirname(pf_sr_path))  
            # unify hyphens => underscores, then keep first 3 parts
            tile_folder_norm = tile_folder.replace('-', '_')
            tile_id = "_".join(tile_folder_norm.split('_')[0:3])
            train_tiles.add(tile_id)
    return train_tiles


def get_available_tile_ids_from_planets(raw_root: str) -> set:
    """
    Walk /raw_data/planet.*, collecting tile IDs from the folder above PF-SR, 
    normalizing them (replace '-' => '_', keep first 3 underscore parts).
    """
    tile_ids = set()
    for entry in os.listdir(raw_root):
        if entry.startswith("planet."):
            planet_path = os.path.join(raw_root, entry)
            if not os.path.isdir(planet_path):
                continue
            for dirpath, _, _ in os.walk(planet_path):
                if os.path.basename(dirpath) == "PF-SR":
                    folder = os.path.basename(os.path.dirname(dirpath))
                    folder_norm = folder.replace('-', '_')
                    tid = "_".join(folder_norm.split('_')[0:3])
                    tile_ids.add(tid)
    return tile_ids

###############################################################################
# Class frequency => dominant class (train only)
###############################################################################
def collect_class_frequencies_for_4classes(label_root_dir: str, train_tile_ids: set):
    from collections import defaultdict, Counter
    tile_to_class_counts = defaultdict(Counter)

    pattern1 = os.path.join(label_root_dir, "*", "Labels", "Raster", "*", "*.tif")
    pattern2 = os.path.join(label_root_dir, "*", "Raster", "*", "*.tif")
    tif_files = list(set(glob.glob(pattern1) + glob.glob(pattern2)))
    if not tif_files:
        print(f"[WARNING] No label TIF files found under {label_root_dir}.")

    for tif_file in tif_files:
        try:
            rel_path = os.path.relpath(tif_file, label_root_dir)
            parts = rel_path.split(os.sep)
            tile_folder = parts[0].replace('-', '_')
            tile_id = "_".join(tile_folder.split('_')[0:3])
            # only proceed if tile is in train set
            if tile_id not in train_tile_ids:
                continue

            lbl_tensor = load_tif_as_tensor(tif_file)
            lbl_tensor = (lbl_tensor > 127).long()  # shape [7, H, W]
            argmax_mask = torch.argmax(lbl_tensor, dim=0)

            remapped = torch.full_like(argmax_mask, -1)
            remapped[argmax_mask == 0] = 0
            remapped[(argmax_mask == 1) | (argmax_mask == 3) |
                     (argmax_mask == 4) | (argmax_mask == 6)] = 1
            remapped[argmax_mask == 2] = 2
            remapped[argmax_mask == 5] = 3

            tile_to_class_counts[tile_id].update(remapped.view(-1).tolist())
        except Exception as e:
            print(f"[ERROR reading] {tif_file}: {e}")

    return tile_to_class_counts

def compute_tile_dominant_class(tile_to_class_counts: dict):
    tile_to_dom = {}
    for tile_id, counts in tile_to_class_counts.items():
        if not counts:
            continue
        dom_class, _ = max(counts.items(), key=lambda x: x[1])
        tile_to_dom[tile_id] = dom_class
    return tile_to_dom

###############################################################################
# Building monthly .pt data for the train set, by dominant class
###############################################################################
def process_tile_series(series_path, tile_to_class, class_folder_map):
    """
    For each tile in the *train* set, produce monthly .pt under 
    /federated/datasets/{urban,mixed,forest,water}/tile_id/YYYY-MM.
    """
    tile_folder = os.path.basename(os.path.dirname(series_path))
    tile_folder_norm = tile_folder.replace('-', '_')
    tile_id = "_".join(tile_folder_norm.split('_')[0:3])

    if tile_id not in tile_to_class:
        # skip if not train or no known class
        return

    dom_class_idx = tile_to_class[tile_id]
    class_subdir  = class_folder_map[dom_class_idx]  # e.g. "urban", "mixed", etc.

    daily_files = sorted(glob.glob(os.path.join(series_path, "*.tif")))
    if not daily_files:
        return

    # planet.* => figure out planet_id => find label folder
    planet_dir = next((p for p in series_path.split(os.sep) if p.startswith("planet.")), None)
    planet_id = planet_dir.split('.')[-1] if planet_dir else "unknown"
    label_folder_name = f"{tile_folder}_{planet_id}"

    raster_label_base = os.path.join(LABEL_ROOT, label_folder_name, "Labels", "Raster")
    raster_tile_folder = None
    if os.path.exists(raster_label_base):
        subfld = os.listdir(raster_label_base)
        if subfld:
            raster_tile_folder = os.path.join(raster_label_base, subfld[0])

    # group daily by month
    monthly_files = {}
    for f in daily_files:
        basename = os.path.basename(f)
        try:
            dt = datetime.strptime(basename.split('.')[0], "%Y-%m-%d")
            key = dt.strftime("%Y-%m")
            monthly_files.setdefault(key, []).append((dt, f))
        except:
            continue

    chosen_months = select_months(monthly_files.keys())
    out_base_dir = os.path.join(FEDERATED_BASE, class_subdir, tile_id)
    os.makedirs(out_base_dir, exist_ok=True)

    for month in chosen_months:
        if month not in monthly_files:
            continue
        tif_paths = [x[1] for x in sorted(monthly_files[month], key=lambda x: x[0])]
        if len(tif_paths) > MAX_TIFS_PER_MONTH:
            continue
        try:
            features, H, W = build_month_tensor_from_tifs(tif_paths)
            # find label
            label_candidates = []
            if raster_tile_folder:
                pat1 = os.path.join(raster_tile_folder, f"*{month.replace('-', '_')}_01.tif")
                pat2 = os.path.join(raster_tile_folder, f"*{month}-01.tif")
                label_candidates = glob.glob(pat1) + glob.glob(pat2)
            if not label_candidates:
                del features
                gc.collect()
                continue

            lbl_tensor = load_tif_as_tensor(label_candidates[0]).float()
            labels = convert_labels_to_class_indices(lbl_tensor, H, W)
            del lbl_tensor

            save_dir = os.path.join(out_base_dir, month)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"pixel_dataset_{month}.pt")

            torch.save({"features": features, "labels": labels}, save_path)
            print(f"[SAVED to {class_subdir}] {save_path}")

            del features
            del labels
        except Exception as e:
            print(f"[ERROR building {tile_id}, {month}]: {e}")

        if GC_AFTER_EACH_MONTH:
            gc.collect()


def main():
    # 1) read train.txt => train_tiles
    train_tiles = parse_train_tiles(TRAIN_TXT)
    print(f"[INFO] Found {len(train_tiles)} tile IDs in train.txt")

    # 2) among planet.* tiles, keep only those in train set
    all_planet_tiles = get_available_tile_ids_from_planets(RAW_ROOT)
    train_planet_tiles = all_planet_tiles.intersection(train_tiles)
    print(f"[INFO] Among planet.* tiles, {len(train_planet_tiles)} are in train set")

    # 3) compute dominant class for these train tiles
    tile_class_counts = collect_class_frequencies_for_4classes(LABEL_ROOT, train_planet_tiles)
    tile_to_class = compute_tile_dominant_class(tile_class_counts)
    print(f"[INFO] Dominant class found for {len(tile_to_class)} train tiles")

    # 4) create subfolders for each class
    class_folder_map = {
        0: "urban",
        1: "mixed",
        2: "forest",
        3: "water"
    }
    for cdir in class_folder_map.values():
        os.makedirs(os.path.join(FEDERATED_BASE, cdir), exist_ok=True)

    # 5) walk planet.* PF-SR => build monthly data for train tiles only
    for entry in os.listdir(RAW_ROOT):
        if entry.startswith("planet."):
            pdir = os.path.join(RAW_ROOT, entry)
            if not os.path.isdir(pdir):
                continue
            for dirpath, _, _ in os.walk(pdir):
                if os.path.basename(dirpath) == "PF-SR":
                    process_tile_series(dirpath, tile_to_class, class_folder_map)
                    gc.collect()


    print("[DONE] Created /urban, /mixed, /forest, /water from train tiles.")


if __name__ == "__main__":
    main()
