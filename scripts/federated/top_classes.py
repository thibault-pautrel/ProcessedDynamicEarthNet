import os
import glob
import torch
import rasterio
import heapq
from collections import Counter, defaultdict

SUBSAMPLE = 8  # Spatial downsampling

def load_tif_as_tensor(filepath, subsample=1):
    with rasterio.open(filepath) as src:
        img = src.read()  # shape: [C, H, W]
        if subsample > 1:
            img = img[:, ::subsample, ::subsample]
    return torch.tensor(img)

def get_available_tile_ids_from_planets(raw_root: str) -> set:
    """
    Extract and normalize tile_ids from /raw_data/planet.*/.../<tile_id>/PF-SR.
    Normalization: replace hyphens with underscores and keep first three parts.
    """
    tile_ids = set()
    for entry in os.listdir(raw_root):
        if entry.startswith("planet."):
            for dirpath, _, _ in os.walk(os.path.join(raw_root, entry)):
                if os.path.basename(dirpath) == "PF-SR":
                    # Extract the directory name above PF-SR and normalize it.
                    tile_folder = os.path.basename(os.path.dirname(dirpath))
                    tile_folder_norm = tile_folder.replace('-', '_')
                    tile_id = "_".join(tile_folder_norm.split("_")[0:3])
                    tile_ids.add(tile_id)
    return tile_ids

def collect_class_frequencies_from_raw_labels(label_root_dir: str, allowed_tile_ids: set, n=3):
    """
    Read raw .tif label files and count class frequency for matching tile_ids.
    
    The function adapts to both cases where the labels folder structure is either:
      <label_root>/<tile_folder>/Labels/Raster/<raster_tile_folder>/<file.tif>
    or
      <label_root>/<tile_folder>/Raster/<raster_tile_folder>/<file.tif>
    
    The tile_id is extracted from <tile_folder> (after normalizing hyphens to underscores).
    """
    tile_class_summary = defaultdict(Counter)
    
    # Define both glob patterns
    pattern_with_labels = os.path.join(label_root_dir, "*", "Labels", "Raster", "*", "*.tif")
    pattern_without_labels = os.path.join(label_root_dir, "*", "Raster", "*", "*.tif")
    
    # Combine files from both patterns, deduplicating them.
    tif_files = list(set(glob.glob(pattern_with_labels) + glob.glob(pattern_without_labels)))

    for tif_file in tif_files:
        try:
            # Compute the path relative to label_root_dir to extract the label folder.
            rel_path = os.path.relpath(tif_file, label_root_dir)
            parts = rel_path.split(os.sep)
            # parts[0] is the tile folder name in both cases.
            label_folder = parts[0]
            label_folder_norm = label_folder.replace('-', '_')
            tile_id = "_".join(label_folder_norm.split("_")[0:3])
            if tile_id not in allowed_tile_ids:
                continue
            tensor = load_tif_as_tensor(tif_file, subsample=SUBSAMPLE)
            tensor = (tensor > 127).long()
            class_map = torch.argmax(tensor, dim=0)
            tile_class_summary[tile_id].update(class_map.view(-1).tolist())
        except Exception as e:
            print(f"[ERROR] Failed to process {tif_file}: {e}")

    return {
        tile_id: heapq.nlargest(n, counter.items(), key=lambda x: x[1])
        for tile_id, counter in tile_class_summary.items()
    }

def group_tiles_by_dominant_class(tile_top_classes: dict) -> dict:
    class_to_group = {
        0: "Urban areas",
        1: "Agricultural areas",
        2: "Forest-dominated areas",
        3: "Mixed land-use areas",  # Wetlands
        4: "Mixed land-use areas",  # Soil
        5: "Water-dominated areas",
        6: "Mixed land-use areas",  # Snow & Ice
    }

    grouped_tiles = defaultdict(list)
    for tile_id, top_classes in tile_top_classes.items():
        if not top_classes:
            continue
        dominant_class = top_classes[0][0]
        group = class_to_group.get(dominant_class, "Unknown")
        grouped_tiles[group].append(tile_id)

    return grouped_tiles

if __name__ == "__main__":
    raw_root = "/media/thibault/DynEarthNet/raw_data"
    label_root = os.path.join(raw_root, "labels")
    top_n = 4

    print(f"Scanning: {label_root}")
    available_tile_ids = get_available_tile_ids_from_planets(raw_root)
    tile_top_classes = collect_class_frequencies_from_raw_labels(label_root, allowed_tile_ids=available_tile_ids, n=top_n)
    grouped_tiles = group_tiles_by_dominant_class(tile_top_classes)

    for group, tiles in grouped_tiles.items():
        print(f"\n{group} ({len(tiles)} tiles):")
        for tid in sorted(tiles):
            print(f"  - {tid}")
