import os
import glob
import shutil
import top_classes  # Assumes top_classes.py is in the same directory

def main():
    # --- Step 1: Group tile IDs based on raw label frequencies ---

    # Directories used for scanning raw data (adjust if needed)
    raw_root = "/media/thibault/DynEarthNet/raw_data"
    label_root = os.path.join(raw_root, "labels")

    # Get available tile IDs from raw data (from top_classes.py)
    allowed_tile_ids = top_classes.get_available_tile_ids_from_planets(raw_root)
    
    # Compute class frequencies and get top classes per tile
    tile_top_classes = top_classes.collect_class_frequencies_from_raw_labels(label_root, allowed_tile_ids, n=4)
    
    # Group tiles by their dominant class as defined in top_classes.py
    grouped_tiles = top_classes.group_tiles_by_dominant_class(tile_top_classes)
    
    # Map the original group names to simplified folder names.
    # top_classes.py produces group names like "Urban areas", "Agricultural areas", etc.
    mapping = {
        #"Urban areas": "urban",
        #"Agricultural areas": "agri",
        #"Forest-dominated areas": "forest",
        "Mixed land-use areas": "mixed",
        #"Water-dominated areas": "water"
    }
    
    # Build a dictionary with simplified group keys mapping to lists of tile IDs.
    grouped_tiles_simple = {name: [] for name in mapping.values()}
    for group_name, tile_ids in grouped_tiles.items():
        if group_name in mapping:
            simple_name = mapping[group_name]
            grouped_tiles_simple[simple_name].extend(tile_ids)
        else:
            print(f"[INFO] Group '{group_name}' not recognized; skipping those tiles.")

    # --- Step 2: Copy matching tile folders for each dataset type ---
    dataset_types = ["spdnet_monthly","unet"]
    base_source = "/media/thibault/DynEarthNet/subsampled_data/datasets"
    base_dest = "/home/thibault/ProcessedDynamicEarthNet/subsampled_data/clients"
    
    for ds in dataset_types:
        source_dir = os.path.join(base_source, ds)
        dest_dir = os.path.join(base_dest, ds)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"\n[Processing dataset type: {ds}]")
        for group, tile_ids in grouped_tiles_simple.items():
            dest_group_dir = os.path.join(dest_dir, group)
            os.makedirs(dest_group_dir, exist_ok=True)
            print(f"\n  [Group: {group}]")
            
            for tile_id in tile_ids:
                # Assume the dataset folder structure is: {source_dir}/{planet_folder}/{tile_id}/...
                # We search for any folder matching */<tile_id>
                pattern = os.path.join(source_dir, "*", tile_id)
                matches = glob.glob(pattern)
                if not matches:
                    print(f"    [INFO] Tile folder for tile ID '{tile_id}' not found in {source_dir}")
                    continue
                for tile_folder in matches:
                    # The destination folder will be: {dest_group_dir}/{tile_id}
                    dest_tile_folder = os.path.join(dest_group_dir, os.path.basename(tile_folder))
                    if os.path.exists(dest_tile_folder):
                        print(f"    [SKIP] {dest_tile_folder} already exists.")
                    else:
                        try:
                            shutil.copytree(tile_folder, dest_tile_folder)
                            print(f"    [COPIED] {tile_folder} -> {dest_tile_folder}")
                        except Exception as e:
                            print(f"    [ERROR] Could not copy {tile_folder} to {dest_tile_folder}: {e}")

if __name__ == "__main__":
    main()
