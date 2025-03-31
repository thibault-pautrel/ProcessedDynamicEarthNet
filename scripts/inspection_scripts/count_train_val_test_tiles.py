#!/usr/bin/env python

import os

# Directories and text file paths (adjust as needed)
UNET_DIR = "/media/thibault/DynEarthNet/raw_data"
TRAIN_TXT = "/media/thibault/DynEarthNet/raw_data/train.txt"
VAL_TXT   = "/media/thibault/DynEarthNet/raw_data/val.txt"
TEST_TXT  = "/media/thibault/DynEarthNet/raw_data/test.txt"

def parse_tiles_from_txt(txt_file):
    """
    Parses a text file and returns a set of tile IDs.
    Each line is expected to contain a path with a PF-SR folder.
    Example line:
        /reprocess-cropped/.../1973_3709_13/PF-SR /labels/... 2018-01
    The tile ID is the parent folder of 'PF-SR' (e.g. '1973_3709_13').
    """
    tiles = set()
    if not os.path.exists(txt_file):
        print(f"[WARN] {txt_file} not found, skipping.")
        return tiles

    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pf_sr_path = parts[0]  # e.g. .../1973_3709_13/PF-SR
            tile_id = os.path.basename(os.path.dirname(pf_sr_path))
            tiles.add(tile_id)
    return tiles

def get_available_tiles(unet_dir):
    """
    Scans all planet.* directories inside the given unet_dir and returns a set
    of available tile IDs.
    """
    available_tiles = set()
    if not os.path.isdir(unet_dir):
        print(f"[ERROR] UNET directory {unet_dir} does not exist.")
        return available_tiles

    # Iterate over planet.* folders
    for planet_dir in os.listdir(unet_dir):
        if not planet_dir.startswith("planet."):
            continue
        planet_path = os.path.join(unet_dir, planet_dir)
        if not os.path.isdir(planet_path):
            continue
        # Tiles are subdirectories in the planet directory
        for tile in os.listdir(planet_path):
            tile_path = os.path.join(planet_path, tile)
            if os.path.isdir(tile_path):
                available_tiles.add(tile)
    return available_tiles

def main():
    # Parse the tile IDs from the text files
    train_tiles = parse_tiles_from_txt(TRAIN_TXT)
    val_tiles   = parse_tiles_from_txt(VAL_TXT)
    test_tiles  = parse_tiles_from_txt(TEST_TXT)

    print(f"Tiles in train.txt: {len(train_tiles)}")
    print(f"Tiles in val.txt:   {len(val_tiles)}")
    print(f"Tiles in test.txt:  {len(test_tiles)}")

    # Get all available tile IDs from the UNET directory
    available_tiles = get_available_tiles(UNET_DIR)
    print(f"\nTotal available tiles (in UNET_DIR): {len(available_tiles)}")

    # Intersection of available tiles with each split
    available_train = available_tiles.intersection(train_tiles)
    available_val   = available_tiles.intersection(val_tiles)
    available_test  = available_tiles.intersection(test_tiles)

    print(f"\nAvailable tiles that are in train.txt: {len(available_train)}")
    print(f"Available tiles that are in val.txt:   {len(available_val)}")
    print(f"Available tiles that are in test.txt:  {len(available_test)}")

if __name__ == "__main__":
    main()
