#!/usr/bin/env python

import os
import shutil

UNET_DIR = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet"
SPDNET_DIR = "/media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly"

TRAIN_TXT = "/media/thibault/DynEarthNet/raw_data/train.txt"
VAL_TXT   = "/media/thibault/DynEarthNet/raw_data/val.txt"
TEST_TXT  = "/media/thibault/DynEarthNet/raw_data/test.txt"

def parse_tiles_from_txt(txt_file):
    """
    Parses a text file containing paths with tile information and extracts the tile ID.
    
    Example line:
      /reprocess-cropped/.../1973_3709_13/PF-SR /labels/... 2018-01
    The tile ID is extracted as the parent folder of "PF-SR" (e.g. '1973_3709_13').
    
    Args:
        txt_file (str): Path to the text file.
        
    Returns:
        set: A set of unique tile IDs extracted from the file.
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

            pf_sr_path = parts[0]  # e.g. /reprocess-cropped/.../1973_3709_13/PF-SR
            # tile ID is the parent directory of PF-SR:
            tile_id = os.path.basename(os.path.dirname(pf_sr_path))
            tiles.add(tile_id)
    return tiles

def ensure_dir_exists(path):
    """
    Ensures that the given directory path exists.
    
    Args:
        path (str): The directory path.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def alternate_tile_id(tile_id):
    """
    Generates an alternate version of a tile ID by swapping hyphens and underscores.
    
    If the tile_id contains underscores, they are replaced with hyphens.
    Otherwise, if it contains hyphens, they are replaced with underscores.
    
    Args:
        tile_id (str): The original tile ID.
        
    Returns:
        str: The alternate tile ID.
    """
    if "_" in tile_id:
        return tile_id.replace("_", "-")
    elif "-" in tile_id:
        return tile_id.replace("-", "_")
    else:
        return tile_id

def move_tile_if_exists(tile_id, from_basedir, to_basedir, split_name):
    """
    Searches for a tile_id folder within from_basedir under any planet.* subdirectory.
    If found, moves it to to_basedir/split_name/tile_id.
    
    If the folder is not found using the given tile_id, an alternate version (hyphen vs underscore)
    is also attempted.
    
    Args:
        tile_id (str): The tile ID to move.
        from_basedir (str): The base directory to search (e.g. UNET_DIR or SPDNET_DIR).
        to_basedir (str): The base directory where the tile should be moved.
        split_name (str): The target split subdirectory (e.g. "train", "val", or "test").
    """
    if not os.path.isdir(from_basedir):
        return  # Nothing to do

    planet_dirs = [d for d in os.listdir(from_basedir) if d.startswith("planet.")]
    for planet_dir in planet_dirs:
        # Try the original tile_id first
        tile_path = os.path.join(from_basedir, planet_dir, tile_id)
        if not os.path.isdir(tile_path):
            # If not found, try the alternate tile_id
            alt_tile = alternate_tile_id(tile_id)
            tile_path = os.path.join(from_basedir, planet_dir, alt_tile)
            if os.path.isdir(tile_path):
                tile_id = alt_tile  # update tile_id to the alternate version
        
        if os.path.isdir(tile_path):
            # Found the tile folder; move it
            target_dir = os.path.join(to_basedir, split_name)
            ensure_dir_exists(target_dir)
            dest_path = os.path.join(target_dir, tile_id)
            if os.path.exists(dest_path):
                print(f"[SKIP] {dest_path} already exists, skipping tile {tile_id}")
                return
            print(f"Moving {tile_path} -> {dest_path}")
            shutil.move(tile_path, dest_path)
            return  # Stop after first match

def main():
    # Collect tile IDs from train, val, and test files
    train_tiles = parse_tiles_from_txt(TRAIN_TXT)
    val_tiles   = parse_tiles_from_txt(VAL_TXT)
    test_tiles  = parse_tiles_from_txt(TEST_TXT)

    print(f"Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}, Test tiles: {len(test_tiles)}")

    # Process UNET data
    for t in train_tiles:
        move_tile_if_exists(t, UNET_DIR, UNET_DIR, "train")
    for t in val_tiles:
        move_tile_if_exists(t, UNET_DIR, UNET_DIR, "val")
    for t in test_tiles:
        move_tile_if_exists(t, UNET_DIR, UNET_DIR, "test")

    # Process SPDNET data
    for t in train_tiles:
        move_tile_if_exists(t, SPDNET_DIR, SPDNET_DIR, "train")
    for t in val_tiles:
        move_tile_if_exists(t, SPDNET_DIR, SPDNET_DIR, "val")
    for t in test_tiles:
        move_tile_if_exists(t, SPDNET_DIR, SPDNET_DIR, "test")


def remove_empty_planet_dirs(basedir):
    """
    Iterates over all planet.* directories in the given base directory.
    If a planet directory is empty (i.e., it contains no subdirectories), it is deleted.
    
    Args:
        basedir (str): The base directory (e.g. UNET_DIR or SPDNET_DIR).
    """
    planet_dirs = [d for d in os.listdir(basedir) if d.startswith("planet.")]
    for planet_dir in planet_dirs:
        planet_path = os.path.join(basedir, planet_dir)
        # If the directory is empty (no files or subdirectories), delete it.
        if not os.listdir(planet_path):
            print(f"Removing empty directory {planet_path}")
            shutil.rmtree(planet_path)

if __name__ == "__main__":
    main()
    remove_empty_planet_dirs(UNET_DIR)
    remove_empty_planet_dirs(SPDNET_DIR)
    print("Done!")
