import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_raw_and_label(label_file_path, raw_data_file_path):
    """
    Loads and processes raw data and label files to generate an RGB image and a color-mapped label overlay.

    Args:
        label_file_path (str): Path to the label raster file (.pt) containing label information.
        raw_data_file_path (str): Path to the raw data file (.pt) containing image data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - rgb_np (np.ndarray): The raw RGB image array, shape (H, W, 3).
            - color_label_map (np.ndarray): The color-mapped label overlay, shape (H, W, 3).
    """
    labels = torch.load(label_file_path)
    labels_binary = (labels > 0).int()
    label_map = torch.argmax(labels_binary, dim=0).cpu().numpy()

    color_mapping = {
        0: (127/255, 127/255, 127/255),
        1: (189/255, 189/255, 34/255),
        2: (51/255, 204/255, 51/255),
        3: (0/255, 0/255, 153/255),
        4: (153/255, 102/255, 51/255),
        5: (51/255, 153/255, 255/255),
        6: (153/255, 204/255, 204/255),
    }

    color_label_map = np.zeros((label_map.shape[0], label_map.shape[1], 3))
    for class_index, color in color_mapping.items():
        mask = label_map == class_index
        color_label_map[mask] = color

    raw_data = torch.load(raw_data_file_path)
    rgb_tensor = raw_data[:3, :, :]
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()

    if rgb_np.max() > 1.0:
        rgb_np = rgb_np / rgb_np.max()

    return rgb_np, color_label_map

def display_24_images_grid(image_label_pairs, month_labels, alpha=0.4, save_path=None):
    """
    Displays a 4x6 grid of RGB images with overlaid label maps for 24 months.

    Args:
        image_label_pairs (list of tuples): List of (label_file_path, raw_data_file_path) pairs for each month.
        month_labels (list of str): List of month names corresponding to each image, e.g., '2018-01'.
        alpha (float, optional): Transparency for the label overlay on top of the RGB image. Default is 0.4.
        save_path (str, optional): If provided, the figure is saved to this path. Default is None (no save).

    Returns:
        None
    """
    if len(image_label_pairs) != 24:
        print(f"Expected 24 image/label pairs but got {len(image_label_pairs)}.")
        return

    fig, axs = plt.subplots(4, 6, figsize=(24, 16))
    for idx, (label_file, raw_file) in enumerate(image_label_pairs):
        row = idx // 6
        col = idx % 6
        rgb_img, label_overlay = load_raw_and_label(label_file, raw_file)
        axs[row, col].imshow(rgb_img)
        axs[row, col].imshow(label_overlay, alpha=alpha)
        month_label = month_labels[idx]
        axs[row, col].set_title(f'{month_label}', fontsize=14)
        axs[row, col].axis('off')
    plt.tight_layout()
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved at {save_path}")
    plt.close(fig)

def inspect_and_display_labels_24_months(planet_tile_path, alpha=0.4, save_dir=None):
    """
    Loads and displays label overlays for 24 months for a single tile and optionally saves the figure.
    If a figure for this tile already exists in the save directory, processing is skipped.

    Args:
        planet_tile_path (str): Full path to the tile directory, e.g., /media/thibault/DynEarthNet/planet.11N/1417_3281_13.
        alpha (float, optional): Transparency for label overlay on top of RGB image. Default is 0.4.
        save_dir (str, optional): Directory to save the generated figure. Default is None (no save).

    Returns:
        None
    """
    path_parts = planet_tile_path.rstrip('/').split('/')
    planet_folder = path_parts[-2] if len(path_parts) >= 2 else "unknown_planet"
    time_series_id = path_parts[-1] if len(path_parts) >= 1 else "unknown_tile"

    # Build expected save path and check if figure exists
    save_path = None
    if save_dir:
        save_filename = f"label_overlay_grid_{planet_folder}_{time_series_id}.png"
        save_path = os.path.join(save_dir, save_filename)
        if os.path.isfile(save_path):
            print(f"Skipping {planet_folder}/{time_series_id}: Figure already exists at {save_path}")
            return

    # Build list of months (expected 24 months: 2018-01 to 2019-12)
    months = [f"{year}-{month:02d}" for year in range(2018, 2020) for month in range(1, 13)]
    image_label_pairs = []
    valid_month_labels = []

    for month in months:
        month_path = os.path.join(planet_tile_path, month)
        label_file = os.path.join(month_path, 'labels', 'raster.pt')
        raw_files = glob.glob(os.path.join(month_path, 'data_*.pt'))
        if not os.path.isdir(month_path):
            continue
        if not raw_files or not os.path.isfile(label_file):
            continue
        raw_file = raw_files[0]
        image_label_pairs.append((label_file, raw_file))
        valid_month_labels.append(month)

    if len(image_label_pairs) != 24:
        print(f"Skipping {planet_folder}/{time_series_id}: Only found {len(image_label_pairs)} months.")
        return

    print(f"Processing tile {planet_folder}/{time_series_id} ...")
    display_24_images_grid(image_label_pairs, month_labels=valid_month_labels, alpha=alpha, save_path=save_path)

def batch_process_all_planet_tiles(base_dataset_dir, save_dir, alpha=0.4):
    """
    Processes and displays label overlays for all tiles in all planet.* directories found in the base dataset directory.
    Checks if a figure for a tile already exists; if so, it skips that tile.

    Args:
        base_dataset_dir (str): Root directory containing planet.* folders, e.g., /media/thibault/DynEarthNet.
        save_dir (str): Directory to save the generated figures.
        alpha (float, optional): Transparency for label overlay on top of RGB image. Default is 0.4.

    Returns:
        None
    """
    planet_folders = sorted(glob.glob(os.path.join(base_dataset_dir, 'planet.*')))
    if not planet_folders:
        print(f"No planet.* directories found in {base_dataset_dir}")
        return

    print(f"Found {len(planet_folders)} planet folders.")
    for planet_folder in planet_folders:
        tile_folders = sorted(glob.glob(os.path.join(planet_folder, '*_*_*')))
        if not tile_folders:
            print(f"No tiles found in {planet_folder}")
            continue
        print(f"{planet_folder}: Found {len(tile_folders)} tiles.")
        for tile_folder in tile_folders:
            try:
                inspect_and_display_labels_24_months(
                    planet_tile_path=tile_folder,
                    alpha=alpha,
                    save_dir=save_dir
                )
            except Exception as e:
                print(f"Error processing {tile_folder}: {e}")

################################################################################
# New functionality: Display only 2018-01 labels for unet splits (train,test,val) #
################################################################################

def inspect_and_display_2018_01_labels_unet(save_path=None):
    """
    Loads and displays only the 2018-01 labels for each tile found in the three unet splits:
    /media/thibault/DynEarthNet/subsampled_data/unet/{train,test,val}.
    
    For each tile, the function:
      - Searches for the pixel dataset file corresponding to 2018-01 (either in a '2018-01' subfolder or directly).
      - Loads the "labels" tensor (of shape [H, W]) from the dataset (as constructed by month_stacked_labels.py).
      - Converts the label matrix into a colorized image using a defined color mapping.
      - Computes the top-2 most frequent label classes and uses a lookup dictionary (similar to top_classes.py) to get class names.
    
    Finally, the function displays a single figure arranged in three columns (one for each split) with one column per split.
    Each subplot shows a tile’s 2018-01 label image with a title that includes the tile name and the top-2 classes.
    
    Args:
        save_path (str, optional): If provided, saves the figure to the given path.
    
    Returns:
        None
    """
    base_unet_dir = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet"
    splits = ["train", "test", "val"]

    # Color mapping (same as before)
    color_mapping = {
        0: (127/255, 127/255, 127/255),
        1: (189/255, 189/255, 34/255),
        2: (51/255, 204/255, 51/255),
        3: (0/255, 0/255, 153/255),
        4: (153/255, 102/255, 51/255),
        5: (51/255, 153/255, 255/255),
        6: (153/255, 204/255, 204/255),
    }
    # Optional: mapping to more descriptive class names
    class_to_group = {
        0: "Urban areas",
        1: "Agricultural areas",
        2: "Forest-dominated areas",
        3: "Mixed land-use areas",
        4: "Mixed land-use areas",
        5: "Water-dominated areas",
        6: "Mixed land-use areas",
    }
    
    # Dictionary to store (tile_name, color_image, top2 legend) per split.
    images_by_split = {split: [] for split in splits}

    for split in splits:
        split_dir = os.path.join(base_unet_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Directory not found: {split_dir}")
            continue
        # Assuming each tile is a directory inside the split folder.
        tile_names = sorted(os.listdir(split_dir))
        for tile in tile_names:
            tile_path = os.path.join(split_dir, tile)
            if not os.path.isdir(tile_path):
                continue
            # Look for the 2018-01 pixel dataset file.
            # First try: inside a "2018-01" subfolder.
            candidate_path = os.path.join(tile_path, "2018-01", "pixel_dataset_2018-01.pt")
            if not os.path.isfile(candidate_path):
                # Second try: directly in the tile folder.
                candidate_path = os.path.join(tile_path, "pixel_dataset_2018-01.pt")
            if not os.path.isfile(candidate_path):
                print(f"[INFO] 2018-01 label file not found for tile {tile} in split {split}.")
                continue
            try:
                dataset = torch.load(candidate_path)
            except Exception as e:
                print(f"[ERROR] Could not load file {candidate_path}: {e}")
                continue
            if "labels" not in dataset:
                print(f"[ERROR] 'labels' key not found in {candidate_path}")
                continue
            labels = dataset["labels"]  # expected shape: [H, W]
            # Convert to numpy array.
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.array(labels)
            # Compute top-2 most frequent classes.
            unique, counts = np.unique(labels_np, return_counts=True)
            pairs = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            if len(pairs) >= 2:
                top2 = pairs[:2]
            elif len(pairs) == 1:
                top2 = pairs
            else:
                top2 = []
            if top2:
                top2_str = "Top: " + ", ".join(f"{cls} ({class_to_group.get(cls, str(cls))})" for cls, cnt in top2)
            else:
                top2_str = "No classes"
            # Create a color image from the label array.
            H, W = labels_np.shape
            color_img = np.zeros((H, W, 3), dtype=np.float32)
            for cls, color in color_mapping.items():
                mask = labels_np == cls
                color_img[mask] = color
            images_by_split[split].append((tile, color_img, top2_str))
    
    # Determine the maximum number of tiles among splits for layout.
    max_tiles = max(len(images_by_split[split]) for split in splits)
    if max_tiles == 0:
        print("No tiles found for any split. Exiting.")
        return
    
    # Create a figure with three columns (one per split) and max_tiles rows.
    fig, axs = plt.subplots(max_tiles, len(splits), figsize=(len(splits)*4, max_tiles*4))
    
    # In case there is only one row, wrap axs into 2D array.
    if max_tiles == 1:
        axs = np.expand_dims(axs, axis=0)
    
    for col_idx, split in enumerate(splits):
        tiles = images_by_split[split]
        for row_idx in range(max_tiles):
            ax = axs[row_idx, col_idx]
            if row_idx < len(tiles):
                tile_name, color_img, legend_str = tiles[row_idx]
                ax.imshow(color_img)
                ax.set_title(f"{split} / {tile_name}\n{legend_str}", fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Unet 2018-01 labels figure saved at {save_path}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # Uncomment one of the following calls depending on what you want to run.
    
    # To run the original batch processing of planet tiles (24-month overlays):
    # base_dataset_dir = "/media/thibault/DynEarthNet/subsampled_data"
    # save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/subsampled_data/labeled_images_displayed"
    # batch_process_all_planet_tiles(
    #     base_dataset_dir=base_dataset_dir,
    #     save_dir=save_dir,
    #     alpha=0.4
    # )
    
    # To run the new function for unet 2018-01 labels, arranged in 3 columns (train, test, val):
    save_path = "/home/thibault/ProcessedDynamicEarthNet/figures/train_val_test_labels_2018-01.png"
    inspect_and_display_2018_01_labels_unet(save_path=save_path)
