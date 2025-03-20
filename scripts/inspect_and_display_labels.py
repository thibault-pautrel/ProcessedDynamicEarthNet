import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_raw_and_label(label_file_path, raw_data_file_path):
    """
    Loads and processes raw data and label files into an RGB image and a color-mapped label overlay.

    Args:
        label_file_path (str): Path to the label raster .pt file.
        raw_data_file_path (str): Path to the raw data .pt file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (RGB image, color label overlay).
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
    Displays 24 raw + label overlay images in a 4-row x 6-column grid.

    Args:
        image_label_pairs (list of tuples): List of (label_file, raw_data_file) pairs.
        month_labels (list of str): List of corresponding month strings (e.g., '2018-01').
        alpha (float): Transparency factor for label overlay.
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
    """
    if len(image_label_pairs) != 24:
        print(f"Expected 24 image/label pairs but got {len(image_label_pairs)}.")
        return

    fig, axs = plt.subplots(4, 6, figsize=(24, 16))  # 4 rows, 6 columns

    for idx, (label_file, raw_file) in enumerate(image_label_pairs):
        row = idx // 6
        col = idx % 6

        rgb_img, label_overlay = load_raw_and_label(label_file, raw_file)

        axs[row, col].imshow(rgb_img)
        axs[row, col].imshow(label_overlay, alpha=alpha)

        # Display month label from the provided month_labels list
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

    plt.show()


def inspect_and_display_labels_24_months(planet_tile_path, alpha=0.4, save_dir=None):
    """
    Displays raw images and their label overlays for 24 consecutive months (2018-01 to 2019-12).

    Args:
        planet_tile_path (str): Base path to a tile folder, e.g., '/media/thibault/DynEarthNet/planet.11N/1417_3281_13'.
        alpha (float): Transparency factor for label overlay.
        save_dir (str, optional): Directory where to save the output figure. If None, figure won't be saved.
    """
    # Parse path for planet and tile ID
    path_parts = planet_tile_path.rstrip('/').split('/')
    planet_folder = path_parts[-2] if len(path_parts) >= 2 else "unknown_planet"
    time_series_id = path_parts[-1] if len(path_parts) >= 1 else "unknown_tile"

    # Expected months from 2018-01 to 2019-12
    months = [f"{year}-{month:02d}" for year in range(2018, 2020) for month in range(1, 13)]
    print(f"Looking for months: {months}")

    image_label_pairs = []
    valid_month_labels = []

    for month in months:
        month_path = os.path.join(planet_tile_path, month)

        label_file = os.path.join(month_path, 'labels', 'raster.pt')
        raw_files = glob.glob(os.path.join(month_path, 'data_*.pt'))

        if not os.path.isdir(month_path):
            print(f"Month folder missing: {month_path}. Skipping...")
            continue

        if not raw_files:
            print(f"No raw data files found in {month_path}. Skipping...")
            continue

        raw_file = raw_files[0]  # First available data file for the month

        if not os.path.isfile(label_file):
            print(f"No label file found in {month_path}. Skipping...")
            continue

        image_label_pairs.append((label_file, raw_file))
        valid_month_labels.append(month)

    if len(image_label_pairs) != 24:
        print(f"Warning: Expected 24 months, but found {len(image_label_pairs)} valid months.")

    # Auto-generate save path if save_dir is provided
    save_path = None
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = f"label_overlay_grid_{planet_folder}_{time_series_id}.png"
        save_path = os.path.join(save_dir, save_filename)

    display_24_images_grid(image_label_pairs, month_labels=valid_month_labels, alpha=alpha, save_path=save_path)


if __name__ == "__main__":
    inspect_and_display_labels_24_months(
        planet_tile_path="/media/thibault/DynEarthNet/planet.15N/2029_3764_13",
        alpha=0.4,
        save_dir="/home/thibault/ProcessedDynamicEarthNet/figures/labeled_images_displayed"
    )
