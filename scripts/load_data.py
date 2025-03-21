import os
import glob
import torch
import rasterio
import json
from datetime import datetime

def load_tif_as_tensor(filepath):
    """
    Load a GeoTIFF file as a PyTorch tensor.
    Args:
        filepath (str): Path to the .tif file.
    Returns:
        torch.Tensor: Tensor containing the image data.
    """
    with rasterio.open(filepath) as src:
        img = src.read()  # typically shape: (bands, H, W)
    return torch.tensor(img)

def load_geojson_as_tensor(filepath):
    """
    Load a GeoJSON file as a Python dictionary.
    Args:
        filepath (str): Path to the .geojson file.
    Returns:
        dict: Parsed GeoJSON data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def process_time_series(series_path, label_base_path, output_base):
    """
    Process a time series directory, converting daily imagery to individual .pt files
    and processing the corresponding labels, then saving everything into monthly folders.
    In each month folder, daily imagery is saved as data_YYYY-MM-DD.pt.
    
    Args:
        series_path (str): Path to the time series folder containing daily .tif files.
        label_base_path (str): Base path where labels are stored.
        output_base (str): Base directory where the processed outputs will be saved.
    """
    # Determine time_series_id and planet info from the series_path.
    time_series_id = os.path.basename(os.path.dirname(series_path))
    planet_dir = next((part for part in series_path.split(os.sep) if part.startswith("planet.")), None)
    if not planet_dir:
        print(f"[ERROR] No planet dir found in {series_path}")
        return
    planet_id = planet_dir.split('.')[-1]
    label_folder_name = f"{time_series_id}_{planet_id}"

    # Locate label folders (raster and vector)
    raster_label_base = os.path.join(label_base_path, label_folder_name, "Labels", "Raster")
    raster_tile_folder = None
    if os.path.exists(raster_label_base):
        raster_tile_folders = os.listdir(raster_label_base)
        if raster_tile_folders:
            raster_tile_folder = os.path.join(raster_label_base, raster_tile_folders[0])

    vector_label_base = os.path.join(label_base_path, label_folder_name, "Labels")
    vector_label_folder = None
    for vec_folder in ["vector", "Vector"]:
        path = os.path.join(vector_label_base, vec_folder)
        if os.path.exists(path):
            vector_label_folder = path
            break

    # Create the output folder for this time series
    out_dir = os.path.join(output_base, planet_dir, time_series_id)
    os.makedirs(out_dir, exist_ok=True)

    # Find all daily .tif files in the series folder
    daily_files = sorted(glob.glob(os.path.join(series_path, "*.tif")))
    if not daily_files:
        print(f"[WARNING] No tif files in {series_path}")
        return

    # Group daily files by month (e.g. "2018-01")
    monthly_files = {}
    for file in daily_files:
        basename = os.path.basename(file)
        date_str = basename.split('.')[0]
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            month_key = dt.strftime("%Y-%m")
            monthly_files.setdefault(month_key, []).append((dt, file))
        except Exception as e:
            print(f"[ERROR] Date parsing error for {basename}: {e}")

    for month, files in monthly_files.items():
        month_out_dir = os.path.join(out_dir, month)
        os.makedirs(month_out_dir, exist_ok=True)
        labels_out_dir = os.path.join(month_out_dir, "labels")
        os.makedirs(labels_out_dir, exist_ok=True)

        # ===== Process Daily Imagery =====
        # Save each day's data as a separate .pt file.
        files_sorted = sorted(files, key=lambda x: x[0])
        for dt, file in files_sorted:
            date_str = dt.strftime("%Y-%m-%d")
            daily_out_file = os.path.join(month_out_dir, f"data_{date_str}.pt")
            if os.path.exists(daily_out_file):
                print(f"[INFO] Daily file for {date_str} already processed. Skipping.")
                continue
            try:
                tensor = load_tif_as_tensor(file)  # expected shape: (bands, H, W)
                torch.save(tensor, daily_out_file)
                print(f"[SAVED] Daily file for {date_str} -> {daily_out_file}")
            except Exception as e:
                print(f"[ERROR] Failed to process {file}: {e}")

        # ===== Process Raster Labels =====
        raster_label_out_file = os.path.join(labels_out_dir, "raster.pt")
        if os.path.exists(raster_label_out_file):
            print(f"[INFO] Raster label already exists for {month}. Skipping raster label processing...")
        else:
            if raster_tile_folder:
                print(f"[INFO] Processing raster label for {month}...")
                label_date_us = f"{month.replace('-', '_')}_01"
                pattern_us = f"*{label_date_us}.tif"
                candidates_us = glob.glob(os.path.join(raster_tile_folder, pattern_us))

                label_date_hy = f"{month}-01"
                pattern_hy = f"*{label_date_hy}.tif"
                candidates_hy = glob.glob(os.path.join(raster_tile_folder, pattern_hy))

                candidates = candidates_us + candidates_hy

                if candidates:
                    raster_tensor = load_tif_as_tensor(candidates[0])
                    torch.save(raster_tensor, raster_label_out_file)
                    print(f"[SAVED] Raster label for {month} -> {raster_label_out_file}")
                else:
                    print(f"[WARNING] No raster label found for {month} (tried underscores and hyphens).")
            else:
                print(f"[WARNING] Raster tile folder not found for {month}.")

        # ===== Process Vector Labels =====
        if vector_label_folder:
            vector_output_dir = os.path.join(labels_out_dir, "vector")
            os.makedirs(vector_output_dir, exist_ok=True)
            vector_classes = [d for d in os.listdir(vector_label_folder) if os.path.isdir(os.path.join(vector_label_folder, d))]
            for vclass in vector_classes:
                vector_label_out_file = os.path.join(vector_output_dir, f"{vclass}.pt")
                if os.path.exists(vector_label_out_file):
                    print(f"[INFO] Vector label for class {vclass} already exists for {month}. Skipping...")
                    continue
                class_folder = os.path.join(vector_label_folder, vclass)
                tile_folder = None
                if raster_tile_folder:
                    tile_name = os.path.basename(raster_tile_folder)
                    tile_folder = os.path.join(class_folder, tile_name)
                else:
                    subfolders = [d for d in os.listdir(class_folder) if os.path.isdir(os.path.join(class_folder, d))]
                    if subfolders:
                        tile_folder = os.path.join(class_folder, subfolders[0])
                if not tile_folder or not os.path.exists(tile_folder):
                    print(f"[WARNING] Tile folder for vector class {vclass} does not exist for {month}.")
                    continue
                label_date_us = f"{month.replace('-', '_')}_01"
                pattern_us = f"*{label_date_us}.geojson"
                candidates_us = glob.glob(os.path.join(tile_folder, pattern_us))
                label_date_hy = f"{month}-01"
                pattern_hy = f"*{label_date_hy}.geojson"
                candidates_hy = glob.glob(os.path.join(tile_folder, pattern_hy))
                candidates = candidates_us + candidates_hy
                if candidates:
                    vector_data = load_geojson_as_tensor(candidates[0])
                    torch.save(vector_data, vector_label_out_file)
                    print(f"[SAVED] Vector label for {month}, class {vclass} -> {vector_label_out_file}")
                else:
                    print(f"[WARNING] No vector label found for {month}, class {vclass} (tried underscores and hyphens).")
        else:
            print(f"[WARNING] No vector label folder found for {month}.")

def tile_has_nonempty_labels(series_path, label_base_path):
    """
    Checks whether the expected labels for a given tile (PF-SR folder) are non-empty.
    This function computes the expected label folder name from the series_path and checks if its 
    Raster subfolder exists and contains at least one file.
    
    Args:
        series_path (str): Path to the PF-SR folder.
        label_base_path (str): Base path where labels are stored (e.g., raw_data/labels).
        
    Returns:
        bool: True if a non-empty labels folder is found, False otherwise.
    """
    time_series_id = os.path.basename(os.path.dirname(series_path))
    planet_dir = next((part for part in series_path.split(os.sep) if part.startswith("planet.")), None)
    if not planet_dir:
        return False
    planet_id = planet_dir.split('.')[-1]
    label_folder_name = f"{time_series_id}_{planet_id}"
    # Expected raster label folder:
    raster_label_base = os.path.join(label_base_path, label_folder_name, "Labels", "Raster")
    if os.path.isdir(raster_label_base):
        items = os.listdir(raster_label_base)
        if items:
            return True
    return False

def main():
    """
    Entry point for processing all planet datasets found in the raw data root directory.
    Looks for "planet." folders, navigates to "PF-SR" subdirectories, and processes each time series.
    Only processes a PF-SR folder if it has non-empty corresponding labels.
    """
    root_dir = "/media/thibault/DynEarthNet/raw_data"
    label_base = os.path.join(root_dir, "labels")
    output_base = "/media/thibault/DynEarthNet"

    for item in os.listdir(root_dir):
        if item.startswith("planet."):
            planet_path = os.path.join(root_dir, item)
            for dirpath, dirnames, filenames in os.walk(planet_path):
                if os.path.basename(dirpath) == "PF-SR":
                    # Check if this PF-SR folder has non-empty labels
                    if tile_has_nonempty_labels(dirpath, label_base):
                        print(f"[INFO] Processing PF-SR folder: {dirpath}")
                        process_time_series(dirpath, label_base, output_base)
                    else:
                        print(f"[SKIP] Skipping PF-SR folder {dirpath} because its labels are empty or missing.")

if __name__ == "__main__":
    main()
