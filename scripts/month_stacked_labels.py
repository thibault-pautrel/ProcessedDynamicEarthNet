import os
import glob
import re
import torch

def build_month_tensor(month_folder):
    daily_files = sorted(glob.glob(os.path.join(month_folder, "data_*.pt")))
    if not daily_files:
        raise FileNotFoundError(f"No 'data_*.pt' files found in {month_folder}")

    example = torch.load(daily_files[0])  # shape [4, H, W]
    _, H, W = example.shape

    stacked_data = torch.zeros((H, W, 4 * len(daily_files)), dtype=torch.float32)
    for t, file in enumerate(daily_files):
        data = torch.load(file)  # shape [4, H, W]
        stacked_data[:, :, t * 4:(t + 1) * 4] = data.permute(1, 2, 0)

    return stacked_data

def convert_labels_to_class_indices(label_tensor):
    label_tensor = (label_tensor > 127).long()
    class_indices = torch.argmax(label_tensor, dim=0)
    return class_indices

def build_pixel_datasets_for_all_months(root_dir, output_dataset_dir):
    month_regex = re.compile(r'^\d{4}-\d{2}$')
    
    # Parse planet and tile from root_dir (FIXED)
    path_parts = root_dir.strip('/').split('/')
    planet = path_parts[-2]   # e.g., 'planet.13N'
    tile_id = path_parts[-1]  # e.g., '1700_3100_13'

    for entry in sorted(os.listdir(root_dir)):
        month_folder = os.path.join(root_dir, entry)
        if not (os.path.isdir(month_folder) and month_regex.match(entry)):
            continue

        print(f"\n[Processing] {month_folder}")

        try:
            features = build_month_tensor(month_folder)

            label_file = os.path.join(month_folder, "labels", "raster.pt")
            if not os.path.isfile(label_file):
                print(f"[Warning] Label file missing: {label_file}, skipping.")
                continue

            raw_labels = torch.load(label_file)
            labels = convert_labels_to_class_indices(raw_labels)

            # Output path: dataset/unet/{planet}/{tile_id}/{month}/
            out_folder = os.path.join(output_dataset_dir, 'unet', planet, tile_id, entry)
            os.makedirs(out_folder, exist_ok=True)

            output_file = os.path.join(out_folder, f"pixel_dataset_{entry}.pt")

            torch.save({"features": features, "labels": labels}, output_file)

            print(f"[Saved] {output_file}")

        except Exception as e:
            print(f"[Error] Failed to process {month_folder}: {e}")


if __name__ == "__main__":
    # Input: where the raw data is
    raw_root_dir = "/media/thibault/DynEarthNet/planet.11N/1487_3335_13"
    
    # Output: base folder for dataset
    output_dataset_dir = "/media/thibault/DynEarthNet/datasets"

    build_pixel_datasets_for_all_months(raw_root_dir, output_dataset_dir)
