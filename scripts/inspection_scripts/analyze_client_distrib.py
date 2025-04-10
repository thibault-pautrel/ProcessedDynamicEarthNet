import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from spd_datasets import InferenceSlidingCovDataset

# Assume fed_spdnet.py defines this:
from fed_spdnet import CLIENT_SPLITS, DATASET_ROOT  # modify if needed

NUM_CLASSES = 4
OUTPUT_DIR = "/home/thibault/ProcessedDynamicEarthNet/histograms/federated/client_class_distributions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_class_distribution_dataset(dataset):
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    for _, labels in loader:
        labels_np = labels.numpy().astype(np.int64)
        for label in labels_np:
            if 0 <= label < NUM_CLASSES:
                class_counts[label] += 1
    return class_counts


def plot_histogram(class_counts, subset_name, output_dir=None):
    plt.figure()
    x_positions = np.arange(NUM_CLASSES)
    plt.bar(x_positions, class_counts)
    plt.title(f"Class Distribution - {subset_name}")
    plt.xlabel("Class Index")
    plt.ylabel("Pixel Count")
    plt.xticks(x_positions)

    total = np.sum(class_counts)
    for i, count in enumerate(class_counts):
        if count > 0:
            pct = (count / total) * 100
            plt.text(i, count, f"{count}\n({pct:.1f}%)", ha="center", va="bottom")

    if output_dir:
        out_path = os.path.join(output_dir, f"class_distribution_{subset_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved histogram for {subset_name} to {out_path}")
        plt.close()
    else:
        plt.show()

# ---------- MAIN LOOP ----------
for client_name, h5_files in CLIENT_SPLITS.items():
    print(f"[INFO] Processing {client_name} with {len(h5_files)} files...")
    dataset = InferenceSlidingCovDataset(h5_files, w_size=17, stride=7)
    class_counts = compute_class_distribution_dataset(dataset)
    plot_histogram(class_counts, subset_name=client_name, output_dir=OUTPUT_DIR)





