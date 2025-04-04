#!/usr/bin/env python

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_class_distribution(dir_path):
    """
    Recursively find all 'pixel_dataset_*.pt' files in dir_path,
    load each file's ['labels'] tensor, and accumulate the total
    pixel counts for classes 0..3

    Returns:
        A 1D NumPy array of length 4 with class counts.
    """
    class_counts = np.zeros(4, dtype=np.int64)

    # Recursively gather all .pt files named 'pixel_dataset_*.pt'
    pt_files = glob.glob(os.path.join(dir_path, "**", "pixel_dataset_*.pt"), recursive=True)

    for pt_file in pt_files:
        data = torch.load(pt_file)
        # data["labels"] is a 2D tensor [H, W] with values in [0..6]
        labels = data["labels"].flatten()  # shape [H*W]
        labels_np = labels.cpu().numpy().astype(np.int64)

        # Update counts
        for c in range(4):  # classes 0..3
            class_counts[c] += np.sum(labels_np == c)

    return class_counts

def plot_histogram(class_counts, subset_name, output_dir=None):
    """
    Create a bar plot showing the frequency of each class (0..6).

    Args:
        class_counts (ndarray): shape [4], total pixel counts per class
        subset_name (str)     : one of 'train', 'val', or 'test'
        output_dir (str)      : if provided, save the figure there; else show inline
    """
    plt.figure()
    x_positions = np.arange(4)
    plt.bar(x_positions, class_counts)
    plt.title(f"Class Distribution in {subset_name}")
    plt.xlabel("Class Index")
    plt.ylabel("Pixel Count")
    plt.xticks(x_positions, [str(i) for i in range(4)])

    total_pixels = np.sum(class_counts)
    for i, count in enumerate(class_counts):
        if count > 0:
            pct = (count / total_pixels) * 100
            plt.text(i, count, f"{count}\n({pct:.1f}%)", 
                     ha="center", va="bottom", fontsize=8)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"class_distribution_{subset_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved histogram to {out_path}")
        plt.close()
    else:
        plt.show()

def main():
    # Root folder containing train/, val/, and test/ subdirectories
    data_root = "/media/thibault/DynEarthNet/federated/datasets"
    subsets = ["mixed", "forest", "water"]

    # Directory for saving histograms
    output_dir = "./histograms"

    for subset_name in subsets:
        dir_path = os.path.join(data_root, subset_name)
        if not os.path.isdir(dir_path):
            print(f"[WARNING] {dir_path} does not exist, skipping.")
            continue

        print(f"\n[INFO] Computing class distribution for {subset_name.upper()} ...")
        class_counts = compute_class_distribution(dir_path)

        print(f"   Counts (class 0..3): {class_counts}")
        print(f"   Total pixels: {np.sum(class_counts)}")

        # Make and save a histogram
        plot_histogram(class_counts, subset_name=subset_name, output_dir=output_dir)

if __name__ == "__main__":
    main()
