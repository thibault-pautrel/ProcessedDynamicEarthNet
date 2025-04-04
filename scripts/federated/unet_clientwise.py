#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# Adjust this path so Python can locate unet_pipeline.py
sys.path.append("/home/thibault/ProcessedDynamicEarthNet")

# Import the relevant pieces from your unet_pipeline
from unet_pipeline import (
    UNetCropDataset,
    UNet,
    get_dataloader,
    train_model,
    set_seed
)

##################################################
# Configuration
##################################################
SEED = 42
set_seed(SEED)

# Point this to the directory where your monthly .pt files for a single client are stored:
client_group_dir = "/home/thibault/ProcessedDynamicEarthNet/subsampled_data/clients/unet/mixed"

# Name your model run
model_name = "unet_mixed_group"

# U-Net expects in_channels = 4*T if T=28 daily steps each month
T = 28
in_channels = 4 * T
num_classes = 7

# Training hyperparameters
batch_size = 1
num_workers = 2
epochs = 15
lr = 1e-4
weight_decay = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################
# New helper functions for class distribution
##################################################
def compute_class_distribution_unet(dataset, num_classes=7):
    """
    Loops over the entire dataset, accumulating how many pixels belong to each class (0..6).
    Each dataset[i] = (features, labels), with labels shape [H, W].
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        _, label_tensor = dataset[i]
        # Convert to CPU NumPy and flatten
        label_np = label_tensor.cpu().numpy().astype(np.int64).ravel()
        hist = np.bincount(label_np, minlength=num_classes)
        class_counts += hist
    return class_counts

def plot_class_histogram(class_counts, subset_name, output_dir="./histograms"):
    """
    Given a 1D array of class_counts for [0..6], produce a bar chart
    with absolute counts and percentages. Saves as a PNG in 'output_dir'.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    x = np.arange(len(class_counts))
    plt.bar(x, class_counts)

    plt.title(f"Class Distribution ({subset_name})")
    plt.xlabel("Class Index")
    plt.ylabel("Pixel Count")
    plt.xticks(x, [str(i) for i in range(len(class_counts))])

    total = class_counts.sum()
    for i, cnt in enumerate(class_counts):
        if cnt > 0:
            pct = (cnt / total) * 100
            plt.text(i, cnt, f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(output_dir, f"class_distribution_{model_name}_{subset_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {subset_name} histogram to: {out_path}")

##################################################
# Main
##################################################
if __name__ == "__main__":
    print(f"Loading dataset from: {client_group_dir}")

    # Build the dataset from the entire client group directory
    dataset = UNetCropDataset(
        split_dir=client_group_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T,
        augment=False  # set True if you want data augmentation
    )

    # 70/15/15 split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    print(f"Total samples: {total_size} -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Random split with a fixed seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    ##################################################
    # Compute & plot class distribution
    ##################################################
    print("\n--- Computing Class Distributions ---")

    train_counts = compute_class_distribution_unet(train_dataset, num_classes=num_classes)
    plot_class_histogram(train_counts, "train")

    val_counts = compute_class_distribution_unet(val_dataset, num_classes=num_classes)
    plot_class_histogram(val_counts, "val")

    test_counts = compute_class_distribution_unet(test_dataset, num_classes=num_classes)
    plot_class_histogram(test_counts, "test")

    ##################################################
    # Build DataLoaders
    ##################################################
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = get_dataloader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = get_dataloader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ##################################################
    # Build and train the UNet model
    ##################################################
    print("\n--- Building U-Net Model for single client group data ---")
    model = UNet(in_channels=in_channels, num_classes=num_classes)

    train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        num_classes=num_classes,
        weight_decay=weight_decay
    )
