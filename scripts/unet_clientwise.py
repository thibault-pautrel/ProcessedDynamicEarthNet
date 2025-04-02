#!/usr/bin/env python
import os
import sys
import torch
from torch.utils.data import random_split

# Adjust this path so Python can locate unet_pipeline.py if needed:
sys.path.append("/home/thibault/ProcessedDynamicEarthNet")

# Import the relevant pieces from your unet_pipeline
from unet_pipeline import (
    UNetCropDataset,
    UNet,
    get_dataloader,
    train_model,
    set_seed
)

########################################
# Configuration
########################################
SEED = 42
set_seed(SEED)

# Point this to the directory where your monthly .pt files for a single client are stored:
client_group_dir = "/home/thibault/ProcessedDynamicEarthNet/subsampled_data/clients/unet/water"

# Name your model run
model_name = "unet_water_group"

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

########################################
# Load dataset from the client group directory
########################################
print(f"Loading dataset from: {client_group_dir}")
dataset = UNetCropDataset(
    split_dir=client_group_dir,
    pattern="pixel_dataset_*.pt",
    final_H=118,
    final_W=118,
    max_T=T
)

# 70/15/15 split
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

print(f"Total samples: {total_size} -> "
      f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Build DataLoaders
train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
val_loader   = get_dataloader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = get_dataloader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

########################################
# Build and train the UNet model
########################################
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
