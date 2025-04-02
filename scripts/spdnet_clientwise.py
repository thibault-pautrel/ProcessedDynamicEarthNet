import os
import sys
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader


sys.path.append("/home/thibault/ProcessedDynamicEarthNet")

from basic_spdnet_pipeline import (
    MonthlyCovarianceDataset,
    SPDNet3BiRe,
    get_loader,
    train_model,
    set_seed
)

########################################
# Configuration
########################################
SEED = 42
set_seed(SEED)

client_group_dir = "/home/thibault/ProcessedDynamicEarthNet/subsampled_data/clients/spdnet_monthly/water"
model_name = "spdnet_water_group"
planet_folder = "water"

T = 28
input_dim = 4 * T
batch_size = 2
num_workers = 1
epochs = 15
lr = 1e-4
num_classes = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# Load dataset from client group directory
########################################
print(f"Loading dataset from: {client_group_dir}")

dataset = MonthlyCovarianceDataset(
    split_dir=client_group_dir,
    block_pattern="cov_label_*.pt",
    max_T=T
)

# Determine split sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print(f"Total samples: {total_size} -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Split dataset with a fixed seed for reproducibility
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = get_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = get_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = get_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

########################################
# Build and Train Model on Client Group Data
########################################
print("\n--- Building SPDNet Model for mixed tile group ---")
model = SPDNet3BiRe(
    input_dim=input_dim,
    num_classes=num_classes,
    epsilon=1e-3,
    use_batch_norm=False  
)

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
    planet_folder=planet_folder
)

