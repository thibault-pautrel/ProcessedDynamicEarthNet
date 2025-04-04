#!/usr/bin/env python

import os
import torch
import sys
from torch.utils.data import random_split

# 1) Import from your pipeline module:
sys.path.append("/home/thibault/ProcessedDynamicEarthNet/scripts")  
from basic_spdnet_pipeline import (
    set_seed,
    OnTheFlyLedoitCovDataset,
    SPDNet3BiRe,    # or SPDNet2BiRe
    get_loader,
    train_model
)


##############################
# Configuration
##############################
SEED = 42
set_seed(SEED)

# Point this to one of your client directories: "urban", "mixed", "forest", or "water"
CLIENT_GROUP_DIR = "/media/thibault/DynEarthNet/federated/datasets/water"
MODEL_NAME = "spdnet_water"
PLANET_FOLDER = "water"

# Covariance hyperparameters
T = 28                       # number of days to use
INPUT_DIM = 4 * T            # 4 bands × T daily images
NUM_CLASSES = 4             # we have 4-class labeling
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################
# Prepare the Dataset
##############################
print(f"Loading client dataset from: {CLIENT_GROUP_DIR}")

dataset = OnTheFlyLedoitCovDataset(
    split_dir=CLIENT_GROUP_DIR,
    max_T=T
)
total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

print(f"[INFO] Found {total_size} monthly samples.")
print(f"       Train: {train_size}, Val: {val_size}, Test: {test_size}")

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = get_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = get_loader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = get_loader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


##############################
# Build and Train SPDNet
##############################
print("\n--- Building SPDNet Model for this client ---")
model = SPDNet3BiRe(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    epsilon=1e-3,
    use_batch_norm=False,
    p=0.3
)
# Alternatively, to use a 2-layer SPDNet instead:
# model = SPDNet2BiRe(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, epsilon=1e-3)

train_model(
    model=model,
    model_name=MODEL_NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    num_classes=NUM_CLASSES,
    planet_folder=PLANET_FOLDER
)
