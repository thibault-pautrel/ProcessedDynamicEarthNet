import os
import sys
sys.path.append("/home/thibault/ProcessedDynamicEarthNet/scripts")
import torch
import numpy as np
from torch.utils.data import Subset
from cov_dataset_class import OnTheFlyLedoitCovDataset
from basic_spdnet_pipeline import (
    SPDNet2BiRe, SPDNet3BiRe,
    train_model, get_loader,
    set_seed
)
from class_distrib_histogram import plot_histogram

# ========== CONFIG ==========
TRAIN_DIR = "/media/thibault/DynEarthNet/full_data/datasets/train"
VAL_DIR   = "/media/thibault/DynEarthNet/full_data/datasets/val"
TEST_DIR  = "/media/thibault/DynEarthNet/full_data/datasets/test"

T = 28
INPUT_DIM = 4 * T
NUM_CLASSES = 4
BATCH_SIZE = 2
EPOCHS = 10

MODEL_NAME = "spdnet3BiRe"  # or "spdnet2BiRe"
USE_SPDNET_3 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

def compute_class_distribution(dataset):
    """
    Accumulate class counts over the entire dataset by iterating 
    over each item and counting labels in [H',W'].
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, labels in dataset:
        labels_np = labels.cpu().numpy()
        for i in range(NUM_CLASSES):
            counts[i] += np.sum(labels_np == i)
    return counts

def main():
    set_seed(42)

    # --- Build train/val/test datasets ---
    train_ds = OnTheFlyLedoitCovDataset(split_dir=TRAIN_DIR, max_T=T)
    val_ds   = OnTheFlyLedoitCovDataset(split_dir=VAL_DIR,   max_T=T)
    test_ds  = OnTheFlyLedoitCovDataset(split_dir=TEST_DIR,  max_T=T)

    print(f"[INFO] Loaded dataset sizes:")
    print(f"  Train: {len(train_ds)} tile-months")
    print(f"  Val:   {len(val_ds)} tile-months")
    print(f"  Test:  {len(test_ds)} tile-months\n")

    # --- Optional subsampling (quick sanity check) ---
    N_TRAIN = len(train_ds)
    N_VAL   = len(val_ds)  
    N_TEST  = len(test_ds) 

    rng = torch.Generator().manual_seed(42)

    train_sub = Subset(train_ds, torch.randperm(len(train_ds), generator=rng)[:N_TRAIN])
    val_sub   = Subset(val_ds,   torch.randperm(len(val_ds),   generator=rng)[:N_VAL])
    test_sub  = Subset(test_ds,  torch.randperm(len(test_ds),  generator=rng)[:N_TEST])

    # --- Build DataLoaders ---
    train_loader = get_loader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_loader(val_sub,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = get_loader(test_sub,  batch_size=BATCH_SIZE, shuffle=False)

    # --- Build the model ---
    if USE_SPDNET_3:
        model = SPDNet3BiRe(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                            epsilon=1e-3, use_batch_norm=True)
        name = "spdnet3BiRe"
    else:
        model = SPDNet2BiRe(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                            epsilon=1e-3, use_batch_norm=True)
        name = "spdnet2BiRe"

    print(f"\n[INFO] Training model: {name}")

    # --- Optionally train the model here ---
    train_model(
        model=model,
        model_name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=7e-3,         # as in basic_spdnet_pipeline
        weight_decay=1e-4,
        num_classes=NUM_CLASSES,
        planet_folder="onthefly_run"  # for naming outputs
    )

if __name__ == "__main__":
    main()
