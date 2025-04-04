import os
import sys
import torch
import random
import json
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append("/home/thibault/ProcessedDynamicEarthNet/scripts")
from unet_pipeline import (
    UNetLite, train_model, get_dataloader, UNetCropDataset,
    eval_epoch, display_confusion_matrix, set_seed
)

# Set seed and device
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
T = 28
IN_CHANNELS = 4 * T
NUM_CLASSES = 4
BATCH_SIZE = 4
FED_EPOCHS = 5
FINETUNE_EPOCHS = 10
CLIENTS_PER_ROUND = 2
NUM_ROUNDS = 3

checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Client directories
client_dirs = {
    "mixed": "/media/thibault/DynEarthNet/federated/datasets/mixed",
    "forest": "/media/thibault/DynEarthNet/federated/datasets/forest",
    "water": "/media/thibault/DynEarthNet/federated/datasets/water"
}
all_clients = list(client_dirs.items())

def average_model_weights(model_paths):
    avg_state_dict = None
    n_models = len(model_paths)
    for path in model_paths:
        state_dict = torch.load(path, map_location='cpu')
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float()
    for k in avg_state_dict:
        avg_state_dict[k] /= n_models
    return avg_state_dict

def train_one_client(client_name, data_dir, global_weights, round_idx):
    dataset = UNetCropDataset(split_dir=data_dir, max_T=T, augment=True)
    size = len(dataset)
    train_len = int(size * 0.7)
    val_len = int(size * 0.15)
    test_len = size - train_len - val_len
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

    model = UNetLite(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dropout_p=0.1)
    if global_weights:
        model.load_state_dict(global_weights)

    model_name = f"{client_name}_round{round_idx}"
    train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=FED_EPOCHS,
        lr=1e-4,
        num_classes=NUM_CLASSES,
        weight_decay=5e-4
    )
    return os.path.join(checkpoint_dir, f"{model_name}_best.pt")

# Federated training
global_weights = None
for round_idx in range(NUM_ROUNDS):
    print(f"\n================ Federated Round {round_idx} ================\n")
    selected_clients = random.sample(all_clients, k=CLIENTS_PER_ROUND)
    print(f"[Round {round_idx}] Selected clients: {[name for name, _ in selected_clients]}")
    client_ckpts = [train_one_client(name, path, global_weights, round_idx) for name, path in selected_clients]
    global_weights = average_model_weights(client_ckpts)

# Save final aggregated model
final_model_path = os.path.join(checkpoint_dir, "unet_federated_final.pt")
torch.save(global_weights, final_model_path)
print(f"\n✅ Saved final federated model to: {final_model_path}\n")

# Load and fine-tune
model = UNetLite(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dropout_p=0.1)
model.load_state_dict(torch.load(final_model_path, map_location=device))
model.to(device)
model_name = "unet_federated_finetuned"

val_dir = "/media/thibault/DynEarthNet/full_data/datasets/val"
test_dir = "/media/thibault/DynEarthNet/full_data/datasets/test"
val_full_dataset = UNetCropDataset(val_dir, max_T=T, augment=True)
size = len(val_full_dataset)
train_len = int(size * 0.8)
val_len = size - train_len
train_ds, val_ds = torch.utils.data.random_split(val_full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = UNetCropDataset(test_dir, max_T=T, augment=False)
test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_model(
    model=model,
    model_name=model_name,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    epochs=FINETUNE_EPOCHS,
    lr=1e-4,
    num_classes=NUM_CLASSES,
    weight_decay=5e-4
)

