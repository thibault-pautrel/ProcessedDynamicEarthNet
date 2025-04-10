import os,sys
import glob
import torch
import random
import numpy as np
import re
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from UNet.unet_pipeline import UNetLite, train_model, set_seed, eval_epoch, compute_class_weights, FocalLoss
from UNet.unet_from_h5 import SlidingPatchH5Dataset

set_seed(42)

NUM_CLASSES = 4
BATCH_SIZE = 16
FED_EPOCHS = 2
NUM_ROUNDS = 10
DATASET_ROOT = "/media/thibault/DynEarthNet/datasets"
CHECKPOINT_DIR = "/home/thibault/ProcessedDynamicEarthNet/checkpoints/unet"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
tile_ids = sorted([d for d in os.listdir(os.path.join(DATASET_ROOT, "train")) if os.path.isdir(os.path.join(DATASET_ROOT, "train", d))])
NUM_CLIENTS = len(tile_ids)
print(f'Number of total clients: {NUM_CLIENTS}')

CLIENTS_PER_ROUND = 5
CLIENT_SPLITS = {
    f"client_{i+1}": sorted([
        os.path.join(DATASET_ROOT, "train", tile_id, f)
        for f in os.listdir(os.path.join(DATASET_ROOT, "train", tile_id)) if f.endswith(".h5")
    ]) for i, tile_id in enumerate(tile_ids)
}
all_clients = list(CLIENT_SPLITS.items())

def subsample_dataset(dataset, max_samples=1000):
    indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
    return Subset(dataset, indices)

def get_client_loaders(h5_files):
    full_ds = SlidingPatchH5Dataset(h5_files, w_size=19, stride=9, max_samples=1000)
    idxs = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(idxs, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def average_model_weights(model_paths):
    avg_state_dict = None
    for path in model_paths:
        state_dict = torch.load(path, map_location='cpu')
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float()
    for k in avg_state_dict:
        avg_state_dict[k] /= len(model_paths)
    return avg_state_dict

def train_one_client(client_name, h5_files, global_weights, round_idx):
    train_loader, val_loader, test_loader = get_client_loaders(h5_files)
    model = UNetLite(in_channels=112, num_classes=NUM_CLASSES, dropout_p=0.1).to(DEVICE)
    if global_weights:
        model.load_state_dict(global_weights)

    model_name = f"{client_name}_round{round_idx}"
    train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=FED_EPOCHS,
        lr=1e-4,
        num_classes=NUM_CLASSES,
        weight_decay=5e-4
    )
    return os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")

def _train_one_client_wrapper(args):
    name, h5s, weights, round_idx = args
    return train_one_client(name, h5s, weights, round_idx)

def load_loader(h5_list, max_samples=1200):
    ds = SlidingPatchH5Dataset(h5_list, w_size=19, stride=9, max_samples=max_samples)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

if __name__ == "__main__":
    weight_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "unet_federated_round*_latest.pt")))

    resume_round = 0
    global_weights = None

    if weight_ckpts:
        last_path = weight_ckpts[-1]
        match = re.search(r"round(\d+)_latest", os.path.basename(last_path))
        if match:
            resume_round = int(match.group(1)) + 1
            global_weights = torch.load(last_path, map_location=DEVICE)
            print(f"[Resuming] Starting from round {resume_round}")
        else:
            print("[WARN] Could not parse round number. Starting from scratch.")

    best_val_iou = 0.0
    best_weights = None

    val_h5 = glob.glob(os.path.join(DATASET_ROOT, "val", "*", "*.h5"))
    val_loader = load_loader(val_h5, max_samples=1200)
    class_weights = compute_class_weights(val_loader, NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)

    for round_idx in range(resume_round, NUM_ROUNDS):
        print(f"\n--- Federated Round {round_idx} ---")
        selected_clients = random.sample(all_clients, CLIENTS_PER_ROUND)
        client_args = [(name, h5s, global_weights, round_idx) for name, h5s in selected_clients]

        ckpts = []
        with ProcessPoolExecutor(max_workers=CLIENTS_PER_ROUND) as executor:
            futures = {executor.submit(_train_one_client_wrapper, args): args[0] for args in client_args}
            for future in as_completed(futures):
                client_name = futures[future]
                try:
                    ckpt_path = future.result()
                    ckpts.append(ckpt_path)
                except Exception as e:
                    print(f"[ERROR] Client {client_name} failed: {e}")

        if not ckpts:
            print(f"[WARNING] No successful clients in round {round_idx}. Skipping aggregation.")
            continue

        global_weights = average_model_weights(ckpts)
        torch.save(global_weights, os.path.join(CHECKPOINT_DIR, f"unet_federated_round{round_idx}_latest.pt"))

        model = UNetLite(in_channels=112, num_classes=NUM_CLASSES, dropout_p=0.1).to(DEVICE)
        model.load_state_dict(global_weights)
        val_loss, val_acc, val_ious, _, _ = eval_epoch(model, val_loader, criterion, DEVICE, num_classes=NUM_CLASSES)
        mean_iou = np.nanmean(val_ious)
        print(f"Round {round_idx} Val mIoU: {mean_iou:.4f}, Val accuracy: {val_acc:.4f}")
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_weights = {k: v.clone() for k, v in global_weights.items()}

    final_model_path = os.path.join(CHECKPOINT_DIR, "unet_federated_best_val.pt")
    torch.save(best_weights, final_model_path)
    print(f"\nSaved best validation model to {final_model_path}")

    test_h5 = glob.glob(os.path.join(DATASET_ROOT, "test", "*", "*.h5"))
    test_loader = load_loader(test_h5, max_samples=1200)
    model = UNetLite(in_channels=112, num_classes=NUM_CLASSES, dropout_p=0.1).to(DEVICE)
    model.load_state_dict(best_weights)
    print("\n--- Final Evaluation on Test Set ---")
    eval_epoch(model, test_loader, criterion, DEVICE, num_classes=NUM_CLASSES)
