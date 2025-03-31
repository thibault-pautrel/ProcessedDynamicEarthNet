#!/usr/bin/env python

import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # <-- for padding
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Additional metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# SEEDING
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

###############################################################################
# Analyze class distribution
###############################################################################
def analyze_class_distribution(labels, label_type="Data"):
    counts = torch.bincount(labels.flatten().cpu())
    print(f"{label_type} Label Distribution: {counts.tolist()}")

###############################################################################
# UNet Model with pad_and_cat fix
###############################################################################
class UNet(nn.Module):
    """
    A standard U-Net model for semantic segmentation with padded skip connections.

    Input:
        - x: Tensor of shape [B, C, H, W], typically [B, 112, 118, 118]
    Output:
        - logits: Tensor of shape [B, num_classes, H, W], typically [B, 7, 118, 118]

    Skip connections are padded using `pad_and_cat` to ensure spatial alignment.
    """

    def __init__(self, in_channels=112, num_classes=7):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.middle = self.conv_block(512, 1024)

        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def pad_and_cat(self, enc_feat, dec_feat):
        """
        Pad dec_feat (the upsampled decoder feature map) 
        so it matches enc_feat's spatial size, then concat along channels.
        """
        diffY = enc_feat.size(2) - dec_feat.size(2)
        diffX = enc_feat.size(3) - dec_feat.size(3)
        dec_feat = F.pad(
            dec_feat,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([enc_feat, dec_feat], dim=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Middle
        m = self.middle(self.pool(e4))

        # Decoder
        d4 = self.up4(m)
        d4 = self.pad_and_cat(e4, d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.pad_and_cat(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.pad_and_cat(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.pad_and_cat(e1, d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

###############################################################################
# UNet Dataset that looks for .pt files in a single directory
###############################################################################
class UNetCropDataset(Dataset):
    """
    Loads and processes monthly pixel datasets from .pt files into UNet-ready tensors
    from a single directory (and its subdirectories).

    Each sample:
        - features: [C, 118, 118], where C = 4 * T = 112 (i.e. considering 28 days/month)
        - labels:   [118, 118] (integer class indices)

    Steps:
        - Permute feature tensor from [H, W, C] to [C, H, W]
        - Crop top-left to fixed shape [118, 118]
        - Slice input channels to 112 if needed (e.g. from 124→112)

    Args:
        split_dir (str): Directory containing the .pt files for train/val/test.
        pattern (str): Filename glob for the monthly data, e.g. 'pixel_dataset_*.pt'
        final_H/W (int): Final spatial resolution (crop target), default 118×118
        max_T (int): Number of time steps (4×max_T = #channels), default 28 → 112 channels
    """

    def __init__(
        self,
        split_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=28
    ):
        super().__init__()
        self.final_H = final_H
        self.final_W = final_W
        self.max_C = 4 * max_T  # 112 channels
        self.files = sorted(
            glob.glob(os.path.join(split_dir, "**", pattern), recursive=True)
        )
        if not self.files:
            raise ValueError(
                f"No .pt files found under {split_dir} with pattern={pattern}"
            )

        print(f"UNetCropDataset: Found {len(self.files)} .pt files in {split_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path)

        feats = data["features"].permute(2, 0, 1).float()  # [C, H, W]
        labs  = data["labels"].long()                      # [H, W]

        C, H, W = feats.shape
        # Crop to 118×118 
        if H < self.final_H or W < self.final_W:
            raise ValueError(
                f"Cannot crop from shape=({C},{H},{W}) to ({self.final_H},{self.final_W}). "
                f"File={file_path}"
            )
        feats = feats[:, :self.final_H, :self.final_W]  # [C,118,118]
        labs  = labs[:self.final_H, :self.final_W]      # [118,118]

        # Slice channels down to 112 if needed
        if C < self.max_C:
            raise ValueError(
                f"File has only {C} channels but we need {self.max_C}. File={file_path}"
            )
        if C > self.max_C:
            feats = feats[:self.max_C, :, :]  # => [112,118,118]

        return feats, labs

###############################################################################
# DataLoader Helpers
###############################################################################
def get_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0):
    """
    Wrap a dataset into a PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

###############################################################################
# Metrics
###############################################################################
def calculate_iou(preds, labels, num_classes=7):
    ious = []
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds_np == cls, labels_np == cls).sum()
        union = np.logical_or(preds_np == cls, labels_np == cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.array(ious)

def display_confusion_matrix(y_true, y_pred, split_tag, model_name, num_classes=7):
    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{split_tag}.png")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = (cm_normalized / row_sums) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=list(range(num_classes)),
        yticklabels=list(range(num_classes)),
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"Row-normalized Confusion Matrix (%)\n{model_name} [{split_tag}]")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Normalized confusion matrix saved to {save_path}")

def plot_loss_curves(train_losses, val_losses, model_name, split_tag):
    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/loss_curves"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"loss_curve_{model_name}_{split_tag}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves ({model_name}) [{split_tag}]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curves saved to {save_path}")

###############################################################################
# Training + Evaluation
###############################################################################
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch using standard cross-entropy loss.

    Inputs per batch:
        - feats: [B, 112, 118, 118]
        - labels: [B, 118, 118]
    """
    model.train()
    total_loss, total_correct, total_pixels = 0.0, 0, 0

    for feats, labels in dataloader:
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(feats)  # [B,7,118,118]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_pixels += labels.numel()
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc  = total_correct / total_pixels
    return avg_loss, avg_acc

def eval_epoch(model, dataloader, criterion, device, num_classes=7):
    """
    Evaluates the model on a split (val/test).

    Returns:
        - avg_loss
        - avg_acc
        - ious (per-class)
        - all_preds (flattened)
        - all_labels (flattened)
    """
    model.eval()
    total_loss, total_correct, total_pixels = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)

            outputs = model(feats)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_pixels += labels.numel()
            total_loss += loss.item() * labels.size(0)

            all_preds.append(preds.cpu().view(-1))
            all_labels.append(labels.cpu().view(-1))

    all_preds  = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    ious = calculate_iou(all_preds, all_labels, num_classes=num_classes)
    print(f"IoU per class: {ious}")
    analyze_class_distribution(all_labels, label_type="Validation")
    analyze_class_distribution(all_preds, label_type="Predictions")

    return (total_loss / len(dataloader.dataset),
            total_correct / total_pixels,
            ious,
            all_preds,
            all_labels)

def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-3,
    num_classes=7
):
    """
    Full training and evaluation loop for UNet:
    - Train+Val across epochs
    - Save best checkpoint (by mIoU on val)
    - Final test evaluation
    - Confusion matrix, classification report, AUC
    - Loss curve plotting
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0
    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mean_val_iou = np.nanmean(val_ious)
        print(
            f"[{model_name}] Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {mean_val_iou:.4f}"
        )

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            save_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    # Final test
    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device, num_classes
    )
    print(
        f"[{model_name}] Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
        f"mIoU: {np.nanmean(test_ious):.4f}"
    )

    # Confusion matrix & classification report
    display_confusion_matrix(
        test_labels.numpy(), test_preds.numpy(),
        split_tag="test",  # label the confusion matrix figure
        model_name=model_name, 
        num_classes=num_classes
    )
    print(classification_report(
        test_labels.numpy(), test_preds.numpy(), zero_division=0
    ))
    # --- Save final test metrics to JSON ---
    import json

    #Convert classification_report into a dict so we can extract macro precision/recall/f1
    report_dict = classification_report(
        test_labels.numpy(),
        test_preds.numpy(),
        zero_division=0,
        output_dict=True
    )

    # 2) Gather metrics of interest
    test_accuracy = test_acc
    test_mIoU = float(np.nanmean(test_ious))
    precision_macro = report_dict["macro avg"]["precision"]
    recall_macro = report_dict["macro avg"]["recall"]
    f1_macro = report_dict["macro avg"]["f1-score"]

    metrics_dict = {
        "test_accuracy": test_accuracy,
        "test_mIoU": test_mIoU,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }

    # Save to JSON in your desired directory
    save_metrics_path = f"/home/thibault/ProcessedDynamicEarthNet/test_metrics_{model_name}.json"
    with open(save_metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Test metrics saved to {save_metrics_path}")

    # Attempt multi-class AUC
    try:
        probs = []
        model.eval()
        with torch.no_grad():
            for feats, _ in test_loader:
                feats = feats.to(device)
                out = model(feats)
                sm = torch.softmax(out, dim=1)  # [B,7,118,118]
                sm = sm.permute(0,2,3,1).reshape(-1, num_classes)  # => [B*118*118,7]
                probs.append(sm.cpu())
        probs = torch.cat(probs, dim=0).numpy()

        test_labels_np = test_labels.numpy()  # shape [B*118*118]
        one_hot = np.eye(num_classes)[test_labels_np]  # [N,7]
        auc_val = roc_auc_score(one_hot, probs, average='macro', multi_class='ovr')
        print(f"Multiclass AUC (macro, OVR): {auc_val:.4f}")
    except Exception as e:
        print("AUC could not be computed:", e)

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, model_name, "train_val")

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories holding the .pt files for each split
    train_dir = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/train"
    val_dir   = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/val"
    test_dir  = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/test"

    # Number of monthly time steps = 28, so channels = 4*T = 112
    T = 28  
    model_name = "unet_crop_fixed"

    # Build train, val, test datasets
    train_dataset = UNetCropDataset(
        split_dir=train_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T
    )

    val_dataset = UNetCropDataset(
        split_dir=val_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T
    )

    test_dataset = UNetCropDataset(
        split_dir=test_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T
    )

    #==========================================
    #    Subsampling the datasets
    #===========================================

    from torch.utils.data import Subset

    # Subsample sizes for quick sanity test
    N_TRAIN = len(train_dataset)//2
    N_VAL = len(val_dataset)//2
    N_TEST = len(test_dataset)//2
    print(f"Subsampling train/val/test datasets to {N_TRAIN}/{N_VAL}/{N_TEST} samples.")

    # Use fixed random seed for reproducibility
    rng = torch.Generator().manual_seed(42)

    # Subsample indices
    train_subset = Subset(train_dataset, torch.randperm(len(train_dataset), generator=rng)[:N_TRAIN])
    val_subset   = Subset(val_dataset,   torch.randperm(len(val_dataset), generator=rng)[:N_VAL])
    test_subset  = Subset(test_dataset,  torch.randperm(len(test_dataset), generator=rng)[:N_TEST])

    train_loader = get_dataloader(train_subset, batch_size=1, shuffle=True,  num_workers=2)
    val_loader   = get_dataloader(val_subset,   batch_size=1, shuffle=False, num_workers=2)
    test_loader  = get_dataloader(test_subset,  batch_size=1, shuffle=False, num_workers=2)



    # Create data loaders
    #train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=True,  num_workers=0)
    #val_loader   = get_dataloader(val_dataset,   batch_size=2, shuffle=False, num_workers=0)
    #test_loader  = get_dataloader(test_dataset,  batch_size=2, shuffle=False, num_workers=0)

    # Instantiate UNet: in_channels=112
    unet_model = UNet(in_channels=4*T, num_classes=7)

    # Train and evaluate
    train_model(
        model=unet_model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=30,   
        lr=1e-3,
        num_classes=7
    )
