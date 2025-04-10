#!/usr/bin/env python

import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

########################################
# Matching SPDNet pipeline's class imbalance handling
########################################
def compute_class_weights(loader, num_classes):
    """
    Match the SPDNet approach: use 1.0 / log1p(class_counts).
    If a class has zero pixels, weight=0 is assigned for that class.
    """
    class_counts = torch.zeros(num_classes)
    for feats, labels in loader:
        labels = labels.view(-1)
        counts = torch.bincount(labels, minlength=num_classes)
        class_counts += counts

    with torch.no_grad():
        class_weights = 1.0 / torch.log1p(class_counts)
        # If some class has zero pixels
        class_weights[torch.isinf(class_weights)] = 0.0

    if (class_counts == 0).any():
        missing = torch.where(class_counts == 0)[0].tolist()
        print(f"[WARNING] No pixel occurrences for classes: {missing}. Setting their weights to 0.")

    return class_weights

########################################
# Focal Loss
########################################
class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    gamma = focusing parameter
    alpha can be a tensor of per-class weights (similar to class_weights).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # can be a 1D tensor [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C], raw logits
        targets: [N]
        """
        log_probs = F.log_softmax(inputs, dim=1)   # shape [N, C]
        probs = torch.exp(log_probs)               # shape [N, C]

        focal_weight = (1.0 - probs)**self.gamma   # shape [N, C]
        # Gather log_probs at target indices
        log_probs_target = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight_target = focal_weight.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_target = self.alpha[targets]     # shape [N]
            focal_loss = -alpha_target * focal_weight_target * log_probs_target
        else:
            focal_loss = -focal_weight_target * log_probs_target

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

###############################################################################
# U-Net and Lighter U-Net
###############################################################################
class UNet(nn.Module):
    def __init__(self, in_channels=112, num_classes=4, dropout_p=0.2):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64, dropout_p)
        self.encoder2 = self.conv_block(64, 128, dropout_p)
        self.encoder3 = self.conv_block(128, 256, dropout_p)
        self.encoder4 = self.conv_block(256, 512, dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.middle = self.conv_block(512, 1024, dropout_p)

        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512, dropout_p)

        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256, dropout_p)

        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128, dropout_p)

        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64, dropout_p)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def pad_and_cat(self, enc_feat, dec_feat):
        diffY = enc_feat.size(2) - dec_feat.size(2)
        diffX = enc_feat.size(3) - dec_feat.size(3)
        dec_feat = F.pad(
            dec_feat,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([enc_feat, dec_feat], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        m = self.middle(self.pool(e4))

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


class UNetLite(nn.Module):
    def __init__(self, in_channels=112, num_classes=4, dropout_p=0.2):
        super(UNetLite, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 32, dropout_p)
        self.encoder2 = self.conv_block(32, 64, dropout_p)
        self.encoder3 = self.conv_block(64, 128, dropout_p)
        self.encoder4 = self.conv_block(128, 256, dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.middle = self.conv_block(256, 512, dropout_p)

        self.up4 = self.up_conv(512, 256)
        self.dec4 = self.conv_block(512, 256, dropout_p)

        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(256, 128, dropout_p)

        self.up2 = self.up_conv(128, 64)
        self.dec2 = self.conv_block(128, 64, dropout_p)

        self.up1 = self.up_conv(64, 32)
        self.dec1 = self.conv_block(64, 32, dropout_p)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def pad_and_cat(self, enc_feat, dec_feat):
        diffY = enc_feat.size(2) - dec_feat.size(2)
        diffX = enc_feat.size(3) - dec_feat.size(3)
        dec_feat = F.pad(
            dec_feat,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([enc_feat, dec_feat], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        m = self.middle(self.pool(e4))

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
# UNetCropDataset: 4-class data + 93×93 region
###############################################################################
class UNetCropDataset(Dataset):
    """
    Similar cropping logic as OnTheFlyLedoitCovDataset:
    - We skip 5 pixels at each edge (from 1024×1024) => final is 93×93 center.
    - 4 classes total (0..3).
    - Optional random flips/rotations if 'augment=True'.
    """

    def __init__(
        self,
        split_dir,
        pattern="pixel_dataset_*.pt",
        max_T=28,
        augment=False
    ):
        super().__init__()
        self.max_C = 4 * max_T  # 112 channels
        self.files = sorted(
            glob.glob(os.path.join(split_dir, "**", pattern), recursive=True)
        )
        self.augment = augment
        if not self.files:
            raise ValueError(
                f"No .pt files found under {split_dir} with pattern={pattern}"
            )

        print(f"UNetCropDataset: Found {len(self.files)} .pt files in {split_dir}.")

        # We'll define the final crop size: 93×93, skipping 5 px from each edge
        self.crop_top = 5
        self.crop_left = 5
        self.crop_size = 93

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path)  # contains {"features", "labels"}

        feats = data["features"].permute(2, 0, 1).float()  # [C, 1024, 1024]
        labs  = data["labels"].long()                      # [1024, 1024]

        C, H, W = feats.shape
        if H < self.crop_top + self.crop_size or W < self.crop_left + self.crop_size:
            raise ValueError(
                f"Image {file_path} is too small ({H}×{W}) to crop 93×93 skipping 5 px boundary."
            )

        # Crop to match OnTheFlyLedoitCovDataset’s 93×93 center region
        feats = feats[
            : self.max_C,
            self.crop_top : self.crop_top + self.crop_size,
            self.crop_left: self.crop_left + self.crop_size
        ]
        labs = labs[
            self.crop_top : self.crop_top + self.crop_size,
            self.crop_left: self.crop_left + self.crop_size
        ]

        # Ensure we have all 112 channels
        C_cropped = feats.shape[0]
        if C_cropped < self.max_C:
            raise ValueError(
                f"File {file_path} has only {C_cropped} channels, need {self.max_C}."
            )

        # Data augmentation
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                feats = torch.flip(feats, dims=[2])
                labs  = torch.flip(labs,  dims=[1])
            # vertical flip
            if random.random() < 0.5:
                feats = torch.flip(feats, dims=[1])
                labs  = torch.flip(labs,  dims=[0])
            # random 90° rotation
            k = random.randint(0, 3)
            if k > 0:
                feats = torch.rot90(feats, k, dims=[1, 2])
                labs  = torch.rot90(labs,  k, dims=[0, 1])
                
        feats = (feats - feats.mean()) / (feats.std() + 1e-8)

        return feats, labs

###############################################################################
# DataLoader
###############################################################################
def get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

###############################################################################
# Metrics (4 classes)
###############################################################################
def calculate_iou(preds, labels, num_classes=4):
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

def display_confusion_matrix(y_true, y_pred, split_tag, model_name, num_classes=4):
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
# Train + Eval
###############################################################################
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for feats, labels in dataloader:
        feats = feats.to(device)    # [B, C, H, W]
        labels = labels.to(device)  # [B] (per-patch labels)

        optimizer.zero_grad()
        outputs = model(feats)      # [B, C, H, W]

        # Flatten for loss
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])  # [B*H*W, C]
        labels_flat = labels.view(-1)                                             # [B] → matches patch-level

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)

        # Accuracy for patch-level classification
        patch_logits = outputs.mean(dim=(2, 3))  # [B, C]
        preds = patch_logits.argmax(dim=1)       # [B]

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc
def eval_epoch(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)     # [B, C, H, W]
            labels = labels.to(device)   # [B] (per-patch labels)

            outputs = model(feats)       # [B, C, H, W]

            # Flatten for loss
            outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])  # [B*H*W, C]
            labels_flat = labels.view(-1)                                             # [B]

            loss = criterion(outputs_flat, labels_flat)
            running_loss += loss.item() * feats.size(0)

            # Accuracy for patch-level classification
            patch_logits = outputs.mean(dim=(2, 3))  # [B, C]
            preds = patch_logits.argmax(dim=1)       # [B]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute IoU
    ious = []
    preds_np = all_preds.numpy()
    labels_np = all_labels.numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds_np == cls, labels_np == cls).sum()
        union = np.logical_or(preds_np == cls, labels_np == cls).sum()
        ious.append(intersection / union if union else float('nan'))

    return epoch_loss, epoch_acc, np.array(ious), all_preds, all_labels


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-4,
    num_classes=4,
    weight_decay=1e-4
    ):
    model.to(device)

    # -- Use the same compute_class_weights as SPDNet approach
    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    print("Computed class weights:", class_weights)

    # Focal loss with log-based alpha
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    # If you prefer CrossEntropy:  criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2
    )

    early_stopping_patience=5
    best_val_loss=float('inf')
    trigger_times=0

    best_val_iou = 0.0
    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints/unet"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        mean_val_iou = np.nanmean(val_ious)
        print(
            f"[{model_name}] Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {mean_val_iou:.4f}"
        )

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            trigger_times=0
            save_path=os.path.join(checkpoint_dir,f"{model_name}_best.pt")
            torch.save(model.state_dict(),save_path)
            print(f" -> Saved best model to {save_path}")
        else:
            trigger_times+=1
            if trigger_times>=early_stopping_patience:
                print('Early stopping triggered. Stopping training.')
                break
    # Final test
    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device, num_classes
    )
    print(
        f"[{model_name}] Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
        f"mIoU: {np.nanmean(test_ious):.4f}"
    )

    display_confusion_matrix(
        test_labels.numpy(), test_preds.numpy(),
        split_tag="test",
        model_name=model_name,
        num_classes=num_classes
    )
    print(classification_report(test_labels.numpy(), test_preds.numpy(), zero_division=0))

    # Save final test metrics
    import json
    report_dict = classification_report(
        test_labels.numpy(),
        test_preds.numpy(),
        zero_division=0,
        output_dict=True
    )
    test_accuracy = test_acc
    test_mIoU = float(np.nanmean(test_ious))
    precision_macro = report_dict["macro avg"]["precision"]
    recall_macro    = report_dict["macro avg"]["recall"]
    f1_macro        = report_dict["macro avg"]["f1-score"]

    metrics_dict = {
        "test_accuracy": test_accuracy,
        "test_mIoU": test_mIoU,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }
    save_metrics_path = f"/home/thibault/ProcessedDynamicEarthNet/eval_metrics/test_metrics_{model_name}.json"
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
                sm = torch.softmax(out, dim=1)  # [B,4,93,93]
                sm = sm.permute(0, 2, 3, 1).reshape(-1, num_classes)
                probs.append(sm.cpu())
        probs = torch.cat(probs, dim=0).numpy()

        test_labels_np = test_labels.numpy()
        one_hot = np.eye(num_classes)[test_labels_np]
        auc_val = roc_auc_score(one_hot, probs, average='macro', multi_class='ovr')
        print(f"Multiclass AUC (macro, OVR): {auc_val:.4f}")
    except Exception as e:
        print("AUC could not be computed:", e)

    plot_loss_curves(train_losses, val_losses, model_name, "train_val")


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories for your 4-class PT files
    train_dir = "/media/thibault/DynEarthNet/full_data/datasets/train"
    val_dir   = "/media/thibault/DynEarthNet/full_data/datasets/val"
    test_dir  = "/media/thibault/DynEarthNet/full_data/datasets/test"

    T = 28  # => 4*T=112 channels
    NUM_CLASSES = 4

    # Build train, val, test
    train_dataset = UNetCropDataset(split_dir=train_dir, pattern="pixel_dataset_*.pt",
                                    max_T=T, augment=False)
    val_dataset   = UNetCropDataset(split_dir=val_dir,   pattern="pixel_dataset_*.pt",
                                    max_T=T, augment=False)
    test_dataset  = UNetCropDataset(split_dir=test_dir,  pattern="pixel_dataset_*.pt",
                                    max_T=T, augment=False)

    # Subsampling
    #from torch.utils.data import Subset
    # e.g. random subset for debugging:
    # rng = torch.Generator().manual_seed(42)
    # N_TRAIN = len(train_dataset)//4
    # train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset), generator=rng)[:N_TRAIN])
    # etc.

    # Build DataLoaders
    train_loader = get_dataloader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader   = get_dataloader(val_dataset,   batch_size=4, shuffle=False, num_workers=2)
    test_loader  = get_dataloader(test_dataset,  batch_size=4, shuffle=False, num_workers=2)

    # Heavy UNET
    model_name_full="unet_full_4class"
    unet_full_model=UNet(in_channels=4*T,num_classes=NUM_CLASSES,dropout_p=0.1)

    # Light UNET
    model_name_lite = "unet_lite_4class"
    unet_lite_model = UNetLite(in_channels=4*T, num_classes=NUM_CLASSES, dropout_p=0.1)

    train_model(
        model=unet_full_model,
        model_name=model_name_full,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=30,   
        lr=1e-4,
        num_classes=NUM_CLASSES,
        weight_decay=5e-3
    )
