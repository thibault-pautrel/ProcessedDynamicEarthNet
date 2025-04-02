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

########################################
# Helper function to handle class imbalance
#########################################

def compute_class_weights(loader, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, labels in loader:
        labels = labels.view(-1)
        counts = torch.bincount(labels, minlength=num_classes)
        class_counts += counts
    class_weights = class_counts.sum() / (num_classes * class_counts)
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

        focal_weight = (1.0 - probs) ** self.gamma # shape [N, C]
        # Gather log_probs at target indices: shape [N, 1]
        log_probs_target = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight_target = focal_weight.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # If alpha is provided, multiply by alpha of the target class
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
# UNet Model with pad_and_cat fix + Dropout (ADDED) 
###############################################################################
class UNet(nn.Module):
    """
    A standard U-Net model for semantic segmentation with padded skip connections,
    plus dropout inserted into encoder/decoder blocks to help regularize.
    """

    def __init__(self, in_channels=112, num_classes=7, dropout_p=0.2):
        """
        dropout_p: probability for dropout in each encoder/decoder block
        """
        super(UNet, self).__init__()

        # We pass dropout_p to each conv_block
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

    def conv_block(self, in_channels, out_channels, dropout_p=0.2):
        """
        Two conv layers each followed by BN + ReLU, plus a dropout layer in between.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),  # ADDED dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)   # ADDED dropout
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
# Lighter UNet (ADDED)
###############################################################################
class UNetLite(nn.Module):
    """
    A 'lighter' U-Net variant with fewer feature maps to reduce capacity 
    (and thus help prevent overfitting on smaller datasets).
    """

    def __init__(self, in_channels=112, num_classes=7, dropout_p=0.2):
        """
        We use half the channels (32→64→128→256→512) instead of (64→128→256→512→1024).
        """
        super(UNetLite, self).__init__()

        # Encoders
        self.encoder1 = self.conv_block(in_channels, 32, dropout_p)
        self.encoder2 = self.conv_block(32, 64, dropout_p)
        self.encoder3 = self.conv_block(64, 128, dropout_p)
        self.encoder4 = self.conv_block(128, 256, dropout_p)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Middle
        self.middle = self.conv_block(256, 512, dropout_p)

        # Decoders
        self.up4 = self.up_conv(512, 256)
        self.dec4 = self.conv_block(512, 256, dropout_p)

        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(256, 128, dropout_p)

        self.up2 = self.up_conv(128, 64)
        self.dec2 = self.conv_block(128, 64, dropout_p)

        self.up1 = self.up_conv(64, 32)
        self.dec1 = self.conv_block(64, 32, dropout_p)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_p=0.2):
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
# UNet Dataset with data augmentation 
###############################################################################
class UNetCropDataset(Dataset):
    """
    Loads and processes monthly pixel datasets from .pt files into UNet-ready tensors
    from a single directory (and its subdirectories).

    We add random flips and random 90° rotations for data augmentation if 'augment=True'.
    """

    def __init__(
        self,
        split_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=28,
        augment=False  # ADDED: augmentation flag
    ):
        super().__init__()
        self.final_H = final_H
        self.final_W = final_W
        self.max_C = 4 * max_T  # 112 channels
        self.files = sorted(
            glob.glob(os.path.join(split_dir, "**", pattern), recursive=True)
        )
        self.augment = augment  # ADDED
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

        # -------------------------
        # ADDED DATA AUGMENTATION
        # -------------------------
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                feats = torch.flip(feats, dims=[2])  # Flip W dimension
                labs  = torch.flip(labs,  dims=[1])

            # Random vertical flip
            if random.random() < 0.5:
                feats = torch.flip(feats, dims=[1])  # Flip H dimension
                labs  = torch.flip(labs,  dims=[0])

            # Random 90° rotations (0, 90, 180, 270)
            k = random.randint(0, 3)  # 0..3
            if k > 0:
                feats = torch.rot90(feats, k, dims=[1, 2])  # rotate spatial dims
                labs  = torch.rot90(labs,  k, dims=[0, 1])

        return feats, labs

###############################################################################
# DataLoader Helpers
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
# Metrics (same as before)
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
    lr=1e-4,
    num_classes=7,
    weight_decay=1e-4
    ):
    model.to(device)
    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    print("Computed class weights:", class_weights)

    # Optionally use FocalLoss or standard CrossEntropy
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    #criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2, verbose=True
    )

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

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

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

    display_confusion_matrix(
        test_labels.numpy(), test_preds.numpy(),
        split_tag="test",
        model_name=model_name, 
        num_classes=num_classes
    )
    print(classification_report(
        test_labels.numpy(), test_preds.numpy(), zero_division=0
    ))

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
    recall_macro = report_dict["macro avg"]["recall"]
    f1_macro = report_dict["macro avg"]["f1-score"]

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
                sm = torch.softmax(out, dim=1)  # [B,7,H,W]
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

    # Directories holding the .pt files for each split
    train_dir = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/train"
    val_dir   = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/val"
    test_dir  = "/media/thibault/DynEarthNet/subsampled_data/datasets/unet/test"

    T = 28  # => channels = 4*T = 112

    # Build train, val, test datasets
    # (ADDED) set 'augment=True' for training set only:
    train_dataset = UNetCropDataset(
        split_dir=train_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T,
        augment=True    # random flips + random 90° rotation
    )

    val_dataset = UNetCropDataset(
        split_dir=val_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T,
        augment=False   # no augmentation on validation
    )

    test_dataset = UNetCropDataset(
        split_dir=test_dir,
        pattern="pixel_dataset_*.pt",
        final_H=118,
        final_W=118,
        max_T=T,
        augment=False   # no augmentation on test
    )

    #==========================================
    #    Subsampling the datasets
    #===========================================

    from torch.utils.data import Subset

    # Subsample sizes for quick sanity test
    N_TRAIN = len(train_dataset)//4
    N_VAL = len(val_dataset)//4
    N_TEST = len(test_dataset)//4
    print(f"Subsampling train/val/test datasets to {N_TRAIN}/{N_VAL}/{N_TEST} samples.")

    # Use fixed random seed for reproducibility
    rng = torch.Generator().manual_seed(42)

    model_name = "unet_weight_decay_small"

    # Subsample indices
    train_subset = Subset(train_dataset, torch.randperm(len(train_dataset), generator=rng)[:N_TRAIN])
    val_subset   = Subset(val_dataset,   torch.randperm(len(val_dataset), generator=rng)[:N_VAL])
    test_subset  = Subset(test_dataset,  torch.randperm(len(test_dataset), generator=rng)[:N_TEST])

    train_loader = get_dataloader(train_subset, batch_size=4, shuffle=True,  num_workers=2)
    val_loader   = get_dataloader(val_subset,   batch_size=4, shuffle=False, num_workers=2)
    test_loader  = get_dataloader(test_subset,  batch_size=4, shuffle=False, num_workers=2)

    #=======================================
    # Full datasets
    #=======================================


    # Create data loaders
    #train_loader = get_dataloader(train_dataset, batch_size=4, shuffle=True,  num_workers=2)
    #val_loader   = get_dataloader(val_dataset,   batch_size=4, shuffle=False, num_workers=2)
    #test_loader  = get_dataloader(test_dataset,  batch_size=4, shuffle=False, num_workers=2)


    # -------------------------------------------------------------------
    # Original U-Net with dropout 
    # -------------------------------------------------------------------
    #model_name = "unet_dropout_aug"
    #unet_model = UNet(in_channels=4*T, num_classes=7, dropout_p=0.2)

    #train_model(
    #    model=unet_model,
    #    model_name=model_name,
    #    train_loader=train_loader,
    #    val_loader=val_loader,
    #    test_loader=test_loader,
    #    device=device,
    #   epochs=20,
    #   lr=1e-4,
    #   num_classes=7,
    #    weight_decay=1e-4
    #)

    # -------------------------------------------------------------------
    # Lighter U-Net 
    # -------------------------------------------------------------------
    model_name_lite = "unet_lite_dropout_aug"
    unet_lite_model = UNetLite(in_channels=4*T, num_classes=7, dropout_p=0.8)

    train_model(
        model=unet_lite_model,
        model_name=model_name_lite,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=20,
        lr=1e-4,
        num_classes=7,
        weight_decay=5e-4
    )
