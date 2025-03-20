import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

############################################
# Set seed for reproducibility
############################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# ------------------------------
# Dataset Class for U-Net
# ------------------------------
class UNetPatchDataset(Dataset):
    def __init__(self, dataset_root, planet_folder='planet.13N', patch_size=256):
        self.dataset_files = sorted(
            glob.glob(
                os.path.join(dataset_root, 'unet', planet_folder, '**', 'pixel_dataset_*.pt'),
                recursive=True
            )
        )
        self.patch_size = patch_size
        self.fixed_T = 31
        self.fixed_channels = 4 * self.fixed_T

        self.patches = []
        for file_idx, file in enumerate(self.dataset_files):
            data = torch.load(file)
            _, H, W = data["features"].permute(2, 0, 1).shape
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    self.patches.append((file_idx, i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        file_idx, i, j = self.patches[idx]
        data = torch.load(self.dataset_files[file_idx])

        features = data["features"].permute(2, 0, 1).float()

        C, H, W = features.shape
        if C < self.fixed_channels:
            pad_channels = self.fixed_channels - C
            padding = torch.zeros(pad_channels, H, W, dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)

        labels = data["labels"].long()

        features_patch = features[:, i:i+self.patch_size, j:j+self.patch_size]
        labels_patch = labels[i:i+self.patch_size, j:j+self.patch_size]

        return features_patch, labels_patch

# ------------------------------
# U-Net Model 
# ------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
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
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        m = self.middle(self.pool(e4))
        d4 = self.up4(m)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)

# ------------------------------
# Metrics and Visualization
# ------------------------------
def calculate_iou(preds, labels, num_classes=7):
    ious = []
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or(preds == cls, labels == cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.array(ious)

def display_confusion_matrix(y_true, y_pred, planet_folder, model_name, num_classes=7):
    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"confusion_matrix_normalized_{model_name}_{planet_folder}.png")

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
    plt.title(f"Row-normalized Confusion Matrix (%)\n{model_name} [{planet_folder}]")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Normalized confusion matrix saved to {save_path}")

def plot_loss_curves(train_losses, val_losses, model_name, planet_folder):
    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/loss_curves"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"loss_curve_{model_name}_{planet_folder}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves ({model_name}) [{planet_folder}]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curves saved to {save_path}")

# ------------------------------
# Dataloaders, Training & Evaluation
# ------------------------------
def get_dataloaders(dataset, batch_size=16, train_ratio=0.7, val_ratio=0.15, num_workers=2):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_pixels = 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == y).sum().item()
        total_pixels += torch.numel(y)
        total_loss += loss.item() * y.size(0)

    return total_loss / len(dataloader.dataset), total_correct / total_pixels

def eval_epoch(model, dataloader, criterion, device, num_classes=7):
    model.eval()
    total_loss, total_correct, total_pixels = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y).sum().item()
            total_pixels += torch.numel(y)
            total_loss += loss.item() * y.size(0)

            all_preds.append(preds.cpu().flatten())
            all_labels.append(y.cpu().flatten())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    ious = calculate_iou(all_preds, all_labels, num_classes=num_classes)

    return total_loss / len(dataloader.dataset), total_correct / total_pixels, ious, all_preds, all_labels

def train_model(model, model_name, train_loader, val_loader, test_loader, device, planet_folder, epochs=50, lr=1e-4, num_classes=7):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0
    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{model_name}] Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val mIoU: {np.nanmean(val_ious):.4f}")

        if np.nanmean(val_ious) > best_val_iou:
            best_val_iou = np.nanmean(val_ious)
            save_path = os.path.join(checkpoint_dir, f"{model_name}_{planet_folder}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model ({model_name}) to {save_path}")

    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device, num_classes)

    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | mIoU: {np.nanmean(test_ious):.4f}")
    display_confusion_matrix(test_labels.numpy(), test_preds.numpy(), planet_folder, model_name, num_classes=num_classes)
    plot_loss_curves(train_losses, val_losses, model_name, planet_folder)

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = "/media/thibault/DynEarthNet/datasets"
    planet_folder = "planet.10N"
    model_name = "unet"

    print(f"\n--- U-Net Training on {planet_folder} ---")
    unet_dataset = UNetPatchDataset(dataset_root, planet_folder=planet_folder, patch_size=256)

    train_loader, val_loader, test_loader = get_dataloaders(unet_dataset, batch_size=4, num_workers=2)

    unet_model = UNet(in_channels=4*31, num_classes=7)

    train_model(
        unet_model,
        model_name,
        train_loader,
        val_loader,
        test_loader,
        device,
        planet_folder=planet_folder,
        epochs=1,
        lr=1e-3,
        num_classes=7
    )
