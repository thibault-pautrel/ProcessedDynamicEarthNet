# Unified UNet and SPDNet Training Pipelines on Combined Dataset

import os
import sys
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from geoopt.optim import RiemannianAdam
# Import SPDNet layers
sys.path.append('/home/thibault/Documents/implementations/anotherspdnet')
from anotherspdnet.nn import BiMap, ReEig, LogEig, Vech
from anotherspdnet.batchnorm import BatchNormSPD

############################################
# General Setup
############################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################
# Early Stopping
############################################
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): Print messages when improvement happens.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Where to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_score_max = -np.inf

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation score improved! Model saved to {self.path}")






############################################
# UNet Dataset (Merged Planets)
############################################
class UNetPatchDataset(Dataset):
    def __init__(self, dataset_root, patch_size=128):
        self.dataset_files = sorted(
            glob.glob(
                os.path.join(dataset_root, 'unet', '**', 'pixel_dataset_*.pt'),
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

############################################
# SPDNet Dataset (Merged Planets)
############################################
class CovarianceBlockDataset(Dataset):
    def __init__(self, dataset_root, block_pattern='cov_label_block_*.pt', max_T=31):
        root_dir = os.path.join(dataset_root, 'spdnet')
        self.block_files = sorted(glob.glob(os.path.join(root_dir, '**', block_pattern), recursive=True))
        self.max_dim = 4 * max_T

        if len(self.block_files) == 0:
            raise ValueError(f"No files found in {root_dir} with pattern {block_pattern}")

    def __len__(self):
        return len(self.block_files)

    def __getitem__(self, idx):
        block_file = self.block_files[idx]
        block = torch.load(block_file, map_location='cpu')

        cov_block = block['covariance']
        label_block = block['labels']

        H, W, dim, _ = cov_block.shape

        if dim < self.max_dim:
            pad_diff = self.max_dim - dim
            cov_block = torch.nn.functional.pad(cov_block, (0, pad_diff, 0, pad_diff), value=0)

        cov_block = cov_block.reshape(-1, self.max_dim, self.max_dim)
        label_block = label_block.reshape(-1)

        return cov_block.float(), label_block.long()

############################################
# DataLoader Function
############################################
def get_dataloaders(dataset, batch_size=4, train_ratio=0.7, val_ratio=0.15, num_workers=4):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

############################################
# UNet Model
############################################
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

############################################
# SPDNet Model
############################################
class SPDNet3BiRe(nn.Module):
    def __init__(self, input_dim, num_classes=7, epsilon=1e-2, use_batch_norm=False):
        super(SPDNet3BiRe, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.bimap1 = BiMap(input_dim, 64, dtype=torch.float32)
        self.reig1 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn1 = BatchNormSPD(64, max_iter_mean=10)

        self.bimap2 = BiMap(64, 32, dtype=torch.float32)
        self.reig2 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn2 = BatchNormSPD(32, max_iter_mean=10)

        self.bimap3 = BiMap(32, 16, dtype=torch.float32)
        self.reig3 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn3 = BatchNormSPD(16, max_iter_mean=10)

        self.logeig = LogEig()
        self.vec = Vech()
        self.fc = nn.Linear(16 * (16 + 1) // 2, num_classes, dtype=torch.float32)

    def forward(self, x):
        x = self.bimap1(x)
        x = self.reig1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.bimap2(x)
        x = self.reig2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.bimap3(x)
        x = self.reig3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.logeig(x)
        x = self.vec(x)
        x = self.fc(x)
        return x


########################################
# Metrics: IOU
########################################
def calculate_iou(preds, labels, num_classes=7):
    """
    Computes Intersection over Union (IoU) for each class.

    Args:
        preds (torch.Tensor): Predicted class indices.
        labels (torch.Tensor): Ground truth class indices.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: IoU values for each class.
    """
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

#################################################
# Visualization: confusion matrix and loss curves
#################################################
def display_confusion_matrix(y_true, y_pred, planet_folder, model_name, num_classes=7):
    """
    Generates and saves a row-normalized confusion matrix plot for the model predictions.
    Instead of showing counts, it displays the row-wise percentage of pixels per true label.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        planet_folder (str): Name of the planet folder (used in the plot title and filename).
        model_name (str): Name of the model (used in the plot title and filename).
        num_classes (int): Number of classes (default is 7).

    Saves:
        PNG file of the confusion matrix in /home/thibault/ProcessedDynamicEarthNet/figures.
    """

    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{planet_folder}.png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Normalize rows to percentages
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    # Avoid division by zero (when no samples exist for a class)
    row_sums[row_sums == 0] = 1
    cm_normalized = (cm_normalized / row_sums) * 100  # Convert to percentage


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
    """
    Plots and saves the training and validation loss curves over epochs.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        model_name (str): Name of the model (used in the plot title and filename).
        planet_folder (str): Name of the planet folder (used in the plot title and filename).

    Saves:
        PNG file of the loss curves in /home/thibault/ProcessedDynamicEarthNet/figures.
    """
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

############################################
# UNet Training and Evaluation
############################################
def train_epoch_unet(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total += y.numel()
        correct += (preds == y).sum().item()
        running_loss += loss.item() * X.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = correct / total
    return avg_loss, acc

def eval_epoch_unet(model, loader, criterion, device, num_classes=7):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            preds = torch.argmax(outputs, dim=1)
            total += y.numel()
            correct += (preds == y).sum().item()
            running_loss += loss.item() * X.size(0)

            all_preds.append(preds.cpu().flatten())
            all_labels.append(y.cpu().flatten())

    avg_loss = running_loss / len(loader.dataset)
    acc = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    ious = calculate_iou(all_preds, all_labels, num_classes)
    return avg_loss, acc, ious, all_preds, all_labels

def train_model_unet(model, model_name, train_loader, val_loader, test_loader, device, epochs=50, lr=1e-4, num_classes=7, planet_folder="ALL_PLANETS", patience=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(checkpoint_dir, f"{model_name}_{planet_folder}_best.pt")
    )

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_unet(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch_unet(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{model_name}] Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val mIoU: {np.nanmean(val_ious):.4f}")

        # Call early stopping with validation mIoU
        early_stopping(np.nanmean(val_ious), model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final evaluation after early stopping or max epochs
    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch_unet(model, test_loader, criterion, device, num_classes)

    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | mIoU: {np.nanmean(test_ious):.4f}")
    display_confusion_matrix(test_labels.numpy(), test_preds.numpy(), planet_folder, model_name, num_classes)
    plot_loss_curves(train_losses, val_losses, model_name, planet_folder)

############################################
# SPDNet Training and Evaluation
############################################
def train_epoch_spdnet(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (cov_blocks, label_blocks) in enumerate(loader):
        cov_batch = cov_blocks.view(-1, cov_blocks.shape[-2], cov_blocks.shape[-1]).to(device)
        label_batch = label_blocks.view(-1).to(device)

        optimizer.zero_grad()
        outputs = model(cov_batch)

        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * label_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_epoch_spdnet(model, loader, criterion, device, num_classes=7):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for cov_blocks, label_blocks in loader:
            cov_batch = cov_blocks.view(-1, cov_blocks.shape[-2], cov_blocks.shape[-1]).to(device)
            label_batch = label_blocks.view(-1).to(device)

            outputs = model(cov_batch)
            loss = criterion(outputs, label_batch)

            running_loss += loss.item() * label_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(label_batch.cpu())

    avg_loss = running_loss / total
    acc = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    ious = calculate_iou(all_preds, all_labels, num_classes)
    return avg_loss, acc, ious, all_preds, all_labels

def train_model_spdnet(model, model_name, train_loader, val_loader, test_loader, device, epochs=50, lr=1e-3, num_classes=7, planet_folder="ALL_PLANETS", patience=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = RiemannianAdam(model.parameters(), lr=lr)

    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=os.path.join(checkpoint_dir, f"{model_name}_{planet_folder}_best.pt")
    )

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_spdnet(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch_spdnet(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{model_name}] Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val mIoU: {np.nanmean(val_ious):.4f}")

        # Call early stopping with validation mIoU
        early_stopping(np.nanmean(val_ious), model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Final evaluation after early stopping or max epochs
    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch_spdnet(model, test_loader, criterion, device, num_classes)

    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | mIoU: {np.nanmean(test_ious):.4f}")
    display_confusion_matrix(test_labels.numpy(), test_preds.numpy(), planet_folder, model_name, num_classes)
    plot_loss_curves(train_losses, val_losses, model_name, planet_folder)


############################################
# Main Execution
############################################
if __name__ == "__main__":

    unet_dataset_root = "/media/thibault/DynEarthNet/datasets_subsampled"
    spdnet_dataset_root = "/media/thibault/DynEarthNet/datasets"


    print("\n--- Loading UNet Dataset ---")
    unet_dataset = UNetPatchDataset(dataset_root=unet_dataset_root)
    unet_train, unet_val, unet_test = get_dataloaders(unet_dataset, batch_size=4, num_workers=4)

    print("\n--- Training UNet on ALL PLANETS ---")
    unet_model = UNet(in_channels=4 * 31, num_classes=7).to(device)
    train_model_unet(
        model=unet_model,
        model_name="unet",
        train_loader=unet_train,
        val_loader=unet_val,
        test_loader=unet_test,
        device=device,
        epochs=1,
        lr=1e-4,
        num_classes=7,
        planet_folder="ALL_PLANETS"
    )
   
    print("\n--- Loading SPDNet Dataset ---")
    spdnet_dataset = CovarianceBlockDataset(dataset_root=spdnet_dataset_root)
    spd_train, spd_val, spd_test = get_dataloaders(spdnet_dataset, batch_size=1, num_workers=2)

    print("\n--- Training SPDNet on ALL PLANETS ---")
    spdnet_model = SPDNet3BiRe(input_dim=4*31).to(device)
    train_model_spdnet(
        model=spdnet_model,
        model_name="spdnet",
        train_loader=spd_train,
        val_loader=spd_val,
        test_loader=spd_test,
        device=device,
        epochs=1,
        lr=1e-3,
        num_classes=7,
        planet_folder="ALL_PLANETS"
    )

