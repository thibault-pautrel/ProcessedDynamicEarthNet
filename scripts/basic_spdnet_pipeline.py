import os
import glob
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score

# Adjust the path to your local modules, if needed:
sys.path.append("/home/thibault/ProcessedDynamicEarthNet")

from geoopt.optim import RiemannianAdam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from anotherspdnet.nn import BiMap, ReEig, LogEig, Vech
from anotherspdnet.batchnorm import BatchNormSPD

########################################
# SEEDING
########################################
def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch (CPU and CUDA).
    """
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
# SPDNet Model
########################################
class SPDNet3BiRe(nn.Module):
    """
    A 3-block SPDNet model using:
        - BiMap → ReEig → (optional) BatchNormSPD (3x)
        - Then LogEig → Vech → Dropout → Fully Connected

    Input:
        - SPD matrix of shape [N, D, D]
        - Output logits: [N, num_classes]

    Args:
        input_dim (int): Dimension D of input SPD matrices.
        num_classes (int): Output class count.
        epsilon (float): Epsilon for eigenvalue regularization.
        use_batch_norm (bool): Whether to use BatchNormSPD layers.
        p (float): Dropout probability.
    """
    def __init__(self, input_dim, num_classes=7, epsilon=1e-3, use_batch_norm=True, p=0.5):
        super(SPDNet3BiRe, self).__init__()
        self.use_batch_norm = use_batch_norm

        # 1) BiMap -> ReEig -> BN
        self.bimap1 = BiMap(input_dim, 64, dtype=torch.float32)
        self.reig1 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn1 = BatchNormSPD(64, max_iter_mean=20)

        # 2) BiMap -> ReEig -> BN
        self.bimap2 = BiMap(64, 32, dtype=torch.float32)
        self.reig2 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn2 = BatchNormSPD(32, max_iter_mean=20)

        # 3) BiMap -> ReEig -> BN
        self.bimap3 = BiMap(32, 16, dtype=torch.float32)
        self.reig3 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn3 = BatchNormSPD(16, max_iter_mean=20)

        # Final: LogEig -> Vech -> linear classifier
        self.logeig = LogEig()
        self.vec = Vech()
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(16 * (16 + 1) // 2, num_classes, dtype=torch.float32)

    def forward(self, x):

        def fix_if_needed(mat, eps=1e-5):
            mat = 0.5 * (mat + mat.transpose(-1, -2))  # force symmetry
            diag = mat.diagonal(dim1=-2, dim2=-1)
            diag = torch.clamp(diag, min=eps)
            mat = mat.clone()
            mat.diagonal(dim1=-2, dim2=-1).copy_(diag)
            I = torch.eye(mat.size(-1), device=mat.device).unsqueeze(0).expand(mat.size(0), -1, -1)
            mat = mat + eps * I
            return mat

        x = self.bimap1(x)
        x = self.reig1(x)
        x = fix_if_needed(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = fix_if_needed(x)

        x = self.bimap2(x)
        x = self.reig2(x)
        x = fix_if_needed(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = fix_if_needed(x)

        x = self.bimap3(x)
        x = self.reig3(x)
        x = fix_if_needed(x)
        if self.use_batch_norm:
            x = self.bn3(x)

        x = self.logeig(x)
        x = self.vec(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


########################################
# Helper
########################################
def analyze_class_distribution(labels, label_type="Validation"):
    counts = torch.bincount(labels.flatten().cpu())
    print(f"\n{label_type} Label Distribution: ", counts.tolist())

########################################
# MonthlyCovarianceDataset
########################################
class MonthlyCovarianceDataset(Dataset):
    """
    Loads all .pt files (named like cov_label_*.pt) from tile directories in a given split directory.
    The directory structure is like:
       <split_dir>/
         <tile_id>/
            2018-01/ cov_label_2018-01.pt
            2018-02/ cov_label_2018-02.pt
            ...
            2019-12/ cov_label_2019-12.pt

    Since there is no longer planet.* subfolders, the dataset scans the tile IDs directly, then each tile's monthly folders.

    For each found .pt file:
      - 'covariance': [H, W, D, D]  -> flattened to [H*W, D, D]
      - 'labels': [H, W]           -> flattened to [H*W]

    If the dimension D > (4*max_T), we slice down. If it's smaller, error out.

    Args:
        split_dir (str): Path to train/, val/, or test/ (containing tile folders).
        block_pattern (str): Glob pattern for .pt files (e.g. 'cov_label_*.pt').
        max_T (int): Max # of daily channels in time dimension. Typically 28 for 4*28=112.
    """

    def __init__(
        self,
        split_dir,
        block_pattern="cov_label_*.pt",
        max_T=28
    ):
        super().__init__()
        self.split_dir = split_dir
        self.block_pattern = block_pattern
        self.max_dim = 4 * max_T

        # Gather tile directories
        tile_dirs = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]

        all_files = []
        for tile_id in tile_dirs:
            tile_path = os.path.join(split_dir, tile_id)
            # e.g. /media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly/train/1311_3077_13
            monthly_files = sorted(
                glob.glob(os.path.join(tile_path, "**", block_pattern), recursive=True)
            )
            all_files.extend(monthly_files)

        if not all_files:
            raise ValueError(
                f"No .pt files found for pattern={block_pattern} in {split_dir}"
            )

        self.block_files = all_files
        print(f"Found {len(self.block_files)} total blocks in {split_dir}")

    def __len__(self):
        return len(self.block_files)

    def __getitem__(self, idx):
        block_file = self.block_files[idx]
        block = torch.load(block_file, map_location='cpu')  # {'covariance':..., 'labels':...}

        cov_block = block['covariance']  # shape (H, W, dim, dim)
        label_block = block['labels']    # shape (H, W)

        Hcov, Wcov, dim, _ = cov_block.shape
        Hlab, Wlab = label_block.shape
        new_H = min(Hcov, Hlab)
        new_W = min(Wcov, Wlab)
        cov_block = cov_block[:new_H, :new_W, :, :]
        label_block = label_block[:new_H, :new_W]

        if dim > self.max_dim:
            cov_block = cov_block[..., :self.max_dim, :self.max_dim]

        # Throw an error if it's smaller
        if cov_block.shape[2] < self.max_dim:
            raise ValueError(
                f"File has dimension < {self.max_dim} (shape={cov_block.shape}), not enough channels."
            )

        # Force symmetry
        cov_block = 0.5 * (cov_block + cov_block.transpose(-2, -1))
        cov_block = cov_block.reshape(new_H * new_W, cov_block.shape[2], cov_block.shape[3])
        label_block = label_block.reshape(new_H * new_W)

        return cov_block.float(), label_block.long()

########################################
# DataLoader
########################################
def get_loader(dataset, batch_size=1, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

########################################
# IoU and Visualization
########################################
def calculate_iou(preds, labels, num_classes=7):
    """Compute per-class IoU."""
    ious = []
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds_np == cls, labels_np == cls).sum()
        union = np.logical_or(preds_np == cls, labels_np == cls).sum()
        ious.append(intersection / union if union else float('nan'))
    return np.array(ious)

def display_confusion_matrix(y_true, y_pred, planet_folder, model_name, num_classes=7):
    """
    Saves a row-normalized confusion matrix (percentage) to disk.
    """
    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{planet_folder}.png")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = (cm_normalized / row_sums) * 100.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".1f", cmap="Blues",
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

########################################
# Train / Eval
########################################
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for cov_blocks, label_blocks in loader:
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

def eval_epoch(model, loader, criterion, device, num_classes=7):
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
    ious = calculate_iou(all_preds, all_labels, num_classes=num_classes)
    print(f"Validation IoU per class: {ious}")
    analyze_class_distribution(all_labels, label_type="Validation")
    analyze_class_distribution(all_preds, label_type="Predictions")
    return avg_loss, acc, ious, all_preds, all_labels

def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=7e-4,
    weight_decay=1e-4,
    num_classes=7,
    planet_folder="multi_split"
    ):
    model.to(device)
    # Compute class weights
    class_weights=compute_class_weights(train_loader,num_classes)
    class_weights = class_weights.to(device)
    print("Computed class weights:", class_weights)

    #Define weighted loss criterion
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_iou = 0.0
    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
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
            save_path = os.path.join(checkpoint_dir, f"{model_name}_{planet_folder}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device, num_classes
    )
    print(
        f"[{model_name}] Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
        f"mIoU: {np.nanmean(test_ious):.4f}"
    )

    # Confusion matrix + classification report
    display_confusion_matrix(
        test_labels.numpy(), test_preds.numpy(), planet_folder, model_name, num_classes=num_classes
    )
    print(classification_report(test_labels.numpy(), test_preds.numpy(), zero_division=0))
    # --- Save final test metrics to JSON ---
    import json

    # Convert classification_report into a dict
    report_dict = classification_report(
        test_labels.numpy(),
        test_preds.numpy(),
        zero_division=0,
        output_dict=True
    )

    # Extract relevant metrics
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

    save_metrics_path = f"/home/thibault/ProcessedDynamicEarthNet/test_metrics_{model_name}.json"
    with open(save_metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Test metrics saved to {save_metrics_path}")

    # Multi-class AUC attempt
    try:
        probs = []
        model.eval()
        with torch.no_grad():
            for cov_blocks, _ in test_loader:
                cb = cov_blocks.view(-1, cov_blocks.shape[-2], cov_blocks.shape[-1]).to(device)
                out = model(cb)
                probs.append(F.softmax(out, dim=1).cpu())
        probs = torch.cat(probs, dim=0).numpy()
        auc_val = roc_auc_score(
            np.eye(num_classes)[test_labels.numpy()], probs, average='macro', multi_class='ovr'
        )
        print(f"Multiclass AUC (macro, OVR): {auc_val:.4f}")
    except Exception as e:
        print("AUC could not be computed:", e)

    plot_loss_curves(train_losses, val_losses, model_name, planet_folder)

########################################
# Main
########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "/media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly/train"
    val_dir   = "/media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly/val"
    test_dir  = "/media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly/test"

    T = 28  # dimension => 4*T = 112
    input_dim = 4 * T

    # Build the three distinct datasets
    train_dataset = MonthlyCovarianceDataset(
        split_dir=train_dir,
        block_pattern='cov_label_*.pt',
        max_T=T
    )

    val_dataset = MonthlyCovarianceDataset(
        split_dir=val_dir,
        block_pattern='cov_label_*.pt',
        max_T=T
    )

    test_dataset = MonthlyCovarianceDataset(
        split_dir=test_dir,
        block_pattern='cov_label_*.pt',
        max_T=T
    )

    #==========================================
    #    Subsampling the datasets
    #===========================================

    #from torch.utils.data import Subset

    # Subsample sizes for quick sanity test
    #N_TRAIN = len(train_dataset)//8
    #N_VAL = len(val_dataset)//8
    #N_TEST = len(test_dataset)//8
    #print(f"Subsampling train/val/test datasets to {N_TRAIN}/{N_VAL}/{N_TEST} samples.")

    # Use fixed random seed for reproducibility
    #rng = torch.Generator().manual_seed(42)

    # Subsample indices
    #train_subset = Subset(train_dataset, torch.randperm(len(train_dataset), generator=rng)[:N_TRAIN])
    #val_subset   = Subset(val_dataset,   torch.randperm(len(val_dataset), generator=rng)[:N_VAL])
    #test_subset  = Subset(test_dataset,  torch.randperm(len(test_dataset), generator=rng)[:N_TEST])

    #train_loader = get_loader(train_subset, batch_size=1, shuffle=True,  num_workers=2)
    #val_loader   = get_loader(val_subset,   batch_size=1, shuffle=False, num_workers=2)
    #test_loader  = get_loader(test_subset,  batch_size=1, shuffle=False, num_workers=2)





    #==========================================
    # Full Datasets
    #==========================================
    train_loader = get_loader(train_dataset, batch_size=1, shuffle=True,  num_workers=2)
    val_loader   = get_loader(val_dataset,   batch_size=1, shuffle=False, num_workers=2)
    test_loader  = get_loader(test_dataset,  batch_size=1, shuffle=False, num_workers=2)

    print("\n--- Building SPDNet Model with separate train/val/test tile directories ---")
    model = SPDNet3BiRe(
        input_dim=input_dim,
        num_classes=7,
        epsilon=1e-3,
        use_batch_norm=False
    )

    train_model(
        model,
        model_name="spdnet_multi_weight_decay_small",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=20,       
        lr=1e-4,        
        planet_folder="multi_tiles"
    )
