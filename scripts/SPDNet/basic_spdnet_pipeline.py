import os, glob
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json 
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from .spd_datasets import InferenceSlidingCovDataset
sys.path.append('/home/thibault/ProcessedDynamicEarthNet')
# External libraries for SPDNet
from geoopt.optim import RiemannianAdam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# SPDNet code
from anotherspdnet.nn import BiMap, ReEig, LogEig, Vech
from anotherspdnet.batchnorm import BatchNormSPD

# -------------------------------------
# GLOBAL CONFIG
# -------------------------------------
NUM_CLASSES = 4
SEED = 42

def set_seed(seed):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------------------------------------
# Losses / Class Weights
# -------------------------------------
def compute_class_weights(loader, num_classes=4):
    """
    Computes log-frequency-based class weights for a batch loader.
    Returns a tensor of weights suitable for use in CrossEntropyLoss or FocalLoss.
    """
    class_counts = torch.zeros(num_classes)
    for _, labels in loader:
        labels = labels.view(-1)
        counts = torch.bincount(labels, minlength=num_classes)
        class_counts += counts

    # Compute weights with protection against log(0)
    with torch.no_grad():
        class_weights = 1.0 / torch.log1p(class_counts)
        class_weights[torch.isinf(class_weights)] = 0.0

    if (class_counts == 0).any():
        missing = torch.where(class_counts == 0)[0].tolist()
        print(f"[WARNING] No pixel occurrences for classes: {missing}. Setting their weights to 0.")

    return class_weights

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    gamma = focusing parameter
    alpha can be a 1D tensor of per-class weights (similar to class_weights).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # shape [num_classes] or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C], raw logits
        targets: [N], integer class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)   # [N, C]
        probs = torch.exp(log_probs)               # [N, C]

        focal_weight = (1.0 - probs) ** self.gamma # [N, C]
        # Gather for the target class
        log_probs_target = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1) 
        focal_weight_target = focal_weight.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_target = self.alpha[targets]     # [N]
            focal_loss = -alpha_target * focal_weight_target * log_probs_target
        else:
            focal_loss = -focal_weight_target * log_probs_target

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -------------------------------------
# SPDNet Model(s)
# -------------------------------------
class SPDNet3BiRe(nn.Module):
    """
    A 3-block SPDNet model using:
      BiMap -> ReEig -> (optional) BN -> repeated 3 times,
      then LogEig -> Vech -> FC for classification.

    Input shape: [N, D, D] SPD matrices
    """
    def __init__(self, input_dim, num_classes=4, epsilon=1e-3, use_batch_norm=False, p=0.5):
        super(SPDNet3BiRe, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.bimap1 = BiMap(input_dim, 64, dtype=torch.float32)
        self.reig1 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn1 = BatchNormSPD(64, max_iter_mean=20)

        self.bimap2 = BiMap(64, 32, dtype=torch.float32)
        self.reig2 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn2 = BatchNormSPD(32, max_iter_mean=20)

        self.bimap3 = BiMap(32, 16, dtype=torch.float32)
        self.reig3 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn3 = BatchNormSPD(16, max_iter_mean=20)

        self.logeig = LogEig()
        self.vec = Vech()
        self.dropout = nn.Dropout(p=p)
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
        x = self.dropout(x)
        x = self.fc(x)
        return x

class SPDNet2BiRe(nn.Module):
    """
    A simpler 2-block SPDNet. Same concept, fewer layers.
    """
    def __init__(self, input_dim, num_classes=4, epsilon=1e-3, use_batch_norm=False, p=0.5):
        super(SPDNet2BiRe, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.bimap1 = BiMap(input_dim, 32, dtype=torch.float32)
        self.reig1 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn1 = BatchNormSPD(32, max_iter_mean=20)

        self.bimap2 = BiMap(32, 16, dtype=torch.float32)
        self.reig2 = ReEig(eps=epsilon)
        if self.use_batch_norm:
            self.bn2 = BatchNormSPD(16, max_iter_mean=20)

        self.logeig = LogEig()
        self.vec = Vech()
        self.dropout = nn.Dropout(p=p)
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

        x = self.logeig(x)
        x = self.vec(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# -------------------------------------
# Training / Evaluation Helpers
# -------------------------------------
def calculate_iou(preds, labels, num_classes=4):
    """
    Compute per-class IoU.
    """
    ious = []
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for cls in range(num_classes):
        intersection = np.logical_and(preds_np == cls, labels_np == cls).sum()
        union = np.logical_or(preds_np == cls, labels_np == cls).sum()
        ious.append(intersection / union if union else float('nan'))
    return np.array(ious)

def analyze_class_distribution(labels, label_type="Validation"):
    counts = torch.bincount(labels.flatten().cpu())
    print(f"\n{label_type} Label Distribution: ", counts.tolist())

def display_confusion_matrix(y_true, y_pred, run_name, model_name, num_classes=4):
    """
    Saves a row-normalized confusion matrix (percentage) to disk.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/confusion_matrices"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}_{run_name}.png")

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
    plt.title(f"Row-normalized Confusion Matrix (%)\n{model_name} [{run_name}]")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Normalized confusion matrix saved to {save_path}")

def plot_loss_curves(train_losses, val_losses, model_name, run_name):
    import matplotlib.pyplot as plt

    save_dir = "/home/thibault/ProcessedDynamicEarthNet/figures/loss_curves"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"loss_curve_{model_name}_{run_name}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves ({model_name}) [{run_name}]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for cov_batch, label_batch in loader:
        # cov_batch: (batch_size, D, D)
        # label_batch: (batch_size, )
        cov_batch = cov_batch.to(device)
        label_batch = label_batch.to(device)

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

def eval_epoch(model, loader, criterion, device, num_classes=4):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for cov_batch, label_batch in loader:
            cov_batch = cov_batch.to(device)
            label_batch = label_batch.to(device)

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
    lr=7e-3,
    weight_decay=1e-4,
    num_classes=4,
    run_name="default"
):
    model.to(device)

    # Compute class weights
    class_weights = compute_class_weights(train_loader, num_classes)
    class_weights = class_weights.to(device)
    print("Computed class weights:", class_weights)

    # Example: use Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    # or use CrossEntropyLoss: criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    best_val_iou = 0.0
    checkpoint_dir = "/home/thibault/ProcessedDynamicEarthNet/checkpoints/spdnet"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_ious, _, _ = eval_epoch(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mean_val_iou = float(np.nanmean(val_ious))

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {mean_val_iou:.4f}"
        )

        scheduler.step(val_loss)

        # Track best model by val IoU
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            save_path = os.path.join(checkpoint_dir, f"{model_name}_{run_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

    # Final test evaluation
    print(f"\n--- Final Test Evaluation ({model_name}) ---")
    test_loss, test_acc, test_ious, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device, num_classes
    )
    print(f"[{model_name}] Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, mIoU: {np.nanmean(test_ious):.4f}")

    # Confusion matrix + classification report
    display_confusion_matrix(
        test_labels.numpy(),
        test_preds.numpy(),
        run_name,
        model_name,
        num_classes=num_classes
    )
    print(classification_report(test_labels.numpy(), test_preds.numpy(), zero_division=0))

    # Save metrics
    report_dict = classification_report(test_labels.numpy(), test_preds.numpy(), zero_division=0, output_dict=True)

    metrics_dict = {
        "test_accuracy": test_acc,
        "test_mIoU": float(np.nanmean(test_ious)),
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
        "f1_macro": report_dict["macro avg"]["f1-score"]
    }

    save_metrics_path = f"/home/thibault/ProcessedDynamicEarthNet/eval_metrics/test_metrics_{model_name}.json"
    with open(save_metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Test metrics saved to {save_metrics_path}")

    # (Optional) Multi-class AUC
    try:
        from sklearn.metrics import roc_auc_score
        probs = []
        model.eval()
        with torch.no_grad():
            for cov_batch, _ in test_loader:
                cov_batch = cov_batch.to(device)
                out = model(cov_batch)
                probs.append(F.softmax(out, dim=1).cpu())
        probs = torch.cat(probs, dim=0).numpy()
        auc_val = roc_auc_score(
            np.eye(num_classes)[test_labels.numpy()],
            probs,
            average='macro',
            multi_class='ovr'
        )
        print(f"Multiclass AUC (macro, OVR): {auc_val:.4f}")
    except Exception as e:
        print("AUC could not be computed:", e)

    plot_loss_curves(train_losses, val_losses, model_name, run_name)


# -------------------------------------
# Main
# -------------------------------------
if __name__ == "__main__":

    # Paths to your HDF5 dataset folders
    TRAIN_DIR = "/media/thibault/DynEarthNet/datasets/train"
    VAL_DIR   = "/media/thibault/DynEarthNet/datasets/val"
    TEST_DIR  = "/media/thibault/DynEarthNet/datasets/test"

    # Create datasets
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "**", "*.h5"), recursive=True))
    val_files   = sorted(glob.glob(os.path.join(VAL_DIR, "**", "*.h5"), recursive=True))
    test_files  = sorted(glob.glob(os.path.join(TEST_DIR, "**", "*.h5"), recursive=True))

    train_ds = InferenceSlidingCovDataset(train_files, w_size=19, stride=9)
    val_ds   = InferenceSlidingCovDataset(val_files,   w_size=19, stride=9)
    test_ds  = InferenceSlidingCovDataset(test_files,  w_size=19, stride=9)
    print(f'\n Length of train dataset: {len(train_ds)}')
    print(f'\n Length of val dataset: {len(val_ds)}')
    print(f'Length of test dataset: {len(test_ds)}')


    # We can derive SPD input dimension from the dataset's n_times & n_features
    # i.e. each covariance is [C, C], where C = n_times * n_features
    input_dim = train_ds.n_times * train_ds.n_features
    print(f"[INFO] SPDNet input dimension = {input_dim} (n_times={train_ds.n_times} * n_features={train_ds.n_features})")

    # Dataloaders
    BATCH_SIZE = 32

    #---------------------Subsampling--------------------

    from torch.utils.data import Subset

    # Subsampling strategy (use fixed number of samples)
    def subsample(dataset, max_samples=50000, seed=42):
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)
        return Subset(dataset, indices)

    train_ds = subsample(train_ds, max_samples=6600)
    val_ds   = subsample(val_ds,   max_samples=1200)
    test_ds  = subsample(test_ds,  max_samples=1200)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    #----------------------------------------------------




    # Build SPDNet model
    USE_SPDNET_3 = True
    if USE_SPDNET_3:
        model = SPDNet3BiRe(input_dim=input_dim, num_classes=NUM_CLASSES, epsilon=1e-3, use_batch_norm=True)
        model_name = "SPDNet3BiReBN"
    else:
        model = SPDNet2BiRe(input_dim=input_dim, num_classes=NUM_CLASSES, epsilon=1e-3, use_batch_norm=False)
        model_name = "SPDNet2BiRe"

    # Training config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    LR = 7e-3
    WD = 1e-3

    print(f"\n[INFO] Training {model_name} for {EPOCHS} epochs on device: {DEVICE}")

    train_model(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WD,
        num_classes=NUM_CLASSES,
        run_name="hdf5"
    )
