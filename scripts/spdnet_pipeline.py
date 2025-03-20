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
from geoopt.optim import RiemannianAdam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import SPDNet layers
sys.path.append('/home/thibault/Documents/implementations/anotherspdnet')

from anotherspdnet.nn import BiMap, ReEig, LogEig, Vech
from anotherspdnet.batchnorm import BatchNormSPD

########################################
# SEEDING
########################################
def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed (int): Random seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

################################################
# SPDNet Model with BatchNorm after each BiMap
################################################
class SPDNet3BiReBN(nn.Module):
    """
    A Symmetric Positive Definite Network (SPDNet) with 3 BiMap-ReEig-BatchNormSPD layers,
    followed by LogEig and Vech layers for feature extraction and a final linear classifier.

    Args:
        input_dim (int): Dimension of the input SPD matrices (e.g., 4 * T where T is the number of time steps).
        num_classes (int): Number of target classes for classification (default is 7).
        epsilon (float): Small value added for numerical stability in ReEig (default is 1e-2).
    """
    def __init__(self, input_dim, num_classes=7, epsilon=1e-2):
        super(SPDNet3BiReBN, self).__init__()
        self.layer1 = nn.Sequential(
            BiMap(input_dim, 64, dtype=torch.float32),
            ReEig(eps=epsilon),
            BatchNormSPD(64, max_iter_mean=10)
        )
        self.layer2 = nn.Sequential(
            BiMap(64, 32, dtype=torch.float32),
            ReEig(eps=epsilon),
            BatchNormSPD(32, max_iter_mean=10)
        )
        self.layer3 = nn.Sequential(
            BiMap(32, 16, dtype=torch.float32),
            ReEig(eps=epsilon),
            BatchNormSPD(16, max_iter_mean=10)
        )
        self.logeig = LogEig()
        self.vec = Vech()
        self.fc = nn.Linear(16 * (16 + 1) // 2, num_classes, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass of the SPDNet model.

        Args:
            x (torch.Tensor): Input batch of SPD matrices with shape (batch_size, dim, dim).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.logeig(x)
        x = self.vec(x)
        x = self.fc(x)
        return x

########################################
# Dataset Class
########################################
class CovarianceBlockDataset(Dataset):
    """
    A PyTorch Dataset for loading precomputed covariance blocks and corresponding label blocks.

    Each item consists of:
    - A tensor of covariance matrices reshaped as (H*W, dim, dim) where dim=124(=4*T with T=31 and 4 being the nb of channels)
    - A tensor of labels reshaped as (H*W)

    Args:
        dataset_root (str): Root directory containing the SPDNet-processed dataset.
        planet_folder (str or None): Folder name of the planet dataset (e.g., 'planet.10N').
        block_pattern (str): Glob pattern for finding covariance block files. Defaults to 'cov_label_block_*.pt'.
        max_T (int): Maximum number of time steps used to compute the covariance matrices.
    """

    def __init__(self, dataset_root, planet_folder=None, block_pattern='cov_label_block_*.pt', max_T=31):
        if planet_folder is not None:
            root_dir = os.path.join(dataset_root, 'spdnet', planet_folder)
        else:
            root_dir = os.path.join(dataset_root, 'spdnet')

        self.block_files = sorted(glob.glob(os.path.join(root_dir, '**', block_pattern), recursive=True))
        self.max_dim = 4 * max_T

        if len(self.block_files) == 0:
            raise ValueError(f"No files found in {root_dir} with pattern {block_pattern}")

        print(f"Loaded {len(self.block_files)} covariance blocks from {root_dir}")

    def __len__(self):
        return len(self.block_files)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - cov_block (torch.Tensor): Covariance matrices (H*W, dim, dim)
                - label_block (torch.Tensor): Labels (H*W)
        """
        block_file = self.block_files[idx]
        block = torch.load(block_file, map_location='cpu')

        cov_block = block['covariance']
        label_block = block['labels']

        H, W, dim, _ = cov_block.shape

        if dim < self.max_dim:
            pad_diff = self.max_dim - dim
            cov_block = F.pad(cov_block, (0, pad_diff, 0, pad_diff), value=0)

        cov_block = cov_block.reshape(-1, self.max_dim, self.max_dim)
        label_block = label_block.reshape(-1)

        return cov_block.float(), label_block.long()

########################################
# Dataloaders
########################################
def get_dataloaders(dataset, batch_size=1, val_split=0.1, test_split=0.1, num_workers=2):
    """
    Splits a dataset into training, validation, and test sets, and returns their respective dataloaders.

    Args:
        dataset (Dataset): PyTorch dataset to be split.
        batch_size (int): Batch size for the data loaders.
        val_split (float): Fraction of the dataset to use for validation.
        test_split (float): Fraction of the dataset to use for testing.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))

    return train_loader, val_loader, test_loader

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

    save_path = os.path.join(save_dir, f"confusion_matrix_normalized_{model_name}_{planet_folder}.png")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Normalize rows to percentages
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    # Avoid division by zero (when no samples exist for a class)
    row_sums[row_sums == 0] = 1
    cm_normalized = (cm_normalized / row_sums) * 100  # Convert to percentage

    # Plot using seaborn heatmap
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

########################################
# Training and Evaluation
########################################
def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one training epoch over the dataset.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (geoopt.optim): Riemannian Optimizer used for parameter updates.
        device (torch.device): Device to run the training on ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Average training loss and accuracy over the epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    print("\n---> Starting train_epoch loop...")

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

        if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            print(f"  Batch {batch_idx+1}/{len(loader)} processed, loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_epoch(model, loader, criterion, device, num_classes=7):
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader for the validation/test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on ('cpu' or 'cuda').
        num_classes (int): Number of classes.

    Returns:
        Tuple:
            avg_loss (float): Average loss over the dataset.
            acc (float): Accuracy of predictions.
            ious (np.ndarray): IoU per class.
            all_preds (torch.Tensor): Concatenated predicted labels.
            all_labels (torch.Tensor): Concatenated ground truth labels.
    """
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

    return avg_loss, acc, ious, all_preds, all_labels

def train_model(model, model_name, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-3, num_classes=7, planet_folder="planet.13N"):
    """
    Trains the SPDNet model and evaluates it on the test set after training.

    Args:
        model (nn.Module): The SPDNet model to be trained.
        model_name (str): Name of the model (used for saving checkpoints and plots).
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        num_classes (int): Number of classes for classification.
        planet_folder (str): Name of the planet folder (used in saved filenames).

    Saves:
        - Best model checkpoint to /home/thibault/ProcessedDynamicEarthNet/checkpoints.
        - Confusion matrix and loss curves to /home/thibault/ProcessedDynamicEarthNet/figures.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = RiemannianAdam(model.parameters(), lr=lr)

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

########################################
# Main Pipeline
########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = "/media/thibault/DynEarthNet/datasets"
    planet_folder = "planet.13N"

    T = 31
    input_dim = 4 * T

    spdnet_dataset = CovarianceBlockDataset(dataset_root, planet_folder=planet_folder)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=spdnet_dataset,
        batch_size=4,
        num_workers=2,
        val_split=0.1,
        test_split=0.1
    )

    print(f"\n--- Training SPDNet with BatchNorm on {planet_folder} ---")
    spdnet_bn_model = SPDNet3BiReBN(input_dim=input_dim)
    train_model(spdnet_bn_model, "spdnet_bn", train_loader, val_loader, test_loader, device, epochs=1, lr=1e-3, planet_folder=planet_folder)
