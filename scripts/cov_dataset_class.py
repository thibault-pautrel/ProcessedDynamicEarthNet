import torch
from torch.utils.data import Dataset
import glob, os
from sklearn.covariance import LedoitWolf
import numpy as np
class OnTheFlyLedoitCovDataset(Dataset):
    """
    Computes covariance matrices or just labels from pixel-level feature tensors using non-overlapping 11×11 windows.
    Uses Ledoit-Wolf shrinkage estimator for SPDness unless label_only=True.

    Each sample yields:
        - covariances: [8649, C, C] (optional)
        - labels: [8649]            (center-pixel label of each window)

    Args:
        split_dir (str): Directory with monthly tile subfolders containing 'pixel_dataset_*.pt'.
        max_T (int): Number of days to use (C = 4 × T bands). Default = 28 ➝ C = 112.
        label_only (bool): If True, skip covariance computation and return only relabeled outputs.
    """
    def __init__(self, split_dir, max_T=28, label_only=False):
        self.max_dim = 4 * max_T
        self.files = glob.glob(os.path.join(split_dir, "*", "*", "pixel_dataset_*.pt"))
        self.files.sort()
        self.label_only = label_only
        if not self.files:
            raise ValueError(f"No dataset files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        labels = data["labels"]

        H = W = 1024
        window = 11
        stride = 11
        n = (H - window) // stride + 1

        label_matrix = torch.zeros((n, n), dtype=torch.long)
        for i in range(n):
            for j in range(n):
                y, x = i * stride, j * stride
                label_matrix[i, j] = labels[y + window // 2, x + window // 2]

        if self.label_only:
            return None, label_matrix.view(-1)

        features = data["features"][:, :, :self.max_dim]
        C = self.max_dim
        cov_matrices = torch.zeros((n, n, C, C), dtype=torch.float32)
        lw = LedoitWolf()

        for i in range(n):
            for j in range(n):
                y, x = i * stride, j * stride
                patch = features[y:y+window, x:x+window, :]  # [11,11,C]
                pixels = patch.reshape(-1, C).numpy()        # [121, C]
                cov = lw.fit(pixels).covariance_
                cov_matrices[i, j] = torch.from_numpy(cov).float()

        return cov_matrices.view(-1, C, C), label_matrix.view(-1)


def is_spd(matrix: torch.Tensor, tol=1e-6):
    """
    Checks if a square matrix is Symmetric Positive Definite (SPD).

    Args:
        matrix (torch.Tensor): Tensor of shape [C, C] or [N, C, C].
        tol (float): Eigenvalue threshold for positive definiteness.

    Returns:
        bool or List[bool]: True if SPD, else False. List if batched.
    """
    if matrix.ndim == 2:
        # Single matrix
        sym = torch.allclose(matrix, matrix.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(matrix)
        return sym and torch.all(eigvals > tol)
    elif matrix.ndim == 3:
        results = []
        for mat in matrix:
            sym = torch.allclose(mat, mat.T, atol=1e-5)
            eigvals = torch.linalg.eigvalsh(mat)
            results.append(bool(sym and torch.all(eigvals > tol)))
        return results
    else:
        raise ValueError("Input must be a [C, C] or [N, C, C] tensor.")

if __name__ == "__main__":
    # Example path — update this to a specific split (e.g., 'full_data/datasets/unet')
    sample_dir = "/media/thibault/DynEarthNet/full_data/datasets/"

    # Instantiate dataset
    dataset = OnTheFlyLedoitCovDataset(split_dir=sample_dir, max_T=28)

    # Check dataset size
    print(f"Dataset contains {len(dataset)} tile-months.")

    # Load a single sample (should trigger Ledoit-Wolf covariance computation)
    print("[INFO] Loading a single sample...")
    cov, labels = dataset[0]

    # Print results
    print(f"[SUCCESS] Covariance shape: {cov.shape}")  # (8649, 112, 112)
    print(f"[SUCCESS] Labels shape: {labels.shape}")   # (8649,)
    print(f"[INFO] First few labels: {labels[:10].tolist()}")
    print(f"[INFO] Sample covariance matrix:\n{cov[0, :4, :4]}")

    # Run SPD check on a few matrices
    print("[CHECK] SPD status of first 5 matrices:")
    spd_flags = is_spd(cov[:5])  # List[bool]
    for i, flag in enumerate(spd_flags):
        print(f"  Cov[{i}] is SPD? {flag}")
