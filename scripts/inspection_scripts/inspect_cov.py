import torch
import sys
from torch.utils.data import DataLoader
sys.path.append("/home/thibault/ProcessedDynamicEarthNet/scripts")
from spd_datasets import SlidingCovDataset

# Setup: adjust to one of your actual folders
h5_dir = "/media/thibault/DynEarthNet/datasets/train"  # or "val", "test"

# Instantiate dataset
dataset = SlidingCovDataset(h5_root_dir=h5_dir, w_size=17, overlap=False, max_files=1)

# Wrap in DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Grab one batch and print details
for covs, labels in loader:
    print("Covariance batch shape:", covs.shape)  # Expected: [B, C, C]
    print("Labels:", labels)                      # Expected: [B]
    print("Cov example (0):\n", covs[0])
    break
def check_spd(matrix: torch.Tensor, eps=1e-6):
    """
    Ensure matrix is symmetric and positive definite (SPD).
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals = torch.linalg.eigvalsh(sym)
    min_eig = eigvals.min().item()
    is_spd = (min_eig > eps)
    return is_spd, min_eig

def test_covariances_spd(h5_root_dir, w_size=17, max_files=1, max_batches=5):
    """
    Loads samples from SlidingCovDataset and checks for SPD properties.
    """
    dataset = SlidingCovDataset(h5_root_dir=h5_root_dir, w_size=w_size, max_files=max_files)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    failures = 0
    for i, (cov, label) in enumerate(loader):
        cov = cov.squeeze(0)  # shape [C, C]
        spd, min_eig = check_spd(cov)

        if not spd:
            print(f"[NOT SPD] Sample {i}, min_eig = {min_eig:.6f}")
            failures += 1
        else:
            print(f"[OK] Sample {i}, min_eig = {min_eig:.6f}")

        if i + 1 >= max_batches:
            break

    print(f"\n[SUMMARY] Checked {i+1} samples, {failures} failed SPD test.")

test_covariances_spd("/media/thibault/DynEarthNet/datasets/train", w_size=17, max_files=1, max_batches=50)

