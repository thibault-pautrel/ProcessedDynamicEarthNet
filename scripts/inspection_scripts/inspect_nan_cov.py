import torch
import os

def check_spd_properties(matrix: torch.Tensor, eps=1e-6):
    """
    Check SPD-ness: symmetry and positive definiteness via eigenvalues.
    """
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    try:
        eigvals = torch.linalg.eigvalsh(matrix)
        min_eig = eigvals.min().item()
        if min_eig <= eps:
            return False, min_eig
        return True, min_eig
    except Exception as e:
        return False, str(e)


def inspect_covariance_blocks_with_spd(dataset_root, max_files=10):
    """
    Check for NaNs and SPD properties in .pt files.
    """
    checked = 0
    for root, _, files in os.walk(dataset_root):
        for fname in sorted(files):
            if fname.endswith(".pt"):
                path = os.path.join(root, fname)
                data = torch.load(path, map_location="cpu")
                cov = data.get("covariance", None)

                if cov is None or not isinstance(cov, torch.Tensor):
                    print(f"[SKIP] No valid 'covariance' tensor in {fname}")
                    continue

                if torch.isnan(cov).any():
                    print(f"[NaN] {fname}")
                    continue

                H, W, dim, _ = cov.shape
                failures = 0
                for i in range(min(5, H)):
                    for j in range(min(5, W)):
                        spd, min_eig = check_spd_properties(cov[i, j])
                        if not spd:
                            failures += 1
                            print(f"[NOT SPD] {fname} -> ({i},{j}) min_eig: {min_eig}")
                            break
                    if failures > 0:
                        break

                if failures == 0:
                    print(f"[SPD OK] {fname}")

                checked += 1
                if checked >= max_files:
                    print(f"\nChecked max {max_files} files. Stopping.")
                    return

# Use your actual path here
dataset_path = "/media/thibault/DynEarthNet/subsampled_data/datasets/spdnet_monthly/planet.10N"
inspect_covariance_blocks_with_spd(dataset_path, max_files=50)
