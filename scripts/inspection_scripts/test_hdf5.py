import h5py
import glob
import os

train_dir = "/media/thibault/DynEarthNet/datasets/train"
h5_files = sorted(glob.glob(os.path.join(train_dir, "**", "*.h5"), recursive=True))

for path in h5_files:
    try:
        with h5py.File(path, "r") as f:
            _ = f.keys()
    except Exception as e:
        print(f"[ERROR] Cannot open {path}: {e}")
