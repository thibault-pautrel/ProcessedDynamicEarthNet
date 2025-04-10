import os
import glob
import numpy as np
import torch
import h5py
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from .unet_pipeline import UNetLite, train_model

class SlidingPatchH5Dataset(Dataset):
    def __init__(self, h5_files, max_T=28, w_size=19, stride=9, max_samples=15000):
        self.max_T = max_T
        self.w_size = w_size
        self.stride = stride
        self.crop = w_size // 2
        self.files = sorted(h5_files)
        self.sample_index = []  # (file_idx, r, c)
        self.file_cache = {}    # For reusing open h5py.File objects

        np.random.seed(42)
        sample_pool = []

        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                _, H, W, _ = f["data"].shape
            for r in range(self.crop, H - self.crop, stride):
                for c in range(self.crop, W - self.crop, stride):
                    sample_pool.append((file_idx, r, c))

        indices = np.random.choice(len(sample_pool), size=min(max_samples, len(sample_pool)), replace=False)
        self.sample_index = [sample_pool[i] for i in indices]
        print(f"[INFO] Indexed {len(self.sample_index)} patches (lazy load enabled)")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_idx, r, c = self.sample_index[idx]
        path = self.files[file_idx]

        # Open file only once and reuse
        if path not in self.file_cache:
            self.file_cache[path] = h5py.File(path, "r")
        f = self.file_cache[path]

        # Load 19x19x112 patch
        data_patch = f["data"][:self.max_T, r - self.crop:r + self.crop + 1, c - self.crop:c + self.crop + 1, :]
        label = f["labels"][r, c]

        patch = data_patch.transpose(0, 3, 1, 2).reshape(self.max_T * 4, self.w_size, self.w_size)
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 28
    NUM_CLASSES = 4
    w_size = 19

    def list_h5_files(root):
        return sorted(glob.glob(os.path.join(root, "**/*.h5"), recursive=True))

    train_files = list_h5_files("/media/thibault/DynEarthNet/datasets/train")
    val_files   = list_h5_files("/media/thibault/DynEarthNet/datasets/val")
    test_files  = list_h5_files("/media/thibault/DynEarthNet/datasets/test")

    start = time.time()
    train_ds = SlidingPatchH5Dataset(train_files, max_T=T, w_size=w_size, stride=9, max_samples=6600)
    print(f"[TIMER] Train dataset loaded in {time.time() - start:.2f} sec")

    val_ds   = SlidingPatchH5Dataset(val_files,   max_T=T, w_size=w_size, stride=9, max_samples=1200)
    test_ds  = SlidingPatchH5Dataset(test_files,  max_T=T, w_size=w_size, stride=9, max_samples=1200)

    BATCH_SIZE = 16
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    for i, (x, y) in enumerate(train_loader):
        print(f"[TIMER] First training batch loaded at {time.time() - start:.2f} sec")
        break

    input_channels = T * 4
    model = UNetLite(in_channels=input_channels, num_classes=NUM_CLASSES, dropout_p=0.5).to(device)

    train_model(
        model=model,
        model_name="unet_from_h5_patchwise",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=10,
        lr=1e-4,
        num_classes=NUM_CLASSES,
        weight_decay=1e-3
    )
