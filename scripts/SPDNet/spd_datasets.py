import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.covariance import LedoitWolf

class InferenceSlidingCovDataset(Dataset):
    """
    Slides over .h5 files and extracts 4*28-day local covariance matrices using sliding windows.
    Returns a tuple (cov_tensor, label), where:
        - cov_tensor: [D, D] SPD matrix (D=112)
        - label: center pixel class
    """

    def __init__(self, h5_paths, w_size=17, stride=7):
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        self.h5_paths = sorted(h5_paths)

        self.w_size = w_size
        self.stride = stride

        # Use first file for shape
        with h5py.File(self.h5_paths[0], 'r') as f:
            _, self.H, self.W, _ = f['data'].shape

        self.file_info = []
        cumulative_start = 0
        for path in self.h5_paths:
            with h5py.File(path, 'r') as f:
                _, h, w, _ = f["data"].shape
            n_win_rows = (h - w_size) // stride + 1
            n_win_cols = (w - w_size) // stride + 1
            n_windows = n_win_rows * n_win_cols
            self.file_info.append((path, n_win_rows, n_win_cols, cumulative_start))
            cumulative_start += n_windows

        self.total_windows = cumulative_start

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        for (path, n_win_rows, n_win_cols, cum_start) in reversed(self.file_info):
            if idx >= cum_start:
                chosen_path = path
                local_idx = idx - cum_start
                break

        i_win_row = local_idx // n_win_cols
        i_win_col = local_idx %  n_win_cols

        r_start = i_win_row * self.stride
        c_start = i_win_col * self.stride
        r_end = r_start + self.w_size
        c_end = c_start + self.w_size
        center_row = r_start + (self.w_size // 2)
        center_col = c_start + (self.w_size // 2)

        with h5py.File(chosen_path, 'r') as f:
            # Always take first 28 days only (shape: [28, w, w, C])
            window_np = f['data'][:28, r_start:r_end, c_start:c_end, :]
            label = f['labels'][center_row, center_col]

        # Crop shape: (28, w, w, 4) → (w, w, 28, 4)
        window_np = np.transpose(window_np, (1, 2, 0, 3))
        lw_input = window_np.reshape(-1, 28 * window_np.shape[-1])

        cov = LedoitWolf().fit(lw_input).covariance_
        cov_tensor = torch.from_numpy(cov).float()

        return cov_tensor, int(label)

    @property
    def n_times(self):
        return 28

    @property
    def n_features(self):
        with h5py.File(self.h5_paths[0], 'r') as f:
            return f['data'].shape[-1]
