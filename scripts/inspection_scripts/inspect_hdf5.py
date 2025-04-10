import os
import h5py
import numpy as np
from tabulate import tabulate

def summarize_h5_file(filepath):
    """
    Summarizes a single HDF5 file.

    Returns:
        dict with file stats or None if corrupted
    """
    try:
        with h5py.File(filepath, 'r') as f:
            data_shape = f['data'].shape
            label_shape = f['labels'].shape
            label_vals = np.unique(f['labels'][:])
        return {
            "file": filepath,
            "data": str(data_shape),
            "labels": str(label_shape),
            "label_vals": ", ".join(map(str, label_vals))
        }
    except Exception as e:
        return {
            "file": filepath,
            "data": "ERROR",
            "labels": "ERROR",
            "label_vals": str(e)
        }

def scan_and_summarize_h5(root_dir="data"):
    """
    Recursively scans HDF5 files and prints summary table.
    """
    summaries = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".h5"):
                path = os.path.join(root, fname)
                summaries.append(summarize_h5_file(path))

    print(tabulate(summaries, headers="keys", tablefmt="grid"))

# Example usage
scan_and_summarize_h5("/media/thibault/DynEarthNet/datasets/train")  
