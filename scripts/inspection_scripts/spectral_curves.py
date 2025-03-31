import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#======================================================
# Functions for loading monthly labels & daily images
#======================================================
def get_month_label(root_dir, year, month):
    """
    Loads the monthly label: root_dir/YYYY-MM/labels/raster.pt
    shape = [7, H, W], each slice = {0, 255}.
    """
    month_str = f"{year}-{month:02d}"
    label_path = os.path.join(root_dir, month_str, "labels", "raster.pt")
    if not os.path.isfile(label_path):
        return None
    return torch.load(label_path)

def get_daily_image(root_dir, date_obj):
    """
    Loads a 4-channel daily image from:
      root_dir/YYYY-MM/data_YYYY-MM-DD.pt
    shape = [4, H, W], or None if missing.
    """
    y = str(date_obj.year)
    m = f"{date_obj.month:02d}"
    d = f"{date_obj.day:02d}"
    data_path = os.path.join(root_dir, f"{y}-{m}", f"data_{y}-{m}-{d}.pt")
    if os.path.isfile(data_path):
        return torch.load(data_path)
    return None

#======================================================
# Compute daily means for each label
#======================================================
def compute_label_means(root_dir, start_date, end_date):
    """
    Iterates day-by-day between start_date and end_date.
    For each day:
      - Loads the monthly label (0/255) for that month (cached)
      - Loads the daily 4-channel image
      - Where label == 255, computes the mean of those pixels for each channel
    Returns:
      dates: list of datetime objects
      means: a nested list: means[label_idx][channel_idx], each a list of daily means
    """
    dates = []
    means = [[[] for _ in range(4)] for _ in range(7)]  # 7 labels x 4 channels

    label_cache = {}
    day = start_date
    while day <= end_date:
        ym = (day.year, day.month)
        if ym not in label_cache:
            label_cache[ym] = get_month_label(root_dir, ym[0], ym[1])
        label_tensor = label_cache[ym]
        
        image_tensor = get_daily_image(root_dir, day)
        
        if label_tensor is not None and image_tensor is not None:
            dates.append(day)

            label_arr = label_tensor.numpy()  # shape [7, H, W]
            image_arr = image_tensor.numpy()  # shape [4, H, W]

            for label_idx in range(7):
                mask = (label_arr[label_idx] == 255)
                for chan_idx in range(4):
                    vals = image_arr[chan_idx][mask]
                    means[label_idx][chan_idx].append(vals.mean() if vals.size > 0 else np.nan)
                    
        day += timedelta(days=1)

    return dates, means

#======================================================
# Plot all 7 label curves in stacked subplots
#======================================================
def plot_label_wise_curves_stacked(dates, means, figpath):
    """
    Creates a single figure with 7 vertical subplots (one per label).
    Each subplot shows 4 lines (the daily mean for each channel).
    Saves the figure to `figpath`.
    """
    if not dates:
        print(f"[WARNING] No valid data for figure {figpath}, skipping plot.")
        return

    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 16), sharex=True)

    for label_idx in range(7):
        ax = axes[label_idx]
        for chan_idx in range(4):
            ax.plot(dates, means[label_idx][chan_idx], label=f"Channel {chan_idx}")
        ax.set_ylabel("Mean Value")
        ax.set_title(f"Label {label_idx}")
        if label_idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close(fig)
    print(f"[SAVED] {figpath}")

#======================================================
# Main loop over planet.* folders and time-series subfolders
#======================================================
if __name__ == "__main__":
    base_dir = "/media/thibault/DynEarthNet/raw_data"
    out_dir  = "/home/thibault/ProcessedDynamicEarthNet/figures"
    os.makedirs(out_dir, exist_ok=True)

    # We'll create a list of planet.* directories, excluding the ones that aren't planet.*:
    planet_folders = [
        d for d in os.listdir(base_dir)
        if d.startswith("planet.") and os.path.isdir(os.path.join(base_dir, d))
    ]

    # Date range for all time-series
    start_date = datetime(2018, 1, 1)
    end_date   = datetime(2019, 12, 31)

    for planet_folder in planet_folders:
        planet_folder_path = os.path.join(base_dir, planet_folder)

        # Each time series ID is a subfolder, e.g. "1311_3077_13"
        subdirs = [
            sd for sd in os.listdir(planet_folder_path)
            if os.path.isdir(os.path.join(planet_folder_path, sd))
        ]
        for ts_id in subdirs:
            ts_path = os.path.join(planet_folder_path, ts_id)
            
            # Compute daily means
            dates, means = compute_label_means(ts_path, start_date, end_date)

            # Build a figure filename for this planet.* + ts_id
            # Example: spectral_curves_planet.10N_1311_3077_13.png
            fig_name = f"spectral_curves_{planet_folder}_{ts_id}.png"
            figpath  = os.path.join(out_dir, fig_name)

            # Plot & save
            plot_label_wise_curves_stacked(dates, means, figpath)
