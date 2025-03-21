# 🛰️ ProcessedDynamicEarthNet

The goal of this project is to train and evaluate two different models for **pixel-wise semantic segmentation** of Earth observation images:

- **U-Net**: Trained on temporally stacked data (one tensor per month).
- **SPDNet**: Trained on covariance matrices computed from spatio-temporal patches of the data.

Both models aim to predict **land cover classes for each pixel** from satellite imagery.

---

## 📦 Download and Prepare the DynamicEarthNet Dataset

1. Download the dataset from the official [DynamicEarthNet GitHub repository](https://github.com/aysim/dynnet).

2. After downloading and extracting the data (e.g., `planet.10N`, `planet.11N`, etc.), use the following scripts to pre-process it:

   - **Step 1**: Convert `.tif` imagery and labels into PyTorch tensors  
     ```bash
     python load_data.py
     ```  
     To display monthly labeled images for all processed planet tiles in all planet* directories in `./DynEarthNet`, run:  
     ```bash
     python inspect_and_display_labels.py
     ```
   It saves all figures in `./ProcessedDynamicEarthNet/figures/labeled_images_displayed`

   - **Step 2 (For U-Net)**: Generate temporally stacked tensors and label maps for each month  
     ```bash
     python month_stacked_labels.py
     ```
  For each planet tile of planet.* diectory, it saves monthly tensors in `./DynEarthNet/datasets/unet/planet.*/<planet_tile_id>`
   - **Step 3 (For SPDNet)**: Compute pixel-wise covariance matrices and extract labels in blocks  
     ```bash
     python cov_label.py
     ```
For each planet tile of planet.* diectory, it saves monthly tensors in `./DynEarthNet/datasets/spdnet/planet.*/<planet_tile_id>`


---

## 🧠 Model Training Pipelines

### 1. U-Net Pipeline
Train a U-Net model using temporally stacked monthly data:  
```bash
python unet_pipeline.py
```

### 2. SPDNet Pipeline
   Train a SPDNet model using block-wise covariance matrices: 
   ```bash
    python spdnet_pipeline.py
   ```

   It requires anotherspdnet library from [AnotherSPDNet repository](https://github.com/AmmarMian/anotherspdnet/tree/main).
   It processes covariance blocks and performs pixel classification.

### For scheduling
In order to process automatically all planet.* folders on spdnet (without batchnorm) and unet successively, run
```bash
python weekend_scheduler.py
```
   
---

*Remark*:  Due to variability in the temporal dimension T (28, 30, 31 days), padding is applied where necessary for consistency.

---
*References*:
U-Net paper: https://arxiv.org/pdf/1505.04597
SPDNet paper: https://arxiv.org/pdf/1608.04233


