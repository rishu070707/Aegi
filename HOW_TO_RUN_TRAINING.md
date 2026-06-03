# Weapon Detection — How to Run (Step-by-Step)

## Overview
Paper requires:
1. **3 models trained** — YOLOv8n, YOLOv8s, YOLOv8m (ablation study → Table II)
2. **5-fold cross-validation** on YOLOv8s → Table III results
3. **Custom dataset** from Google Open Images + Roboflow + Kaggle + CCTV captures

## Step 1: Prepare Dataset (Google Colab — GPU required)

1. Open: `Weapon_Detection_Training_Colab.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom
4. Download the generated `dataset/` folder from your Google Drive
5. Unzip and place it here: `object detection project/dataset/`

The dataset folder should look like:
```
dataset/
├── train/images/   ← ~17,500 images (70%)
├── valid/images/   ← ~3,750 images (15%)
├── test/images/    ← ~3,750 images (15%)
├── fold1/data_fold1.yaml
├── fold2/data_fold2.yaml
├── fold3/data_fold3.yaml
├── fold4/data_fold4.yaml
└── fold5/data_fold5.yaml
```

## Step 2: Run Ablation Study (Train 3 Models)

```bash
python train.py
```

This trains YOLOv8n, YOLOv8s, YOLOv8m and saves results to `ablation_results.csv`.
Expected output (Table II from paper):

| Model    | mAP@50 | mAP@50:95 |
|----------|--------|-----------|
| YOLOv8n  | ~0.923 | ~0.651    |
| YOLOv8s  | ~0.961 | ~0.712    |
| YOLOv8m  | ~0.967 | ~0.721    |

**YOLOv8s selected** as best speed-accuracy tradeoff.

## Step 3: Run 5-Fold Cross-Validation

```bash
python cross_validate.py
```

Results saved to `cross_validation_results.csv`.
Expected (Table III from paper):

| Class   | AP@50 |
|---------|-------|
| Handgun | 0.959 |
| Knife   | 0.948 |
| Rifle   | 0.974 |
| Shotgun | 0.963 |
| **Mean**| **0.961** |

## Hyperparameters (paper Section V-B)

| Parameter    | Value                  |
|-------------|------------------------|
| Optimizer   | SGD                    |
| LR          | 0.01 → 0.001 (cosine) |
| Momentum    | 0.937                  |
| Weight Decay| 5×10⁻⁴                |
| Batch Size  | 16                     |
| Image Size  | 640×640                |
| Max Epochs  | 100                    |
| Early Stop  | patience=15            |
| NMS conf    | 0.25                   |
| NMS IoU     | 0.45                   |
