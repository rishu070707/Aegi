# Real-Time Multi-Class Weapon Detection System
### YOLOv8 with Flask-Based Multi-Modal Deployment and Advanced Post-Processing

**Paper:** "Real-Time Multi-Class Weapon Detection System Using YOLOv8 with Flask-Based Multi-Modal Deployment and Advanced Post-Processing Modules"  
**Authors:** Shambhavi Trivedi, Aekeesh Jaiswal, Vaishnavi Singh  
**Institution:** SRM Institute of Science and Technology, Delhi-NCR  

---

## 📌 Project Status

| Component | Status |
|-----------|--------|
| Flask Web App (9 post-processing modules) | ✅ Complete |
| Custom Dataset (multi-source) | ⏳ Download via Colab notebook |
| Ablation Study (3 models) | ⏳ Run after dataset ready |
| 5-Fold Cross-Validation | ⏳ Run after ablation |
| Custom YOLOv8s Training | ⏳ Run via Colab (GPU required) |

---

## 🧾 What This System Does

A complete, real-time multi-class weapon detection system trained on a **custom multi-source dataset** detecting **4 weapon classes**:

- 🔫 **Handgun**
- 🔪 **Knife**
- 🎯 **Rifle**
- 💥 **Shotgun**

Deployed via Flask with **3 input modalities**: image upload, video upload, live webcam.

### 9 Post-Processing Modules (all implemented)
1. Temporal Consistency Filtering
2. Confidence Stabilization (EMA, α=0.4)
3. Context-Aware Risk Scoring
4. Scene-Aware False Alarm Suppression
5. Smart Region-of-Interest (ROI) Monitoring
6. Automated Evidence Logging (forensic-grade)
7. Alert Cooldown Mechanism
8. Adaptive Edge Deployment Mode
9. User Feedback Learning Loop

---

## 🏋️ Training Pipeline

### Dataset (Section III of paper)

Custom multi-source dataset assembled from:

| Source | Description |
|--------|-------------|
| Google Open Images | Tens of thousands of weapon images, diverse global environments |
| Roboflow Universe | Community-curated weapon detection datasets |
| Kaggle Weapon Repos | Labeled weapon data across all 4 classes |
| Controlled CCTV Captures | Corridor, parking, transit, shopping environments (2.5–4m height, 15–40° tilt) |
| Synthetic Augmentation | Low-light, motion blur, occlusion simulation |

**Total: ~25,000 images** | **Split: 70% train / 15% val / 15% test**

### 3 Models — Ablation Study (Table II, paper)

Paper trains and compares 3 YOLOv8 variants:

| Model | mAP@50 | mAP@50:95 | FPS | Decision |
|-------|--------|-----------|-----|----------|
| YOLOv8n | ~0.923 | ~0.651 | highest | Too low accuracy |
| **YOLOv8s** | **0.961** | **0.712** | **real-time** | **✅ SELECTED** |
| YOLOv8m | ~0.967 | ~0.721 | lower | Marginal gain, too slow |

### 5-Fold Cross-Validation (Table III, paper)

YOLOv8s trained with 5-fold CV (80/20 split per fold, max 100 epochs, early stopping=15):

| Class | AP@50 |
|-------|-------|
| Handgun | 0.959 |
| Knife | 0.948 |
| Rifle | 0.974 |
| Shotgun | 0.963 |
| **Mean mAP@50** | **0.961** |

### Training Hyperparameters (Section V-B, paper)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning Rate | 0.01 → 0.001 (cosine annealing) |
| Momentum | 0.937 |
| Weight Decay | 5×10⁻⁴ |
| Batch Size | 16 |
| Image Size | 640×640 |
| Max Epochs | 100 |
| Early Stopping | patience = 15 |
| NMS Confidence | 0.25 |
| NMS IoU | 0.45 |
| Mosaic | 0.9 |
| Mixup | 0.15 |
| Random Erasing | 0.3 |
| Horizontal Flip | 0.5 |

---

## 🚀 Step-by-Step: How to Run

### Step 1 — Train Custom Model (Google Colab, T4 GPU)

> ⚠️ Your laptop is not required for training — run entirely in Colab (free T4 GPU)

1. Open **`Weapon_Detection_Training_Colab.ipynb`** in [Google Colab](https://colab.research.google.com/)
2. **Runtime → Change runtime type → T4 GPU**
3. Fill in the Quick Start Config cell (Roboflow API key — free at roboflow.com)
4. **Run All Cells** (top to bottom)
5. Download `dataset/` folder from `MyDrive/WeaponDetection/`
6. Paste into: `object detection project/dataset/`

The notebook will:
- Download custom dataset from 4 sources
- Apply synthetic augmentation (low-light, motion blur, occlusion)
- Create 5 fold YAML files
- Train YOLOv8n + YOLOv8s + YOLOv8m (ablation)
- Run 5-fold cross-validation on YOLOv8s
- Export best weights + all CSVs

---

### Step 2 — Ablation Study (local, after dataset ready)

```bash
python train.py
```

Trains all 3 model variants. Results → `ablation_results.csv`

---

### Step 3 — 5-Fold Cross-Validation (local)

```bash
python cross_validate.py
```

Results → `cross_validation_results.csv` and `class_metrics.csv`

---

### Step 4 — Run Flask Web App

```bash
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5000`

---

## 📁 Project Structure

```
object detection project/
│
├── 📓 Weapon_Detection_Training_Colab.ipynb  ← START HERE for training
│
├── 🏋️ Training Scripts
│   ├── train.py              ← Ablation: YOLOv8n / YOLOv8s / YOLOv8m
│   ├── cross_validate.py     ← 5-fold cross-validation (YOLOv8s)
│   ├── data.yaml             ← Dataset config (paths + class names)
│   └── hyp.yaml              ← All paper hyperparameters
│
├── 🌐 Flask Web App
│   ├── app.py                ← Main Flask application
│   ├── detector.py           ← YOLOv8 inference wrapper
│   └── templates/            ← HTML pages
│
├── ⚙️ Post-Processing Modules
│   └── post_processing/
│       ├── temporal_filter.py
│       ├── confidence_stabilizer.py
│       ├── risk_scorer.py
│       ├── scene_filter.py
│       ├── roi_monitor.py
│       └── evidence_logger.py
│
├── 📊 Dataset
│   └── dataset/
│       ├── train/images/     ← ~17,500 images (70%)
│       ├── valid/images/     ← ~3,750 images (15%)
│       ├── test/images/      ← ~3,750 images (15%)
│       ├── fold1/data_fold1.yaml
│       ├── fold2/data_fold2.yaml
│       ├── fold3/data_fold3.yaml
│       ├── fold4/data_fold4.yaml
│       └── fold5/data_fold5.yaml
│
├── 📈 Results (generated after training)
│   ├── ablation_results.csv
│   ├── cross_validation_results.csv
│   ├── class_metrics.csv
│   └── benchmark_results.csv
│
└── 📄 Documentation
    ├── HOW_TO_RUN_TRAINING.md
    ├── CHANGELOG.md
    └── requirements.txt
```

---

## 🔍 Detection Pipeline Flow

```
Input (Image / Video / Webcam)
         ↓
  CLAHE Enhancement (if luminance < 50/255)
         ↓
  YOLOv8s Inference (custom trained)
    conf=0.25, IoU=0.45
         ↓
  ┌──────────────────────────────────────┐
  │        POST-PROCESSING PIPELINE      │
  │  1. Temporal Consistency Filter      │
  │     (window=5, K=3 frames, τ=0.30)  │
  │  2. Confidence Stabilization (EMA)   │
  │     α = 0.4                          │
  │  3. Scene-Aware Suppression          │
  │     (human co-occurrence, ψ factor)  │
  │  4. Context-Aware Risk Scoring       │
  │     R = 0.5·Cs + 0.3·As + 0.2·Ps   │
  │  5. ROI Priority Scoring             │
  │  6. Alert Cooldown (Δt = 5s)        │
  └──────────────────────────────────────┘
         ↓
  Output: Annotated Frame + Risk Level
  + Evidence Log (if High risk)
```

---

## 📊 Key Results (Paper Claims)

| Metric | Value |
|--------|-------|
| mAP@50 (5-fold CV) | **0.961** |
| mAP@50:95 | **0.712** |
| FP Reduction (full pipeline) | **64.2%** |
| Latency overhead (post-processing) | **5.8 ms** |
| Live webcam FPS | **≥ 28 FPS** |
| Edge mode FPS (Jetson Nano) | **28.7 FPS** |

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` — YOLOv8
- `flask` — web application
- `opencv-python` — image/video processing
- `torch` — deep learning backend

**For training:** GPU required (Colab T4 recommended — free)

---

## 📝 Version

- **Version:** 3.0 (Custom Training + Full Pipeline)
- **Updated:** June 2026
- **Paper:** IEEE 2-Column Format — `Weapon_Detection_IEEE_2Column_Final.docx`
