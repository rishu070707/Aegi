# Real-Time Multi-Class Weapon Detection System

> **mAP@50 = 0.961 · FP Reduction = 64.2% · Latency Overhead = 5.8ms**

A production-ready, real-time weapon detection system built on YOLOv8s with a Flask-based multi-modal deployment platform. Detects four weapon classes — **Handgun, Knife, Rifle, Shotgun** — from images, videos, and live webcam streams, with a full 10-stage advanced post-processing pipeline.

---

## Architecture Overview

```
Frame Input
    │
    ▼
[1] CLAHE Preprocessing (adaptive low-light enhancement)
    │
    ▼
[2] YOLOv8s Inference (640×640, conf=0.25, IoU=0.45)
    │
    ▼
[3] Temporal Consistency Filter (5-frame sliding window, K=3 hits)
    │
    ▼
[4] EMA Confidence Stabilizer (α=0.4 anti-flicker)
    │
    ▼
[5] Scene-Aware Filter (YOLOv8n person detector, ψ multiplier)
    │
    ▼
[6] ROI Monitor (polygonal zone check, Ps elevation)
    │
    ▼
[7] Context-Aware Risk Scorer (R = 0.5·Cs + 0.3·As + 0.2·Ps)
    │
    ▼
[8] Alert Cooldown (5s per-class per-region debounce)
    │
    ▼
[9] Evidence Logger (PNG snapshot + JSON metadata)
    │
    ▼
[10] Annotated Frame → MJPEG Stream / Image Response / Video File
```

---

## Project Structure

```
object detection project/
├── app.py                          # Flask application entry point
├── detector.py                     # YOLOv8 wrapper + CLAHE preprocessing
├── requirements.txt
├── post_processing/
│   ├── __init__.py
│   ├── temporal_filter.py          # Sliding-window temporal consistency
│   ├── confidence_stabilizer.py    # EMA anti-flicker
│   ├── risk_scorer.py              # Context-aware risk R score
│   ├── scene_filter.py             # Person co-occurrence filter
│   ├── roi_monitor.py              # Polygonal ROI zone monitoring
│   ├── evidence_logger.py          # PNG + JSON evidence snapshots
│   ├── alert_cooldown.py           # Per-class 5s cooldown
│   ├── edge_mode.py                # Adaptive latency-based model switching
│   └── feedback_loop.py            # Operator feedback CSV logger
├── templates/
│   └── index.html                  # Dark security dashboard UI
├── static/
│   ├── style.css                   # Dashboard styling
│   └── js/app.js                   # Frontend logic
├── evidence_logs/                  # Auto-saved detection evidence
├── feedback_data/                  # Operator feedback CSV
├── yolov8s.pt                      # YOLOv8s COCO pretrained weights
└── yolov8n.pt                      # YOLOv8n for person detection
```

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

---

## Usage

### 📷 Image Detection
1. Click the **Image** tab
2. Drag & drop or click to upload a JPG/PNG/BMP image
3. Click **Run Detection**
4. View annotated image with bounding boxes and risk-level color coding
5. Use ✓/✗ buttons to provide operator feedback

### 🎬 Video Processing
1. Click the **Video** tab
2. Upload an MP4/AVI/MOV video (up to 500 MB)
3. Click **Process Video**
4. Download the annotated output video with all detections drawn per-frame

### 📹 Live Webcam
1. Click the **Live Webcam** tab
2. Click **Start Stream**
3. Live MJPEG feed with real-time detections overlaid
4. Use the **✏️ Draw ROI** tool to define regions of interest:
   - Click to add polygon points
   - Right-click (or press Enter) to close the polygon
   - Press Escape to cancel current polygon
5. Active detections panel shows class, confidence, and risk level in real-time

### 📁 Evidence Log
- Auto-saved PNG snapshots + JSON metadata for all High/Medium risk confirmed detections
- View thumbnails and download individual evidence files from the **Evidence Log** tab

---

## Post-Processing Pipeline Details

| Module | Description |
|--------|-------------|
| **Temporal Filter** | Requires same class in ≥3/5 consecutive frames (conf ≥ 0.30) |
| **EMA Stabilizer** | Per-class confidence smoothing: S(t) = 0.4·C(t) + 0.6·S(t-1) |
| **Scene Filter** | Person proximity multiplier ψ: 1.0 (co-located), 0.75 (present), 0.50 (absent) |
| **Risk Scorer** | R = 0.5·Cs + 0.3·As + 0.2·Ps → Low/Medium/High |
| **Alert Cooldown** | 5s per (class × region) to prevent alert fatigue |
| **Edge Mode** | Auto-switches to YOLOv8n/512 if latency > 40ms |

---

## Performance Metrics (Paper)

| Metric | Value |
|--------|-------|
| mAP@50 | **96.1%** |
| False Positive Reduction | **64.2%** |
| Post-Processing Latency Overhead | **5.8 ms** |
| Target Classes | Handgun, Knife, Rifle, Shotgun |

---

## Demo Mode

The application runs in **demo mode** by default (`DEMO_MODE = True` in `app.py`).
In demo mode, synthetic weapon detections are injected to allow full pipeline testing without custom model weights.

To use a fine-tuned weapon detection model:
1. Set `DEMO_MODE = False` in `app.py`
2. Replace `yolov8s.pt` with your fine-tuned model path
3. Update `MODEL_PATH` in `app.py` accordingly

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/detect/image` | Detect weapons in uploaded image |
| POST | `/detect/video` | Process uploaded video |
| POST | `/stream/start` | Start webcam capture threads |
| POST | `/stream/stop` | Stop webcam capture |
| GET | `/stream` | MJPEG webcam stream |
| POST | `/feedback` | Record operator feedback |
| POST | `/set_roi` | Set ROI zone polygon coordinates |
| POST | `/clear_roi` | Clear all ROI zones |
| GET | `/evidence` | List all evidence entries |
| GET | `/evidence/<file>` | Serve evidence image/video file |
| GET | `/api/status` | System status JSON |
| GET | `/api/live_detections` | Current live detection list |
