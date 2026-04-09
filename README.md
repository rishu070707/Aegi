*brother update the readme and push it according to the this* {AEGI – Weapon Detection System (Paper-Aligned)

Overview:
Real-time multi-class weapon detection using YOLOv8 + Flask.
Detects (paper classes): Handgun, Knife, Rifle, Shotgun.
Supports static image, video, and live webcam detection via dashboard.

Training (as per paper):
- Implements 5-fold training workflow (Fold 1–5) in `scripts/train_weapon_model.py`.
- Dataset preparation supports mapping/merging sources into the 4 paper classes via `scripts/prepare_weapon_dataset.py`.

Post-Processing (as per paper modules):
- Temporal Consistency Filtering
- Confidence Stabilization (Anti-Flicker)
- Context-Aware Risk Scoring
- Scene-Aware False Alarm Suppression
- Smart ROI Monitoring
- Automated Evidence Logging
- Alert Cooldown Mechanism
- Adaptive Edge Deployment Mode
- User Feedback Learning Loop  
(available under `post_processing/`)

Quick Setup:

1) Requirements
Python (3.10–3.13), Git

2) Clone
git clone https://github.com/rishu070707/Aegi.git
cd Aegi

3) Environment
Windows:
python -m venv .venv
.venv\Scripts\Activate.ps1

Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

4) Install
pip install -r requirements.txt

5) Run
python app.py

6) Open
http://localhost:5000

Important Files:
- `weapon_model.pt` → main detection model
- `yolov8n.pt` → person detection (scene-aware filtering support)
- `app.py` → entry point
- `detector.py` → inference pipeline
- `post_processing/` → all post-processing modules

Train:
python scripts/train_weapon_model.py --data path/to/dataset.yaml --weights yolov8s.pt --epochs 100 --imgsz 640 --batch 16}