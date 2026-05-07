# Weapon Detection Research Prototype: Post-Processing Pipeline

⚠️ **RESEARCH PROTOTYPE ONLY - NOT FOR PRODUCTION USE**

**READ FIRST:** See [SYSTEM_LIMITATIONS.md](SYSTEM_LIMITATIONS.md) and [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) for critical information.

This is a Flask demo application for testing post-processing techniques applied to generic object detection. **This is NOT a custom-trained weapon detection system.**

---

## What This Is

✅ **Research Prototype** - Academic evaluation of post-processing pipeline  
✅ **Demo Application** - Flask interface for testing detection pipeline  
✅ **Code Reference** - Example implementation of filtering modules  
✅ **Pipeline Architecture** - Shows how post-processing can improve detection  

## What This IS NOT

❌ **Custom-Trained Model** - Uses external pretrained models  
❌ **Production System** - Not suitable for deployment  
❌ **Weapon Detection System** - Depends on auxiliary model availability  
❌ **Forensic-Grade** - Basic evidence logging only  
❌ **Secure** - No authentication or security features  

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
cd d:\Projects\Aegi\Aegi
python app.py
```

Navigate to `http://localhost:5000`

### 3. Check Model Setup
```bash
python check_model_setup.py
```

This will show you what models are available and system capabilities.

---

## Model Configuration

### Primary Model: YOLOv8s (COCO)
- **Source:** Ultralytics YOLOv8
- **Training Data:** COCO 2017 (80 generic object classes)
- **Weapon Classes:** NONE
- **Purpose:** Generic object detection baseline

### Optional: Weapon Model (HuggingFace)
- **Source:** Subh775/Threat-Detection-YOLOv8n
- **Purpose:** Weapon-specific detection (if available)
- **Fallback:** System uses string-based label mapping if unavailable

**Important:** Without the auxiliary weapon model, system cannot reliably detect weapons.

---

## Key Limitations

1. **Heuristic-Based Processing** - Post-processing uses rule-based filtering, not ML
2. **Global Variables** - Not thread-safe, unsuitable for concurrent access
3. **Synchronous Inference** - Not scalable, blocks on each request
4. **String-Based Labels** - Label mapping produces false positives
5. **No Security** - No authentication, authorization, or rate limiting
6. **Hardcoded Parameters** - Thresholds manually set, not tuned
7. **Basic Evidence Logging** - Saves frames locally, no verification
8. **COCO Model Dependency** - Generic detection model limits weapon detection

**For complete limitations, see [SYSTEM_LIMITATIONS.md](SYSTEM_LIMITATIONS.md)**

---

## Project Structure

```
Aegi/
├── app.py                      # Flask application
├── detector.py                 # YOLO wrapper
├── check_model_setup.py        # Diagnostic script
│
├── post_processing/            # Pipeline modules
│   ├── temporal_filter.py
│   ├── confidence_stabilizer.py
│   ├── scene_filter.py
│   ├── risk_scorer.py
│   ├── roi_monitor.py
│   └── ... (other modules)
│
├── evaluation/                 # Ablation and analysis
│   ├── ablation_study.csv
│   └── statistics/
│
├── configs/                    # Configuration files
│   ├── baseline.yaml
│   └── fold*.yaml
│
└── Documentation/
    ├── EXECUTIVE_SUMMARY.md
    ├── SYSTEM_LIMITATIONS.md
    ├── MODEL_SOURCES_AND_ATTRIBUTION.md
    ├── BEFORE_AFTER_CLAIMS_ANALYSIS.md
    └── CHANGELOG.md
```

---

## Running Tests

### Check Model Availability
```bash
python check_model_setup.py
```

Output shows:
- YOLOv8s (COCO) model status
- Weapon model availability
- System behavior implications

### Run Benchmarks
```bash
python scripts/benchmark.py
```

---

## Understanding the Pipeline

### Detection Flow
```
Input Image
    ↓
Enhancement (CLAHE)
    ↓
YOLOv8s Detection (Generic Objects)
    ↓
Label Mapping (to Weapon Categories)
    ↓
Post-Processing Filters:
  - Temporal Consistency
  - Confidence Stabilization (EMA)
  - Scene-Aware Filtering
  - Risk Scoring
  - Alert Cooldown
    ↓
Output: Detections + Risk Scores
```

### Post-Processing Module Effectiveness
| Module | Contribution |
|--------|---|
| Temporal Filter | -38% false positives |
| EMA Smoothing | -49% false positives |
| ROI Monitor | -61% false positives |
| Full Pipeline | -68% false positives |

**Note:** These are on test set only. Generalization to new data unknown.

---

## Configuration

### Edit Pipeline Parameters
**File:** `app.py`

```python
# Post-processing parameters
EMA_ALPHA = 0.4              # Confidence smoothing
TEMPORAL_WINDOW = 5          # Frames for consistency
CONFIDENCE_THRESHOLD = 0.5   # Detection threshold
COOLDOWN_TIME = 5000         # Alert throttling (ms)
```

### Add Custom ROI Zones
**File:** `post_processing/roi_monitor.py`

Define regions where detection should be prioritized.

---

## Documentation

**Start Here:**
- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - Overview of project status

**For Understanding Limitations:**
- [SYSTEM_LIMITATIONS.md](SYSTEM_LIMITATIONS.md) - What works/doesn't work
- [MODEL_SOURCES_AND_ATTRIBUTION.md](MODEL_SOURCES_AND_ATTRIBUTION.md) - Model provenance

**For Development:**
- [ACTION_PLAN_THIS_WEEK.md](ACTION_PLAN_THIS_WEEK.md) - Implementation roadmap
- [DOCUMENTATION_GUIDE.md](DOCUMENTATION_GUIDE.md) - Guide to all documentation

---

## Diagnostic Information

### View Available Models
```bash
python check_model_setup.py
```

### Test Without Weapon Model
Simply remove or rename `weapon_model.pt` and run. System will:
- Use generic COCO detection
- Apply string-based label mapping
- Fall back to heuristic classification

### Monitor System
```bash
# Watch logs in real-time
tail -f evidence_logs/*.log
```

---

## Contributing

Improvements welcome! Focus areas:
- [ ] Better object tracking (ByteTrack, DeepSORT)
- [ ] Adaptive parameter tuning
- [ ] Security hardening
- [ ] Cross-domain evaluation
- [ ] Unit tests
- [ ] Benchmarking scripts

---

## Disclaimer

**This is research code for academic evaluation only.**

- Not validated for real-world deployment
- Use at your own risk
- Subject to all limitations documented
- Requires expert review before any operational use

---

## Version Information

- **Version:** 2.0 (Honest Revision)
- **Updated:** 2026-05-07
- **Status:** Research Prototype

See [CHANGELOG.md](CHANGELOG.md) for version history.
