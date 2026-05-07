# Model Sources and Attribution

## Critical Disclosure: This Project Uses External Pretrained Weights

This research project does **NOT** provide custom-trained weapon detection models. Instead, it integrates publicly available pretrained models with a post-processing pipeline.

---

## Model 1: Primary Generic Detector (COCO)

**Model Name:** `yolov8s.pt`  
**Source:** Ultralytics YOLOv8 (COCO-pretrained)  
**Link:** https://github.com/ultralytics/ultralytics  
**Training Data:** COCO 2017 dataset (80 generic object classes)  
**License:** AGPL-3.0 (Ultralytics)

**Important Limitation:**

```
COCO Training Classes: person, car, dog, cat, bicycle, ...
WEAPON CLASSES IN COCO: NONE

This model is NOT trained on weapon detection.
It cannot directly detect handguns, rifles, or knives.
```

**How It's Used in This Repo:**

- Generic object detector as preprocessing step
- Label names are mapped heuristically to weapon categories
- Performance depends entirely on if auxiliary weapon model loads

---

## Model 2: Weapon-Specific Detector (Optional)

**Model Name:** `weapon_model.pt`  
**Source:** Hugging Face Hub  
**Repository:** Subh775/Threat-Detection-YOLOv8n  
**Link:** https://huggingface.co/Subh775/Threat-Detection-YOLOv8n  
**Training Data:** Unknown (Hugging Face user-provided model)  
**License:** Check original repo (likely MIT/Apache)

**Loaded Via:**

```python
from huggingface_hub import hf_hub_download
weapon_model_path = hf_hub_download(
    repo_id="Subh775/Threat-Detection-YOLOv8n",
    filename="weights/best.pt"
)
```

**Important:**

- This model is trained by a third party
- If it fails to download, system silently reverts to COCO-only detection
- **This paper does NOT train this model**
- Credit for weapon detection capability belongs to the original author

---

## Architecture Used

**Core Framework:** Ultralytics YOLOv8  
**Version:** ultralytics >= 8.1.0  
**Citation:**

```bibtex
@software{yolov8,
  author = {Ultralytics},
  title = {YOLOv8},
  url = {https://github.com/ultralytics/ultralytics},
  year = {2023}
}
```

---

## What THIS Paper Contributes

This repository's novel contribution is **NOT** the detection models themselves, but rather:

1. **Post-Processing Pipeline** (`post_processing/`)
   - Temporal consistency filtering
   - Confidence stabilization (EMA)
   - Scene-aware false alarm suppression
   - Risk scoring and alert cooldown

2. **Evaluation Framework** (`evaluation/`)
   - Ablation studies showing pipeline effectiveness
   - Multi-fold cross-validation setup

3. **Application Platform** (`app.py`)
   - Flask-based demo integrating the pipeline

---

## Model Availability & Reproducibility

**If reproducing this work:**

1. COCO weights auto-download via `ultralytics`
2. Weapon model tries to auto-download from Hugging Face
3. If weapon model unavailable: system degrades to COCO-only detection

**To guarantee reproducibility:**

```bash
# Download explicitly
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Subh775/Threat-Detection-YOLOv8n",
    filename="weights/best.pt",
    cache_dir="./models"
)
```

---

## Honest Comparison Table

| Aspect                      | YOLOv8s (COCO) | + Weapon Model | + Post-Processing |
| --------------------------- | -------------- | -------------- | ----------------- |
| Weapon classes trained      | ❌ No          | ✅ Yes         | N/A               |
| Generic object detection    | ✅ Yes         | ✅ Yes         | N/A               |
| False positive suppression  | ❌ No          | ❌ No          | ✅ Yes            |
| Designed for surveillance   | ❌ No          | ⚠️ Maybe       | ✅ Yes            |
| Custom trained (this paper) | ❌ No          | ❌ No          | N/A               |

---

## Security & Ethical Considerations

**Weapon Detection Risks:**

- Potential for false positives on innocent objects
- Privacy implications of surveillance
- Could be misused for social monitoring
- Requires responsible deployment practices

**This repo is for RESEARCH ONLY:**

- Not recommended for actual surveillance without proper vetting
- Requires expert review before deployment
- Subject to local laws and regulations

---

## How to Cite This Work

If using this repository, please cite both the pipeline contribution AND the upstream models:

```bibtex
@article{this_paper,
  title={Your Paper Title},
  author={Your Names},
  year={2026}
}

@software{yolov8,
  author = {Ultralytics},
  title = {YOLOv8},
  url = {https://github.com/ultralytics/ultralytics},
  year = {2023}
}

@software{weapon_model_hf,
  author = {Subh775},
  title = {Threat-Detection-YOLOv8n},
  url = {https://huggingface.co/Subh775/Threat-Detection-YOLOv8n}
}
```

---

## Version History

| Version | Date       | Change                                     |
| ------- | ---------- | ------------------------------------------ |
| 1.0     | 2026-05-07 | Initial honest disclosure of model sources |
