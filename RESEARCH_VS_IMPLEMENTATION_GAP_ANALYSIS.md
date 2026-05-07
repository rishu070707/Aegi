# Research vs Implementation Gap Analysis

## Executive Summary

This document maps the 18 critical issues between paper claims and repository implementation. The goal is to realign the paper and code to be honest and reproducible.

---

## CRITICAL ISSUES REQUIRING IMMEDIATE FIX

### Issue #18: Model Provenance Is Fundamentally Unclear ⚠️ CRITICAL

**Paper Claims:** Custom-trained weapon detection model  
**Code Reality:**

- Primary model: `yolov8s.pt` = **COCO pretrained** (generic YOLO, NOT weapon-specific)
- Optional fallback: `weapon_model.pt` from Hugging Face (HF_WEAPON_REPO = "Subh775/Threat-Detection-YOLOv8n")
- This is **SOMEONE ELSE'S weights**, not this paper's training

**Academic Risk:** VERY HIGH

- Readers assume custom training
- If you used external weights, cite them properly
- Using uncited external weights = potential plagiarism of model

**Action Required:**

1. [ ] Clearly state in paper Abstract: "This work integrates publicly available YOLOv8s (COCO-pretrained) with post-processing pipeline"
2. [ ] Add citation: Hugging Face weapon model repo (give proper credit)
3. [ ] Remove any claim of "custom-trained" unless you actually did 25,000-image training
4. [ ] Add section: "Model Sources and Attribution"

---

### Issue #2: COCO Model Lacks Weapon Classes 🔴 CRITICAL

**Paper Claims:** Weapon detection with high accuracy  
**Code Reality:**

```python
MODEL_PATH = "yolov8s.pt"  # This is COCO weights
```

COCO dataset has 80 classes: person, car, dog, cat, etc.  
**COCO does NOT have:** handgun, rifle, shotgun, knife classes

**Functional Problem:**

- Without `weapon_model.pt` loaded, the system degrades to generic object detection
- A rifle might be detected as "person" or "background"
- SILENTLY unreliable without second model

**Code Evidence:**

```python
def _name_to_weapon(name: str):
    """Map raw YOLO class name to a weapon label, or None if not a weapon class."""
    # This tries to map COCO classes to weapons
    # It's a heuristic, not accurate detection
```

**Action Required:**

1. [ ] Paper must state: "Primary model uses COCO weights; weapon-specific detection requires auxiliary model"
2. [ ] Add fallback/error handling: Warn if `weapon_model.pt` fails to load
3. [ ] Add model card explaining COCO limitations
4. [ ] Remove claims of "high-accuracy weapon detection" unless weapon_model loads

---

### Issue #2b: Label Mapping Logic Is Heuristic-Based 🟠 HIGH

**Code:**

```python
if any(k in n for k in ("knife", "blade", "dagger", "scissors")):
    return "Knife"
```

**Problems:**

- Substring matching is fragile
- "Scissors" != "knife" (false positive)
- "Dagger" in "playground" → wrong
- No confidence threshold

**Paper Claims This As:** "Sophisticated weapon detection"  
**Code Shows:** String matching heuristics

**Action Required:**

1. [ ] Update paper to call this "label mapping layer" not "detection"
2. [ ] Document limitations: "Label mapping is heuristic-based and subject to false positives"
3. [ ] Add confidence threshold to label mapping
4. [ ] Provide false positive examples in paper

---

### Issue #16: fix_app.py Reveals Post-Hoc Patching 🔴 RED FLAG

**File:** `fix_app.py` suggests the paper and code diverged after submission

**Academic Red Flag:** Correction files suggest:

1. Original claims didn't match code
2. Code was patched AFTER paper submission
3. Reviewers will see this as scientific dishonesty

**Action Required:**

1. [ ] Delete or document `fix_app.py` with timestamp
2. [ ] Add CHANGELOG showing what changed and when
3. [ ] In paper appendix, explain: "Repository undergoes maintenance; see CHANGELOG for post-submission updates"

---

### Issue #3: Weapon Label Mapping Is Extremely Risky 🟠 HIGH

**Examples of False Positives:**
| Object | Code Mapping | Wrong? |
|--------|-------------|--------|
| Kitchen scissors | "scissors" ∈ knife group | YES |
| Toy gun | substring match → "handgun" | YES |
| Screwdriver | "blade" similar shape | RISKY |
| Decorative sword | "sword" ∈ knife group | YES |
| Pruning shears | "blade" in name | YES |

**Paper Language vs Code Reality:**

| Paper Says                         | Code Does                 |
| ---------------------------------- | ------------------------- |
| "Advanced false alarm suppression" | String substring matching |
| "Context-aware filtering"          | Fixed IoU thresholds      |
| "Intelligent scene understanding"  | Hardcoded ROI bounds      |

**Action Required:**

1. [ ] Add disclaimer: "Post-processing is heuristic-based; accuracy depends heavily on ground truth data"
2. [ ] Provide confusion matrix with false positive examples
3. [ ] Test on diverse environments (kitchen, street, office)
4. [ ] Document performance degradation on out-of-distribution images

---

### Issue #17: No Competitive Benchmarking 🔴 NO BASELINES

**Paper Claims vs Code:**

Paper comparison table:

```
| System | mAP@50 |
|--------|--------|
| Faster R-CNN | 0.85 |
| SSD | 0.82 |
| YOLOv5 | 0.88 |
| **OUR METHOD** | **0.93** |
```

Code contains:

- ❌ No Faster R-CNN benchmark
- ❌ No SSD benchmark
- ❌ No YOLOv5 benchmark
- ✓ Only YOLOv8s tests

**Proof:** No scripts in `scripts/` or `evaluation/` comparing against these baselines

**Action Required:**

1. [ ] Either add benchmark scripts for competitors OR
2. [ ] Remove competitive claims from paper
3. [ ] Replace with: "Comparison of YOLOv8 variants with post-processing"

---

### Issue #10: FPS Claims Are Unsupported 🟠 HIGH

**Paper Claims:**

- 30 FPS on edge
- Low latency inference
- Real-time surveillance

**Code:**

- No `benchmark.py` script
- No profiler output
- No hardware specs (GPU? CPU? Edge device?)
- FPS numbers in README are unverified

**Action Required:**

1. [ ] Add `scripts/benchmark.py` that:
   - Tests on CPU, GPU, Jetson Nano
   - Logs full hardware specs
   - Measures end-to-end latency
   - Saves reproducible results
2. [ ] Update paper with actual measured values
3. [ ] Add error bars (std deviation)
4. [ ] Run on specified hardware (Jetson Nano model, NVIDIA driver version, etc.)

---

### Issue #11: TensorRT Export Is Incomplete 🟠 HIGH

**Paper Claims:** "TensorRT optimization for edge deployment"  
**Code:** Just file existence check

```python
engine_path = "weapon_model.engine"
if os.path.exists(engine_path):
    # Use engine
```

**Missing:**

- No conversion pipeline
- No calibration script
- No validation that .engine is valid
- No quantization config
- No performance comparison

**Action Required:**

1. [ ] Add `scripts/export_tensorrt_proper.py`:
   - Actual FP16/INT8 conversion
   - Calibration dataset
   - Validation script
2. [ ] Add docs on TensorRT setup
3. [ ] Benchmark .engine vs .pt on Jetson
4. [ ] Update paper with actual TensorRT results (or remove claims)

---

### Issue #1: Mixing Demo with Research 🔴 STRUCTURAL

**Current Architecture Problem:**

```
app.py (Flask demo)
├─ detector.py (YOLO wrapper)
├─ post_processing/ (heuristics)
└─ Global variables everywhere
```

**Paper Presents As:** Production-grade research system  
**Code Shows:** Flask demo with add-ons

**Action Required - Create separation:**

```
README.md (honest about what this is)
├─ research/ (paper-grade implementation)
│  ├─ training/ (if custom training done)
│  ├─ evaluation/ (validation scripts)
│  └─ configs/ (reproducible settings)
├─ demo/ (Flask app - clearly labeled as demo)
│  ├─ app.py
│  ├─ templates/
│  └─ NOTE_THIS_IS_DEMO.md
└─ RESEARCH_STATUS.md (honest about limitations)
```

---

## ISSUES THAT ARE LESS CRITICAL BUT MATTER

### Issue #12: No Separation Between Training & Runtime

**Missing:**

- Training scripts assume paper training completed
- No training code provided
- Evaluation mixes demo with research

**Fix:** Document which parts are provided vs assumed completed

---

### Issue #5: Global Variables Everywhere

**Problem:** Not relevant for demo, but stated as production system

**Fix:** Either fix with proper session management OR be honest: "This is a research demo, not production code"

---

### Issue #8: Evidence Logging Not Forensic-Grade

**Paper Says:** "Forensic-grade evidence logging"  
**Code Does:** Saves frames to disk

**Missing:**

- Hash verification
- Tamper detection
- Chain of custody

**Fix:** Either implement properly OR change wording to "Evidence logging for audit trail"

---

## HONEST PAPER REWRITE CHECKLIST

### What CAN Be Claimed (Supported by Code):

- ✅ Post-processing pipeline reduces false positives
- ✅ Multi-stage filtering architecture (temporal, EMA, ROI, scene)
- ✅ Ablation study showing effectiveness of each module
- ✅ Flask-based demo application
- ✅ Integration with YOLO models

### What CANNOT Be Claimed (Unsupported):

- ❌ Custom-trained weapon detection model (using external weights)
- ❌ COCO-based weapon detection (COCO has no weapon classes)
- ❌ Verified FPS benchmarks (no reproducible profiling)
- ❌ TensorRT-optimized inference (incomplete)
- ❌ Forensic-grade logging (no cryptographic verification)
- ❌ Production-ready deployment (global variables, no security)
- ❌ Competitive baselines tested (no competitor implementations)

---

## PRIORITY ACTION ITEMS

**This Week:**

1. [ ] Update README with honest model provenance
2. [ ] Add MODEL_SOURCES.md citing Hugging Face weapon model
3. [ ] Create HONEST_LIMITATIONS.md file
4. [ ] Remove exaggerated claims from abstract

**This Sprint:** 5. [ ] Add reproducible benchmark script 6. [ ] Create proper TensorRT export pipeline 7. [ ] Add ablation study validation 8. [ ] Document post-processing parameter choices

**For Next Paper Submission:** 9. [ ] Either implement missing features OR rewrite paper 10. [ ] Add "Limitations" section explicitly 11. [ ] Provide reproducible configs 12. [ ] Separate demo from research code

---

## Questions to Answer in Paper

1. **Where do the weights come from?** ← CRITICAL
2. **Is this custom-trained or using pretrained models?** ← CRITICAL
3. **What are the actual FPS measured on real hardware?**
4. **What are the known false positive cases?**
5. **How does this compare to competitors? (Or why no comparison?)**
6. **What are the limitations?** ← Currently missing

---

## Red Flags for Reviewers

| Flag                           | Current Status |
| ------------------------------ | -------------- |
| Model provenance unclear       | 🔴 CRITICAL    |
| Claims exceed implementation   | 🔴 CRITICAL    |
| No reproducible benchmarks     | 🔴 CRITICAL    |
| Post-hoc patching evidence     | 🟠 HIGH        |
| Competitive claims unsupported | 🟠 HIGH        |
| Security issues hidden         | 🟠 HIGH        |
| No proper testing              | 🟠 HIGH        |
