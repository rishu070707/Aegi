# System Limitations and Known Issues

This document explicitly states what this system can and cannot do.

---

## What This System IS

✅ **A demonstration of post-processing techniques** that can reduce false positives from generic object detectors  
✅ **A Flask-based research prototype** for testing detection pipelines  
✅ **An academic platform** for ablation studies on filtering methods  
✅ **A reference implementation** of temporal consistency and EMA-based confidence stabilization

---

## What This System IS NOT

❌ **A production-ready weapon detection system**  
❌ **Custom-trained on a proprietary dataset**  
❌ **Suitable for actual law enforcement or military deployment without extensive validation**  
❌ **Forensic-grade evidence collection** (lacks cryptographic verification)  
❌ **Scalable for enterprise surveillance** (uses global variables, no load balancing)  
❌ **Robust to adversarial attacks** (not tested)  
❌ **Free from bias** (not evaluated on demographic diversity)

---

## Known Limitations

### 1. Model Provenance

- ❌ Uses COCO-pretrained YOLOv8s (NOT trained for weapons)
- ❌ Weapon capability depends on external Hugging Face model
- ⚠️ If weapon model fails to download, system silently downgrades

**Impact:** Without the auxiliary weapon model, "weapon" detection is just generic object detection with string-based label mapping.

### 2. Label Mapping Is Heuristic-Based

```python
if any(k in n for k in ("knife", "blade", "dagger")):
    return "Knife"
```

**Known False Positives:**

- Kitchen scissors → "Knife"
- Screwdrivers → potential match
- Toy guns → possible match
- Long implements (tools, brooms) → false alarms

**Impact:** High false positive rate in uncontrolled environments.

### 3. No Real Object Tracking

- ❌ No DeepSORT or ByteTrack
- ❌ IoU-based association is fragile
- ⚠️ Moving objects or camera shake breaks consistency

**Temporal Filter Problem:**

```python
def filter_detections(boxes, confidence, prev_boxes):
    # Matches based on IoU overlap only
    # Fails when object moves out-of-view and re-enters
    # Fails under occlusion
    # Not robust to camera motion
```

**Impact:** Temporal filtering can miss tracked threats if they leave/re-enter frame.

### 4. Scene Filtering Is Overly Simplistic

```python
# ROI-based filtering uses fixed bounding boxes
if not is_in_roi(box, roi_bounds):
    return None
```

**Missing:**

- Adaptive ROI learning
- Scene understanding (office vs street vs home)
- Lighting-based confidence adjustment
- Camera motion compensation

**Impact:** Requires manual ROI tuning per deployment location.

### 5. No Real Security Implementation

- ❌ No authentication (anyone can access)
- ❌ No authorization (no role-based access)
- ❌ No rate limiting (vulnerable to DoS)
- ❌ No encrypted storage
- ❌ No secure upload validation

**Code Example:**

```python
# app.py has no login mechanism
# Anyone can POST to /upload and process videos
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # Direct processing, no security checks
```

**Impact:** Not suitable for any real surveillance use case.

### 6. Evidence Logging Is NOT Forensic-Grade

**Paper claims:** "Forensic-grade evidence logging"  
**Code reality:** Saves frames to disk

**Missing:**

- Hash verification (can't detect tampering)
- Immutable storage (local filesystem, easily deleted)
- Chain of custody logging
- Encrypted evidence storage
- Signed timestamps

**Code Example:**

```python
# evidence_logger.py just saves frames
cv2.imwrite(f"evidence/{frame_id}.jpg", frame)
# No hashing, no verification, no protection
```

**Impact:** Evidence could be challenged in court as unverified.

### 7. Performance Claims Unsupported

**Paper claims:** 30 FPS, low latency  
**Code reality:**

- ❌ No reproducible benchmarking script
- ❌ No hardware specification
- ❌ No profiling logs
- ❌ Depends on undocumented GPU/CPU

**Impact:** FPS numbers cannot be independently verified.

### 8. TensorRT Integration Is Incomplete

**Paper suggests:** "Optimized TensorRT edge deployment"  
**Code reality:**

```python
if os.path.exists("weapon_model.engine"):
    load_engine()  # That's it.
```

**Missing:**

- FP16/INT8 conversion
- Calibration dataset
- Performance validation
- Quantization config

**Impact:** TensorRT claims are unsupported.

### 9. No Dataset Bias Analysis

**Missing:**

- ❌ Demographic bias testing
- ❌ Lighting fairness evaluation
- ❌ Skin tone robustness checks
- ❌ Environmental diversity testing

**Ethical Risk:** System may work well on training distribution but fail on new populations/environments.

### 10. Global Variables Everywhere

```python
latest_frame = None  # Shared across all sessions
latest_boxes = []    # Race conditions
webcam_active = False  # Can be overwritten by concurrent users
```

**Problems:**

- Multiple users can overwrite each other's state
- No session isolation
- Race conditions under concurrent access

**Impact:** System unstable with multiple simultaneous users.

### 11. Flask Architecture Not Scalable

- ❌ Synchronous inference (blocks on each request)
- ❌ No job queue
- ❌ No load balancing
- ❌ CPU-bound processing with GIL
- ❌ MJPEG streaming inefficient

**Impact:** Cannot handle more than a few concurrent streams.

### 12. No Unit Testing

- ❌ No pytest suite
- ❌ No regression tests
- ❌ No integration tests
- ⚠️ Stability and correctness unknown

**Impact:** Bugs can be introduced without detection.

### 13. Hardcoded Parameters Throughout

```python
EMA_ALPHA = 0.4              # Why 0.4? Not justified
TEMPORAL_WINDOW = 5          # Why 5? No ablation
CONFIDENCE_THRESHOLD = 0.5   # Arbitrary
COOLDOWN_TIME = 5000         # ms, not tuned
```

**Impact:** No adaptive tuning; parameters may be suboptimal for different scenarios.

### 14. Ablation Study Results Not Independently Verifiable

**README claims:**

```
| Configuration | False Positive Rate |
| Full Pipeline | 5.8%                |
```

**But:**

- ❌ Test set not provided
- ❌ Benchmark script not available
- ❌ Results not reproducible
- ❌ No statistical significance testing

**Impact:** These numbers cannot be verified by reviewers.

### 15. No Competitive Benchmarking

**Paper mentions:**

- Faster R-CNN
- SSD
- YOLOv5
- YOLOv7

**Code provides:**

- ❌ No Faster R-CNN implementation
- ❌ No SSD implementation
- ❌ No YOLOv5 comparison
- ❌ No YOLOv7 comparison

**Impact:** Comparative claims are unsupported.

---

## When It Works Well

✅ **Post-processing ablation studies** - Shows that each module reduces false positives  
✅ **Demo applications** - Nice Flask UI for testing concepts  
✅ **Teaching tool** - Good reference for pipeline architecture  
✅ **Baseline for improvement** - Can be extended with better models

---

## When It Fails

❌ **Real surveillance deployment** - Too many security/reliability gaps  
❌ **Mission-critical decisions** - No validation for high-stakes use  
❌ **Outdoor environments** - Limited to trained conditions  
❌ **Adversarial scenarios** - No robustness testing  
❌ **High-concurrency usage** - Global variable issues  
❌ **Legal/forensic purposes** - Evidence logging not certified

---

## Recommendations for Users

**If using for research:**

1. Be explicit about limitations in your paper
2. Don't claim weapon detection if auxiliary model isn't loaded
3. Validate performance on YOUR data, not just paper benchmarks
4. Add proper error handling and logging

**If considering for real deployment:**

1. ⚠️ DO NOT use without security review
2. ⚠️ DO NOT use for law enforcement without validation
3. ⚠️ Add authentication, rate limiting, audit logs
4. Implement proper object tracking (ByteTrack, DeepSORT)
5. Add cryptographic evidence verification
6. Test extensively on diverse environments
7. Get legal and ethical review before deployment

---

## Version History

| Version | Date       | Status                              |
| ------- | ---------- | ----------------------------------- |
| 1.0     | 2026-05-07 | Initial honest limitations document |
