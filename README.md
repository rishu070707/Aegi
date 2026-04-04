# 🎯 Real-Time Weapon Detection System

## 📖 Overview
This project presents an end-to-end AI-powered surveillance system designed to detect weapons such as firearms and knives in real-time using deep learning. The system leverages the **YOLOv8** object detection model to achieve high-speed and high-accuracy detection suitable for real-world deployment in environments like airports, schools, railway stations, and public places. Unlike traditional research models that focus only on accuracy, this project emphasizes deployability, including real-time inference, multi-modal input support, and robustness under adverse conditions.

## ⚠️ Problem Statement
Manual monitoring of surveillance footage is:
- ❌ Time-consuming
- ❌ Error-prone
- ❌ Not scalable

There is a critical need for an automated intelligent system that can detect weapons instantly, reduce human effort, and improve overall public safety.

## 💡 Proposed Solution
We developed a real-time weapon detection system that:
- Uses **YOLOv8** (You Only Look Once v8) for blazing-fast object detection.
- Works on multi-modal inputs: 
  - 🖼️ Images 
  - 🎥 Videos 
  - 📷 Live webcam streams
- Detects weapons with clear bounding boxes and confidence scores.
- Runs at real-time speeds (~47 FPS).

## ⚙️ Key Features
- ✔️ **Real-time detection** (low latency)
- ✔️ **Multi-input support** (image/video/webcam)
- ✔️ **High accuracy** (mAP@50 = 92%)
- ✔️ **Robust to adverse conditions**:
  - Low light
  - Motion blur
  - Occlusion
- ✔️ **Web-based deployment** using Flask
- ✔️ **Comparative analysis** with multiple models

## 🔬 Novel Contributions
This project improves over existing research through:
- **Ablation Study:** Compared YOLOv8n, YOLOv8s, and YOLOv8m variants. Selected the best model based on the ultimate speed + accuracy trade-off.
- **Custom Dataset:** Utilized 3,540 highly diverse images reflecting real-world conditions, thoroughly manually annotated.
- **Adverse Condition Testing:** Specifically tested and optimized for low light, blur, and occlusion.
- **Real Deployment System:** Built a functional Flask web app for live real-time streaming and monitoring.

## 🔄 Workflow / Pipeline
1. Data Collection
2. Annotation
3. Augmentation
4. Training (YOLOv8)
5. Evaluation
6. Deployment (Flask)
7. Real-Time Detection

## 💻 Requirements

### Software Requirements
- **Language:** Python 3.8+
- **Frameworks:** PyTorch, Ultralytics YOLOv8, Flask (for deployment)
- **Libraries:** OpenCV, NumPy, Pandas, Matplotlib
- **Annotation Tool:** LabelImg / Roboflow

### ⚡ Hardware Requirements
- **Minimum:** 
  - CPU: i5 / Ryzen 5 
  - RAM: 8 GB 
  - GPU: Optional
- **Recommended (for optimal FPS):** 
  - GPU: NVIDIA Tesla T4 / RTX 3050+ 
  - RAM: 16 GB

## 🧠 Model & Training
- **Model:** YOLOv8 (n, s, m variants tested)
- **Selected Model:** **YOLOv8s** (best trade-off)
- **Input Size:** 640×640
- **Epochs:** 50
- **Batch Size:** 16
- **Optimizer:** SGD
- **LR Scheduler:** Cosine Annealing

## 📊 Dataset Details
- **Total Images:** 3,540 images
- **Split:** 
  - Train: 66.7% 
  - Validation: 16.8% 
  - Test: 16.5%
- **Annotation Format:** YOLO (.txt)
- **Key Challenge:** 61.3% of objects in the dataset are small-sized, making it a difficult detection problem.

## 🚀 Deployment
- **Server:** Flask server
- **Interface:** Browser-based UI
- **Camera:** Webcam access supported

## 🔥 Performance & Accuracy
Our final model (**YOLOv8s**) boasts the following metrics:

| Metric | Value |
| --- | --- |
| **Precision** | 95% |
| **Recall** | 84% |
| **F1 Score** | 89% |
| **mAP@50** | **92%** |
| **mAP@50:95** | 71% |
| **FPS** | **47.3** |

# Aegi
