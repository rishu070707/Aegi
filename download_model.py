"""
download_model.py
=================
Downloads REAL weapon detection YOLOv8 models from HuggingFace.

Two models available:
  1. Subh775/Threat-Detection-YOLOv8n  -- 4 classes: Gun, Knife, Grenade, ...  (81% mAP)
  2. Subh775/Firearm_Detection_Yolov8n -- 1 class:  Gun                        (89% mAP)

Run:    python download_model.py
Restart: python app.py
"""

import os
import sys
import subprocess

DEST = os.path.join(os.path.dirname(__file__), "weapon_model.pt")


def ensure_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[+] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def download_threat_model():
    """
    Download Subh775/Threat-Detection-YOLOv8n  (Gun / Knife / Grenade / Explosive)
    81.1% mAP@50  --  file: weights/best.pt
    """
    from huggingface_hub import hf_hub_download
    print("\n[1] Downloading Threat-Detection-YOLOv8n (Gun+Knife+Grenade, 81% mAP)...")
    path = hf_hub_download(
        repo_id="Subh775/Threat-Detection-YOLOv8n",
        filename="weights/best.pt",
    )
    import shutil
    shutil.copy(path, DEST)
    sz = os.path.getsize(DEST) / 1e6
    print(f"    Saved -> {DEST}  ({sz:.1f} MB)")
    return True


def download_firearm_model():
    """
    Fallback: Subh775/Firearm_Detection_Yolov8n  (single class: Gun, 89% mAP)
    """
    from huggingface_hub import hf_hub_download
    print("\n[2] Downloading Firearm_Detection_Yolov8n fallback (Gun only, 89% mAP)...")
    path = hf_hub_download(
        repo_id="Subh775/Firearm_Detection_Yolov8n",
        filename="weights/best.pt",
    )
    import shutil
    shutil.copy(path, DEST)
    sz = os.path.getsize(DEST) / 1e6
    print(f"    Saved -> {DEST}  ({sz:.1f} MB)")
    return True


def verify_model():
    print("\n[+] Verifying model...")
    from ultralytics import YOLO
    m = YOLO(DEST)
    print(f"    Classes: {list(m.names.values())}")
    print("    Model OK!")
    return True


def main():
    print("=" * 55)
    print("  Weapon Detection Model Downloader")
    print("=" * 55)

    if os.path.exists(DEST):
        sz = os.path.getsize(DEST) / 1e6
        print(f"\n[i] weapon_model.pt already exists ({sz:.1f} MB)")
        ans = input("    Re-download? (y/N): ").strip().lower()
        if ans != "y":
            print("    Keeping existing model. Run: python app.py")
            return

    # Ensure dependencies
    ensure_package("huggingface_hub")

    ok = False
    try:
        ok = download_threat_model()
    except Exception as e:
        print(f"    [!] Threat model failed: {e}")

    if not ok:
        try:
            ok = download_firearm_model()
        except Exception as e:
            print(f"    [!] Firearm model failed: {e}")

    if not ok:
        print("""
  Automatic download failed. Please download manually:

  Option A (preferred):
    1. Visit https://huggingface.co/Subh775/Threat-Detection-YOLOv8n/tree/main
    2. Download  weights/best.pt
    3. Copy here: weapon_model.pt

  Option B:
    pip install huggingface_hub
    python -c "from huggingface_hub import hf_hub_download; \\
      hf_hub_download('Subh775/Threat-Detection-YOLOv8n', \\
      'weights/best.pt', local_dir='.')"
    rename best.pt weapon_model.pt
""")
        return

    try:
        verify_model()
    except Exception as e:
        print(f"    [!] Verify failed: {e}")
        return

    print("""
  Done! Restart the app:
    python app.py

  The dashboard will now show:  WEAPON MODEL (instead of COCO MODE)
  All 4 classes active: Gun / Knife / Grenade / Explosive
""")


if __name__ == "__main__":
    main()
