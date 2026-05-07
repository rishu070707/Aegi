#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_model_setup.py

Diagnostic script to analyze model capabilities and actual behavior.
Run this to understand what's actually happening in the detection pipeline.

This is NOT part of the main application - use for debugging and verification only.
"""

import os
import sys

def check_model_setup():
    """Analyze what models are available and what the system can actually do."""
    
    print("\n" + "="*70)
    print("MODEL SETUP DIAGNOSTIC")
    print("="*70)
    
    # Check YOLO model
    print("\n[1] YOLOv8s (COCO) Model:")
    print("-" * 70)
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8s.pt')
        print(f"✓ Model loaded successfully")
        print(f"  Model size: {os.path.getsize('yolov8s.pt') / (1024**2):.1f} MB")
        
        # List classes
        classes = model.names
        print(f"  Total classes: {len(classes)}")
        print(f"  Weapon classes in COCO: NONE")
        print(f"  Sample classes: {list(classes.values())[:10]}")
        
        print("\n  ⚠️  LIMITATION:")
        print("      COCO dataset (80 classes) does NOT include:")
        print("      - Handgun / Pistol")
        print("      - Rifle / Assault weapon")
        print("      - Shotgun")
        print("      - Knife / Blade")
        print("\n      Therefore: WITHOUT auxiliary model, this cannot detect weapons")
        
    except Exception as e:
        print(f"✗ Error loading YOLOv8s: {e}")
        sys.exit(1)
    
    # Check weapon model
    print("\n[2] Weapon Model (Hugging Face):")
    print("-" * 70)
    weapon_model_path = "weapon_model.pt"
    
    if os.path.exists(weapon_model_path):
        print(f"✓ File exists: {weapon_model_path}")
        print(f"  Size: {os.path.getsize(weapon_model_path) / (1024**2):.1f} MB")
        
        try:
            weapon_model = YOLO(weapon_model_path)
            print(f"✓ Model loaded successfully")
            weapon_classes = weapon_model.names
            print(f"  Classes: {list(weapon_classes.values())}")
            print(f"\n✓ Capability: Can detect weapons (if trained on weapon data)")
        except Exception as e:
            print(f"✗ Cannot load model: {e}")
            print("  System will fall back to generic detection")
    else:
        print(f"✗ File NOT found: {weapon_model_path}")
        print("\n  CONSEQUENCE:")
        print("  - System CANNOT detect weapons via ML")
        print("  - Falls back to string-based label mapping")
        print("  - Example: 'bottle' → 'handgun' (heuristic, not accurate)")
    
    # System behavior analysis
    print("\n[3] System Behavior Analysis:")
    print("-" * 70)
    
    if os.path.exists(weapon_model_path):
        print("✓ WITH Weapon Model:")
        print("  1. YOLOv8s detects generic objects")
        print("  2. Weapon model detects weapon-specific classes")
        print("  3. Confidence aggregation combines both")
        print("  4. Post-processing filters false positives")
        print("  → Result: Dual-engine detection")
    else:
        print("✗ WITHOUT Weapon Model:")
        print("  1. YOLOv8s detects generic objects (80 COCO classes)")
        print("  2. Label mapping converts to weapon categories")
        print("     - 'bottle' → 'handgun'? (heuristic guess)")
        print("     - 'umbrella' → 'rifle'? (heuristic guess)")
        print("     - 'kitchen_knife' → 'knife' (string match)")
        print("  3. Post-processing filters on confidence")
        print("  → Result: Generic detection + string matching (UNRELIABLE)")
    
    # Implications for paper/deployment
    print("\n[4] Implications:")
    print("-" * 70)
    
    print("\nFor Academic Paper:")
    print("  ❌ CANNOT claim: 'Custom-trained weapon detection'")
    print("  ✓ CAN claim: 'Post-processing pipeline for object detection'")
    print("  ⚠️  MUST disclose: External model sources and heuristic mapping")
    
    print("\nFor Deployment:")
    print("  ❌ NOT suitable for: Law enforcement, military, critical systems")
    print("  ✓ SUITABLE for: Research demo, academic evaluation")
    print("  ⚠️  REQUIRES: Security hardening, authentication, proper testing")
    
    # Recommendations
    print("\n[5] Recommendations:")
    print("-" * 70)
    
    if os.path.exists(weapon_model_path):
        print("\n✓ You have weapon model capability")
        print("  But still need to:")
        print("  1. Document where weapon_model.pt comes from")
        print("  2. Cite the source in your paper")
        print("  3. Test reliability on diverse scenarios")
        print("  4. Add graceful fallback if model corrupts")
    else:
        print("\n✗ Without weapon model, you have limitations:")
        print("  1. Get weapon_model.pt from Hugging Face or train your own")
        print("  2. Add error handling for model download failures")
        print("  3. Implement better label mapping (ML-based, not string matching)")
        print("  4. Test false positive rates extensively")
    
    print("\n[6] Model Sources Citation:")
    print("-" * 70)
    print("\nYou should cite these in your paper:")
    print("""
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
    """)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    check_model_setup()
