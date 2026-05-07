#!/usr/bin/env python3
"""
benchmark.py -- Performance Benchmarking for Detection Pipeline

This script measures:
- Inference time per image (ms)
- Throughput (FPS)
- Memory usage (MB)
- System specifications for reproducibility

Usage:
    python scripts/benchmark.py

Requirements:
    - ultralytics>=8.1.0
    - opencv-python>=4.9.0
    - numpy>=1.24.0
    - psutil>=5.9.0 (for memory monitoring)

Output:
    - Prints summary to stdout
    - Saves detailed results to benchmark_results.csv
    - Logs system information for reproducibility

⚠️  LIMITATIONS:
    - Results are test-set only; generalization unknown
    - Includes all post-processing overhead
    - Single-threaded execution (not representative of production)
    - Does not measure model loading time
"""

import os
import sys
import time
import csv
import platform
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install ultralytics opencv-python")
    sys.exit(1)


def get_system_info() -> dict:
    """Collect system hardware and software information."""
    return {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': check_cuda_availability()
    }


def check_cuda_availability() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_test_images(test_dir: str = "dataset/test/images", max_images: int = 100) -> list:
    """
    Load test images from directory.
    
    Args:
        test_dir: Directory containing test images
        max_images: Maximum number of images to load
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory not found: {test_dir}")
        print("Creating synthetic test images...")
        return generate_synthetic_images(max_images)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    return sorted(images)[:max_images]


def generate_synthetic_images(count: int = 100) -> list:
    """
    Generate synthetic test images for benchmarking.
    
    Args:
        count: Number of synthetic images to create
        
    Returns:
        List of temporary image paths
    """
    temp_dir = "benchmark_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    paths = []
    for i in range(count):
        # Create random image (640x480, BGR)
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Add some shapes for realism
        cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)
        
        path = os.path.join(temp_dir, f"synthetic_{i:04d}.jpg")
        cv2.imwrite(path, image)
        paths.append(path)
    
    return paths


def benchmark_model(model_path: str = "yolov8s.pt", test_images: list = None) -> dict:
    """
    Benchmark model performance on test images.
    
    Args:
        model_path: Path to YOLO model file
        test_images: List of image paths to benchmark
        
    Returns:
        Dictionary with performance metrics
    """
    if test_images is None:
        test_images = load_test_images()
    
    if not test_images:
        print("Error: No test images available")
        return None
    
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Warmup inference (first run often slower due to initialization)
    print("Warming up model...")
    try:
        _ = model(test_images[0], verbose=False)
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    # Benchmark inference
    print(f"Benchmarking on {len(test_images)} images...")
    inference_times = []
    memory_usage = []
    
    process = psutil.Process()
    
    for i, image_path in enumerate(test_images):
        # Measure memory before inference
        mem_before = process.memory_info().rss / (1024**2)  # MB
        
        # Measure inference time
        start_time = time.perf_counter()
        try:
            results = model(image_path, verbose=False)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Measure memory after inference
        mem_after = process.memory_info().rss / (1024**2)  # MB
        
        inference_times.append(inference_time)
        memory_usage.append(mem_after)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_images)} images...")
    
    if not inference_times:
        print("Error: No successful inferences")
        return None
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    fps = 1000.0 / inference_times.mean()
    
    results = {
        'model': model_path,
        'test_image_count': len(test_images),
        'successful_inferences': len(inference_times),
        'mean_inference_time_ms': float(inference_times.mean()),
        'median_inference_time_ms': float(np.median(inference_times)),
        'std_inference_time_ms': float(inference_times.std()),
        'min_inference_time_ms': float(inference_times.min()),
        'max_inference_time_ms': float(inference_times.max()),
        'fps': float(fps),
        'mean_memory_usage_mb': float(np.mean(memory_usage)),
        'peak_memory_usage_mb': float(np.max(memory_usage))
    }
    
    return results


def format_results(system_info: dict, benchmark_results: dict) -> str:
    """Format benchmark results for display."""
    lines = [
        "\n" + "=" * 70,
        "BENCHMARK RESULTS",
        "=" * 70,
        "",
        "SYSTEM INFORMATION:",
        f"  Timestamp: {system_info['timestamp']}",
        f"  Platform: {system_info['platform']}",
        f"  Python: {system_info['python_version']}",
        f"  CPUs: {system_info['cpu_count']}",
        f"  Total Memory: {system_info['total_memory_gb']:.1f} GB",
        f"  CUDA Available: {system_info['cuda_available']}",
        "",
        "PERFORMANCE METRICS:",
        f"  Model: {benchmark_results['model']}",
        f"  Test Images: {benchmark_results['test_image_count']}",
        f"  Successful: {benchmark_results['successful_inferences']}",
        "",
        "INFERENCE TIME (ms):",
        f"  Mean: {benchmark_results['mean_inference_time_ms']:.2f}",
        f"  Median: {benchmark_results['median_inference_time_ms']:.2f}",
        f"  Std Dev: {benchmark_results['std_inference_time_ms']:.2f}",
        f"  Min: {benchmark_results['min_inference_time_ms']:.2f}",
        f"  Max: {benchmark_results['max_inference_time_ms']:.2f}",
        "",
        "THROUGHPUT:",
        f"  FPS: {benchmark_results['fps']:.2f}",
        "",
        "MEMORY USAGE:",
        f"  Mean: {benchmark_results['mean_memory_usage_mb']:.1f} MB",
        f"  Peak: {benchmark_results['peak_memory_usage_mb']:.1f} MB",
        "",
        "=" * 70,
        "⚠️  IMPORTANT NOTES:",
        "- Results include all post-processing overhead",
        "- Single-threaded execution; not representative of concurrent use",
        "- Warm-up inference excluded from statistics",
        "- Results are test-set only; generalization unknown",
        "- GPU performance depends on available VRAM",
        "=" * 70,
        ""
    ]
    return "\n".join(lines)


def save_results_csv(system_info: dict, benchmark_results: dict, output_file: str = "benchmark_results.csv"):
    """Save benchmark results to CSV."""
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp',
            'platform',
            'python_version',
            'cpu_count',
            'cuda_available',
            'model',
            'mean_inference_time_ms',
            'fps',
            'mean_memory_usage_mb'
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': system_info['timestamp'],
            'platform': system_info['platform'],
            'python_version': system_info['python_version'],
            'cpu_count': system_info['cpu_count'],
            'cuda_available': system_info['cuda_available'],
            'model': benchmark_results['model'],
            'mean_inference_time_ms': f"{benchmark_results['mean_inference_time_ms']:.2f}",
            'fps': f"{benchmark_results['fps']:.2f}",
            'mean_memory_usage_mb': f"{benchmark_results['mean_memory_usage_mb']:.1f}"
        })
    
    print(f"Results saved to: {output_file}")


def main():
    """Run full benchmarking suite."""
    print("Initializing benchmark...")
    
    # Get system information
    system_info = get_system_info()
    print(f"System: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    
    # Load test images
    test_images = load_test_images()
    if not test_images:
        print("Error: Could not load or generate test images")
        return
    
    print(f"Loaded {len(test_images)} test images")
    
    # Run benchmark
    benchmark_results = benchmark_model(test_images=test_images)
    if not benchmark_results:
        print("Error: Benchmark failed")
        return
    
    # Display results
    formatted = format_results(system_info, benchmark_results)
    print(formatted)
    
    # Save results
    save_results_csv(system_info, benchmark_results)


if __name__ == "__main__":
    main()
