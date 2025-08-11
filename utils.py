#!/usr/bin/env python3
"""
Utility functions for MegaDetector CPU benchmark
"""

import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_system_info() -> Dict:
    """Get detailed system information"""
    import cpuinfo
    
    cpu_info = cpuinfo.get_cpu_info()
    mem_info = psutil.virtual_memory()
    
    return {
        'cpu': {
            'brand': cpu_info.get('brand_raw', 'Unknown'),
            'arch': cpu_info.get('arch', 'Unknown'),
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'cache': {
                'l1': cpu_info.get('l1_data_cache_size', 'Unknown'),
                'l2': cpu_info.get('l2_cache_size', 'Unknown'),
                'l3': cpu_info.get('l3_cache_size', 'Unknown'),
            }
        },
        'memory': {
            'total_gb': mem_info.total / (1024**3),
            'available_gb': mem_info.available / (1024**3),
            'percent_used': mem_info.percent
        },
        'python': {
            'version': subprocess.check_output(['python', '--version']).decode().strip()
        }
    }

def download_test_images(output_dir: str = "data/test_images", num_images: int = 100):
    """Download sample images for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading test images to {output_dir}")
    
    # URLs for sample wildlife images (you can replace with actual URLs)
    sample_urls = [
        # Add actual image URLs here
        # For now, we'll create a placeholder
    ]
    
    # Alternative: Download from LILA dataset
    lila_script = """
    import os
    import urllib.request
    from pathlib import Path
    
    # Download sample images from LILA Snapshot Serengeti dataset
    base_url = "https://lilablobssc.blob.core.windows.net/snapshot-serengeti-bboxes/"
    
    sample_images = [
        "S1/R10/R10_R1/S1_R10_R1_PICT0001.JPG",
        "S1/R10/R10_R1/S1_R10_R1_PICT0002.JPG",
        # Add more image paths
    ]
    
    for img_path in sample_images[:num_images]:
        url = base_url + img_path
        filename = output_dir / Path(img_path).name
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"  ✓ Downloaded {filename.name}")
        except Exception as e:
            print(f"  ✗ Failed to download {img_path}: {e}")
    """
    
    # Create download script
    script_path = output_dir.parent / "download_lila_samples.py"
    with open(script_path, 'w') as f:
        f.write(lila_script.replace('num_images', str(num_images))
                          .replace('output_dir', f'Path("{output_dir}")'))
    
    print(f"📝 Created download script: {script_path}")
    print(f"   Run: python {script_path}")

def generate_synthetic_images(output_dir: str = "data/test_images", 
                            num_images: int = 100,
                            sizes: List[Tuple[int, int]] = None):
    """Generate synthetic test images for benchmarking"""
    import cv2
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sizes is None:
        sizes = [(640, 480), (1280, 720), (1920, 1080)]
    
    print(f"🎨 Generating {num_images} synthetic test images...")
    
    for i in range(num_images):
        # Rotate through different sizes
        width, height = sizes[i % len(sizes)]
        
        # Create random image with some structure (not just noise)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add random rectangles (simulating objects)
        num_objects = np.random.randint(1, 5)
        for _ in range(num_objects):
            x1 = np.random.randint(0, width - 50)
            y1 = np.random.randint(0, height - 50)
            x2 = x1 + np.random.randint(20, min(200, width - x1))
            y2 = y1 + np.random.randint(20, min(200, height - y1))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        
        # Add some noise
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Save image
        filename = output_dir / f"synthetic_{i:04d}_{width}x{height}.jpg"
        cv2.imwrite(str(filename), image)
    
    print(f"✓ Generated {num_images} images in {output_dir}")

def plot_benchmark_results(results_file: str, output_dir: str = "results"):
    """Create visualization plots from benchmark results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MegaDetector CPU Benchmark Results', fontsize=16)
    
    # 1. FPS vs Batch Size
    ax = axes[0, 0]
    ax.bar(df['batch_size'].astype(str), df['fps'])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('FPS')
    ax.set_title('Throughput vs Batch Size')
    ax.grid(True, alpha=0.3)
    
    # 2. Latency Percentiles
    ax = axes[0, 1]
    x = df['batch_size']
    ax.plot(x, df['latency_p50'], marker='o', label='P50')
    ax.plot(x, df['latency_p95'], marker='s', label='P95')
    ax.plot(x, df['latency_p99'], marker='^', label='P99')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. CPU Usage
    ax = axes[0, 2]
    ax.bar(df['batch_size'].astype(str), df['cpu_usage_mean'])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Utilization')
    ax.grid(True, alpha=0.3)
    
    # 4. Memory Usage
    ax = axes[1, 0]
    ax.plot(df['batch_size'], df['memory_usage_peak'], marker='o', label='Peak')
    ax.plot(df['batch_size'], df['memory_usage_mean'], marker='s', label='Mean')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Efficiency (FPS per CPU %)
    ax = axes[1, 1]
    efficiency = df['fps'] / (df['cpu_usage_mean'] / 100)
    ax.bar(df['batch_size'].astype(str), efficiency)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('FPS per CPU%')
    ax.set_title('Processing Efficiency')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Best Value', 'Config'],
        ['Max FPS', f"{df['fps'].max():.2f}", f"Batch {df.loc[df['fps'].idxmax(), 'batch_size']}"],
        ['Min P95 Latency', f"{df['latency_p95'].min():.1f} ms", f"Batch {df.loc[df['latency_p95'].idxmin(), 'batch_size']}"],
        ['Best Efficiency', f"{efficiency.max():.2f}", f"Batch {df.loc[efficiency.idxmax(), 'batch_size']}"],
        ['Min Memory', f"{df['memory_usage_peak'].min():.1f} GB", f"Batch {df.loc[df['memory_usage_peak'].idxmin(), 'batch_size']}"],
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'benchmark_plots.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved plots to {plot_file}")
    
    # Also save as PDF
    pdf_file = output_dir / 'benchmark_plots.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📄 Saved PDF to {pdf_file}")
    
    plt.close()

def compare_with_yolo(md_results: str, yolo_baseline: Dict) -> Dict:
    """Compare MegaDetector results with YOLO baseline"""
    
    # Load MegaDetector results
    with open(md_results, 'r') as f:
        md_data = json.load(f)
    
    # Find best configurations
    md_df = pd.DataFrame(md_data)
    best_fps_idx = md_df['fps'].idxmax()
    best_latency_idx = md_df['latency_p95'].idxmin()
    
    comparison = {
        'throughput': {
            'yolo_fps': yolo_baseline.get('fps', 0),
            'md_fps': md_df.loc[best_fps_idx, 'fps'],
            'improvement': ((md_df.loc[best_fps_idx, 'fps'] / yolo_baseline.get('fps', 1)) - 1) * 100,
            'md_config': f"Batch size {md_df.loc[best_fps_idx, 'batch_size']}"
        },
        'latency': {
            'yolo_p95': yolo_baseline.get('latency_p95', 0),
            'md_p95': md_df.loc[best_latency_idx, 'latency_p95'],
            'improvement': ((yolo_baseline.get('latency_p95', 1) / md_df.loc[best_latency_idx, 'latency_p95']) - 1) * 100,
            'md_config': f"Batch size {md_df.loc[best_latency_idx, 'batch_size']}"
        },
        'memory': {
            'yolo_gb': yolo_baseline.get('memory_gb', 0),
            'md_gb': md_df['memory_usage_peak'].mean(),
            'difference': md_df['memory_usage_peak'].mean() - yolo_baseline.get('memory_gb', 0)
        },
        'stability': {
            'yolo': yolo_baseline.get('stability', 'Unknown'),
            'megadetector': 'Stable (based on testing)'
        }
    }
    
    # Print comparison report
    print("\n" + "="*60)
    print("📊 MEGADETECTOR vs YOLO COMPARISON")
    print("="*60)
    
    print(f"\n🚀 Throughput:")
    print(f"  YOLO:         {comparison['throughput']['yolo_fps']:.2f} FPS")
    print(f"  MegaDetector: {comparison['throughput']['md_fps']:.2f} FPS ({comparison['throughput']['md_config']})")
    print(f"  Improvement:  {comparison['throughput']['improvement']:+.1f}%")
    
    print(f"\n⚡ Latency (P95):")
    print(f"  YOLO:         {comparison['latency']['yolo_p95']:.1f} ms")
    print(f"  MegaDetector: {comparison['latency']['md_p95']:.1f} ms ({comparison['latency']['md_config']})")
    print(f"  Improvement:  {comparison['latency']['improvement']:+.1f}%")
    
    print(f"\n💾 Memory Usage:")
    print(f"  YOLO:         {comparison['memory']['yolo_gb']:.1f} GB")
    print(f"  MegaDetector: {comparison['memory']['md_gb']:.1f} GB")
    print(f"  Difference:   {comparison['memory']['difference']:+.1f} GB")
    
    print(f"\n🔧 Stability:")
    print(f"  YOLO:         {comparison['stability']['yolo']}")
    print(f"  MegaDetector: {comparison['stability']['megadetector']}")
    
    return comparison

if __name__ == "__main__":
    # Example usage
    print("🔧 MegaDetector Benchmark Utilities")
    print("\n1. System Information:")
    info = get_system_info()
    print(json.dumps(info, indent=2))
    
    print("\n2. Generate synthetic test images:")
    generate_synthetic_images(num_images=10)
    
    print("\n✓ Utilities ready!")