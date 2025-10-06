#!/usr/bin/env python3
"""
MegaDetector CPU Performance Benchmark Suite
Optimized for Intel Xeon E5-1620 v2 with 64GB RAM
"""

import os
import sys
import json
import time
import argparse
import statistics
import psutil
import cpuinfo
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


def _import_megadetector():
    """Lazily import MegaDetector to avoid unnecessary dependencies for --help."""

    try:
        from megadetector.detection.pytorch_detector import PTDetector  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MegaDetector is required to run benchmarks. "
            "Install it with 'pip install git+https://github.com/agentmorris/MegaDetector.git'"
        ) from exc

    return PTDetector

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    batch_size: int
    num_images: int
    total_time: float
    fps: float
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    cpu_usage_mean: float
    memory_usage_peak: float
    memory_usage_mean: float
    num_threads: int
    model_version: str
    timestamp: str

class CPUMonitor:
    """Monitor CPU and memory usage during benchmark"""
    
    def __init__(self):
        self.cpu_percentages = []
        self.memory_usage = []
        self.monitoring = False
        
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.cpu_percentages = []
        self.memory_usage = []
        
        def monitor():
            while self.monitoring:
                self.cpu_percentages.append(psutil.cpu_percent(interval=0.1))
                self.memory_usage.append(psutil.virtual_memory().used / (1024**3))  # GB
                time.sleep(0.1)
        
        from threading import Thread
        self.thread = Thread(target=monitor)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        self.thread.join()
        
        return {
            'cpu_mean': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'cpu_max': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
            'memory_mean': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_peak': np.max(self.memory_usage) if self.memory_usage else 0
        }

class MegaDetectorBenchmark:
    """Benchmark MegaDetector performance on CPU"""
    
    def __init__(self, model_path: str, output_dir: str = "results"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.detector = None
        self._detector_cls = None
        self.results = []
        
        # Configure CPU optimization
        self._configure_cpu_optimization()
        
    def _configure_cpu_optimization(self):
        """Configure PyTorch for optimal CPU performance"""
        # Set thread counts
        num_threads = int(os.environ.get('OMP_NUM_THREADS', 8))
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(2)
        
        # Enable MKL optimizations if available
        if torch.backends.mkl.is_available():
            print(f"✓ Intel MKL enabled")
            torch.backends.mkl.enabled = True
        
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        # Set flush denormal for FP32 speedup
        torch.set_flush_denormal(True)
        
        print(f"✓ PyTorch configured: {num_threads} threads")
        print(f"✓ CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
        print(f"✓ RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
    def load_model(self):
        """Load MegaDetector model"""
        print(f"\n📦 Loading model: {self.model_path}")
        start = time.time()

        if self._detector_cls is None:
            self._detector_cls = _import_megadetector()

        self.detector = self._detector_cls(self.model_path)
        
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Warm up with dummy inference
        print("🔥 Warming up model...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detector.generate_detections_one_image(dummy_image)
        print("✓ Model ready")
        
        return load_time
    
    def prepare_test_images(self, image_dir: str, num_images: int = None) -> List[str]:
        """Prepare list of test images"""
        image_dir = Path(image_dir)
        
        # Support multiple image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(ext))
            image_files.extend(image_dir.glob(ext.upper()))
        
        if num_images:
            image_files = image_files[:num_images]
        
        print(f"📸 Found {len(image_files)} test images")
        return [str(f) for f in image_files]
    
    def benchmark_batch(self, image_files: List[str], batch_size: int = 1) -> BenchmarkResult:
        """Run benchmark with specified batch size"""
        print(f"\n🏃 Running benchmark: batch_size={batch_size}, images={len(image_files)}")
        
        # Monitor system resources
        monitor = CPUMonitor()
        monitor.start()
        
        # Process images in batches
        latencies = []
        num_batches = (len(image_files) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_start = time.time()
            
            # Process batch
            for image_file in batch_files:
                try:
                    # Load and process image
                    import cv2
                    image = cv2.imread(image_file)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        _ = self.detector.generate_detections_one_image(image)
                except Exception as e:
                    print(f"⚠️  Error processing {image_file}: {e}")
            
            batch_time = time.time() - batch_start
            latencies.append(batch_time / len(batch_files))
            
            # Progress indicator
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + batch_size}/{len(image_files)} images...")
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        monitor_stats = monitor.stop()
        
        # Calculate statistics
        fps = len(image_files) / total_time
        latencies_ms = [l * 1000 for l in latencies]  # Convert to milliseconds
        
        result = BenchmarkResult(
            batch_size=batch_size,
            num_images=len(image_files),
            total_time=total_time,
            fps=fps,
            latency_mean=np.mean(latencies_ms),
            latency_p50=np.percentile(latencies_ms, 50),
            latency_p95=np.percentile(latencies_ms, 95),
            latency_p99=np.percentile(latencies_ms, 99),
            latency_max=np.max(latencies_ms),
            cpu_usage_mean=monitor_stats['cpu_mean'],
            memory_usage_peak=monitor_stats['memory_peak'],
            memory_usage_mean=monitor_stats['memory_mean'],
            num_threads=torch.get_num_threads(),
            model_version=os.path.basename(self.model_path),
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        
        # Print summary
        print(f"\n📊 Results for batch_size={batch_size}:")
        print(f"  • FPS: {result.fps:.2f}")
        print(f"  • Latency (P50/P95/P99): {result.latency_p50:.1f}/{result.latency_p95:.1f}/{result.latency_p99:.1f} ms")
        print(f"  • CPU Usage: {result.cpu_usage_mean:.1f}%")
        print(f"  • Memory: {result.memory_usage_peak:.1f} GB peak")
        
        return result
    
    def run_full_benchmark(self, image_dir: str, batch_sizes: List[int] = None):
        """Run complete benchmark suite"""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        # Prepare test images
        image_files = self.prepare_test_images(image_dir)
        if not image_files:
            print("❌ No images found!")
            return
        
        # Load model
        load_time = self.load_model()
        
        print("\n" + "="*60)
        print("🚀 STARTING FULL BENCHMARK")
        print("="*60)
        
        # Test each batch size
        for batch_size in batch_sizes:
            if batch_size > len(image_files):
                print(f"⚠️  Skipping batch_size={batch_size} (only {len(image_files)} images)")
                continue
            self.benchmark_batch(image_files, batch_size)
            time.sleep(2)  # Cool down between tests
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def benchmark_threading(self, image_dir: str, thread_counts: List[int] = None):
        """Benchmark different threading configurations"""
        if thread_counts is None:
            thread_counts = [1, 2, 4, 6, 8]
        
        image_files = self.prepare_test_images(image_dir, num_images=100)
        if not image_files:
            print("❌ No images found!")
            return
        
        self.load_model()
        
        print("\n" + "="*60)
        print("🧵 THREADING BENCHMARK")
        print("="*60)
        
        original_threads = torch.get_num_threads()
        
        for num_threads in thread_counts:
            print(f"\n Testing with {num_threads} threads...")
            torch.set_num_threads(num_threads)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            
            result = self.benchmark_batch(image_files, batch_size=4)
            
        # Restore original settings
        torch.set_num_threads(original_threads)
        
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("📈 BENCHMARK SUMMARY")
        print("="*60)
        
        # Find best configuration
        best_fps = max(self.results, key=lambda r: r.fps)
        best_latency = min(self.results, key=lambda r: r.latency_p95)
        
        print(f"\n🏆 Best Throughput:")
        print(f"  • Batch Size: {best_fps.batch_size}")
        print(f"  • FPS: {best_fps.fps:.2f}")
        print(f"  • CPU Usage: {best_fps.cpu_usage_mean:.1f}%")
        
        print(f"\n⚡ Best Latency:")
        print(f"  • Batch Size: {best_latency.batch_size}")
        print(f"  • P95 Latency: {best_latency.latency_p95:.1f} ms")
        print(f"  • FPS: {best_latency.fps:.2f}")
        
        # Create comparison table
        print("\n📊 Full Results Table:")
        print(f"{'Batch':<8} {'FPS':<10} {'P50 (ms)':<10} {'P95 (ms)':<10} {'CPU %':<8} {'RAM (GB)':<10}")
        print("-" * 60)
        
        for r in sorted(self.results, key=lambda x: x.batch_size):
            print(f"{r.batch_size:<8} {r.fps:<10.2f} {r.latency_p50:<10.1f} "
                  f"{r.latency_p95:<10.1f} {r.cpu_usage_mean:<8.1f} {r.memory_usage_peak:<10.1f}")

def main():
    parser = argparse.ArgumentParser(description="MegaDetector CPU Benchmark")
    parser.add_argument("--model", default="/workspace/models/md_v5a.0.0.pt",
                        help="Path to MegaDetector model")
    parser.add_argument("--images", default="/workspace/data/test_images",
                        help="Directory containing test images")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32],
                        help="Batch sizes to test")
    parser.add_argument("--mode", choices=["full", "threading", "quick"],
                        default="full",
                        help="Benchmark mode")
    parser.add_argument("--num-images", type=int,
                        help="Limit number of test images")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = MegaDetectorBenchmark(args.model, args.output)
    
    # Run appropriate benchmark
    if args.mode == "quick":
        # Quick test with small subset
        images = benchmark.prepare_test_images(args.images, num_images=10)
        benchmark.load_model()
        benchmark.benchmark_batch(images, batch_size=1)
    elif args.mode == "threading":
        benchmark.benchmark_threading(args.images)
    else:
        benchmark.run_full_benchmark(args.images, args.batch_sizes)
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()