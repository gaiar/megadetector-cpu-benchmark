# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaDetector CPU Benchmark Suite — benchmarks MegaDetector v5 (wildlife detection model) inference performance on Intel CPUs without GPU. Two source files (`benchmark.py`, `bench_utils.py`) orchestrated via Docker.

## Build & Run Commands

```bash
# Build Docker image
docker-compose build

# Quick test (10 images, single batch)
docker-compose --profile quick up

# Full benchmark (multiple batch sizes)
docker-compose up

# Threading optimization test
docker-compose --profile threading up

# Model comparison (v5a vs v5b)
docker-compose --profile compare up

# Run benchmark directly (inside container or with deps installed)
python benchmark.py --mode full --batch-sizes 1 4 8 16 --num-images 50
python benchmark.py --mode quick
python benchmark.py --mode threading
python benchmark.py --help
```

There is no test suite, linter configuration, or pyproject.toml. Dependencies are in `requirements.txt`; PyTorch CPU-only wheels are installed separately in the Dockerfile.

## Architecture

**benchmark.py** — Main entry point with argparse CLI:
- `MegaDetectorBenchmark` class: loads model, runs inference in configurable batch sizes, collects latency percentiles (P50/P95/P99) and throughput (FPS)
- `CPUMonitor` class: background thread polling CPU% and memory every 100ms via psutil
- `BenchmarkResult` dataclass: 14 metrics per test run
- `_import_megadetector()`: lazy import — MegaDetector is only imported when actually running inference, not on `--help`. This was a critical fix for GitHub Actions compatibility.

**bench_utils.py** — Support functions:
- `get_system_info()`: system specs via psutil/cpuinfo
- `generate_synthetic_images()`: creates random test images (rectangles on colored backgrounds)
- `plot_benchmark_results()`: 6-subplot matplotlib visualization
- `compare_with_yolo()`: comparison report against YOLO baseline

**Data flow**: Docker builds environment → benchmark.py loads MegaDetector model (auto-downloaded `.pt` files) → processes images from `/workspace/data/test_images` → outputs JSON to `/workspace/results/` → optional plot generation.

## Intel CPU Optimization

The Docker environment sets 25+ environment variables for Intel MKL/OpenMP tuning (see `docker-compose.yml` and `Dockerfile`). Key PyTorch settings in `benchmark.py`:
- `torch.set_num_threads()` / `torch.set_num_interop_threads(2)`
- `torch.backends.mkl.enabled = True`
- `torch.set_flush_denormal(True)`
- Inference runs with `torch.set_grad_enabled(False)`

## CI/CD

- **docker-build.yml**: Triggers on push/PR to main. Builds Docker image, runs quick benchmark with synthetic images, tests Python imports across 3.9/3.10/3.11 matrix.
- **benchmark.yml**: Manual dispatch (`workflow_dispatch`) for full performance benchmarking with configurable batch sizes and image count. Uploads result artifacts.

## MegaDetector Internals (benchmarking context)

MegaDetector is a YOLOv5-based wildlife detector that identifies **animals**, **people**, and **vehicles** in camera trap images — it does not classify species. The `ultralytics-yolov5==0.1.1` dependency provides the inference primitives (NMS, letterboxing, coordinate scaling).

**Models**: v5a and v5b are trained on different data mixes; neither is strictly better — performance varies by ecosystem. Both use 1280px input size (long side). Model weights (`.pt` files) are auto-downloaded to `~/.megadetector/` on first use.

**Detection API** — what this benchmark calls:
```python
from megadetector.detection.pytorch_detector import PTDetector

detector = PTDetector(model_path='md_v5a.0.0.pt')
result = detector.generate_detections_one_image(
    img_original=pil_image,
    image_id='image.jpg',
    detection_threshold=0.2  # v5-specific; v4 used 0.8 — never mix
)
# Returns: {'file': str, 'detections': [{'category': '1', 'conf': float, 'bbox': [x,y,w,h]}], ...}
```
Bounding boxes are normalized `[x_min, y_min, width, height]`. Categories: `'1'`=animal, `'2'`=person, `'3'`=vehicle.

**Preprocessing pipeline** (performance-relevant): load image → EXIF rotation → resize to aspect ratio → letterbox pad (stride 64 for 1280-size models) → normalize to 0-1 float tensor → CHW transpose. This runs on CPU even when inference uses GPU.

**Why the lazy import matters**: `PTDetector.__init__` calls `_initialize_yolo_imports()` which imports from `ultralytics-yolov5`, which in turn imports PyTorch. If MegaDetector or its transitive deps aren't installed (as in CI `--help` checks), a top-level import crashes immediately.

## Key Constraints

- MegaDetector must be lazily imported (never at module level) to avoid ImportError when the package isn't installed (e.g., `--help` in CI).
- PyTorch must be installed as CPU-only from `download.pytorch.org` index, not from PyPI.
- Docker image uses multi-stage build and runs as non-root user `benchmark:1000`.
- The `ultralytics-yolov5==0.1.1` package is pinned — it's a transitive dependency of MegaDetector.
- Never name a Python file `utils.py` in the project root — it shadows YOLOv5's internal `utils` package and breaks `PTDetector` initialization. The utility file is named `bench_utils.py` for this reason.
- Docker models volume: the default service uses models baked into the image. To use local models, use `docker-compose --profile custom-model up`.
