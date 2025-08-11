# MegaDetector CPU Performance Benchmark Suite

## System Specifications
- **CPU**: Intel Xeon E5-1620 v2 @ 3.70GHz (Ivy Bridge-EP)
- **Cores**: 4 physical cores, 8 threads (HyperThreading)
- **RAM**: 64 GB (Excellent for large batch processing)
- **Cache**: 10 MB L3, 1 MB L2, 128 KiB L1
- **Architecture**: x86_64
- **Target**: CPU-only inference optimization to replace flaky YOLO setup

## Project Goals
1. Create standalone Docker-based benchmark suite for MegaDetector
2. Optimize for Intel Xeon CPU with high memory availability
3. Compare performance against current YOLO setup
4. Provide production-ready deployment configuration

## Optimization Strategy

### 1. Memory Optimization (64GB Available)
- **Large Batch Processing**: Test batch sizes up to 64 images
- **Prefetching**: Implement aggressive image prefetching
- **Memory Pooling**: Pre-allocate memory for better performance
- **Parallel Data Loading**: Use multiple workers for I/O
- **In-Memory Caching**: Cache preprocessed images

### 2. Intel CPU Optimizations
```bash
# Environment variables for Intel MKL optimization
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export MKL_DYNAMIC=FALSE
```

### 3. PyTorch CPU Optimizations
- Use PyTorch with Intel MKL-DNN backend
- Configure optimal thread settings:
  - inter_op_parallelism_threads=2
  - intra_op_parallelism_threads=8
- Enable torch.jit.script() for model optimization
- Use torch.set_flush_denormal(True) for FP32 speedup

### 4. Docker Optimizations
- Multi-stage build for minimal image size
- CPU-specific PyTorch wheel
- Remove all GPU dependencies
- Use python:3.11-slim base image
- Configure resource limits properly

## Benchmark Test Matrix

### Test 1: Throughput vs Batch Size
| Batch Size | Expected FPS | Memory Usage |
|------------|--------------|--------------|
| 1          | Baseline     | ~2 GB        |
| 4          | 3-4x         | ~4 GB        |
| 8          | 6-7x         | ~6 GB        |
| 16         | 10-12x       | ~10 GB       |
| 32         | 15-18x       | ~16 GB       |
| 64         | 20-25x       | ~24 GB       |

### Test 2: Threading Configuration
- Test OMP_NUM_THREADS: 1, 2, 4, 6, 8
- Test different PyTorch thread configurations
- Measure CPU utilization per configuration

### Test 3: Image Resolution Impact
- 640x480 (Low)
- 1280x720 (HD)
- 1920x1080 (Full HD)
- 3840x2160 (4K)

### Test 4: Model Variants
- MegaDetector v5a (YOLOv5x6)
- MegaDetector v5b (YOLOv5x6)
- Compare accuracy vs speed tradeoffs

### Test 5: Long-Running Stability
- 1-hour continuous processing
- Monitor for memory leaks
- Check thermal throttling
- Measure performance degradation

## Performance Metrics

### Primary Metrics
1. **Throughput (FPS)**: Images processed per second
2. **Latency**: 
   - P50 (median)
   - P95 
   - P99
   - Max
3. **CPU Utilization**: Per-core and overall
4. **Memory Usage**: 
   - Peak
   - Average
   - Per-image overhead

### Secondary Metrics
1. **Model Load Time**: Cold start performance
2. **First Inference Time**: JIT compilation overhead
3. **Cache Efficiency**: L1/L2/L3 hit rates
4. **Power Consumption**: If measurable
5. **Detection Quality**: mAP comparison with GPU

## Benchmark Scenarios

### Scenario A: Real-time Processing
- Single image stream
- Target: < 200ms latency
- Measure sustainable FPS

### Scenario B: Batch Processing
- Large dataset processing
- Maximize throughput
- Optimal batch size discovery

### Scenario C: Multi-Stream
- Simulate multiple camera feeds
- Test concurrent processing
- Resource allocation strategy

### Scenario D: Mixed Workload
- Various image sizes
- Different batch sizes
- Realistic production scenario

## Comparison with YOLO Baseline
| Metric | Current YOLO | Target MegaDetector | Improvement |
|--------|--------------|---------------------|-------------|
| FPS | TBD | > 5 | > 2x |
| Latency P95 | TBD | < 300ms | -50% |
| Memory | TBD | < 8GB | Similar |
| Stability | Flaky | Stable | Significant |
| Accuracy | Baseline | Equal or better | ≥ 0% |

## Docker Resource Configuration
```yaml
# Optimal Docker resource limits
resources:
  limits:
    cpus: '8'
    memory: 32G  # Leave headroom for OS
  reservations:
    cpus: '4'
    memory: 16G
```

## Expected Deliverables

1. **Docker Image**: Production-ready, CPU-optimized
2. **Benchmark Script**: Comprehensive testing suite
3. **Performance Report**: Detailed metrics and analysis
4. **Deployment Guide**: Production configuration
5. **Comparison Report**: vs current YOLO setup
6. **Optimization Guide**: Tuning recommendations

## Success Criteria

- [ ] **Performance**: ≥ 5 FPS on average
- [ ] **Latency**: P95 < 300ms
- [ ] **Memory**: Peak < 32GB
- [ ] **Stability**: 24-hour run without crashes
- [ ] **Accuracy**: mAP within 2% of GPU baseline
- [ ] **Docker Size**: < 2GB image
- [ ] **Startup Time**: < 30 seconds
- [ ] **CPU Efficiency**: > 70% utilization

## Risk Mitigation

1. **Thermal Throttling**: Monitor CPU temperature
2. **Memory Leaks**: Implement proper cleanup
3. **Thread Contention**: Careful thread pool sizing
4. **I/O Bottleneck**: Use SSD, implement caching
5. **Model Compatibility**: Test multiple MD versions

## Next Steps

1. Build Docker container with optimizations
2. Implement comprehensive benchmark script
3. Download test dataset
4. Run benchmark matrix
5. Generate performance report
6. Compare with YOLO baseline
7. Optimize based on findings
8. Create production deployment