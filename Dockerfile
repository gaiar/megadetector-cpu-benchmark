# Multi-stage build for optimized CPU-only MegaDetector
# Optimized for Intel Xeon E5-1620 v2 with 64GB RAM

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install CPU-optimized PyTorch with Intel MKL
# Using CPU-only wheel to reduce size and avoid GPU dependencies
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip install \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    tqdm==4.66.1 \
    humanfriendly==10.0 \
    jsonpickle==3.0.2 \
    matplotlib==3.8.2 \
    pandas==2.1.3 \
    psutil==5.9.6 \
    py-cpuinfo==9.0.0

# Install YOLOv5 dependencies (for MegaDetector)
RUN pip install \
    ultralytics-yolov5==0.1.1 \
    PyYAML>=5.3.1 \
    scipy>=1.10.0 \
    seaborn>=0.11.0

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies and Intel MKL
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set Intel MKL and OpenMP environment variables for optimal CPU performance
ENV MKL_NUM_THREADS=8 \
    OMP_NUM_THREADS=8 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    KMP_BLOCKTIME=0 \
    MKL_DYNAMIC=FALSE \
    OMP_PROC_BIND=TRUE \
    OMP_PLACES=cores

# Disable GPU/CUDA
ENV CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU=1

# Set PyTorch threading for CPU
ENV TORCH_NUM_THREADS=8 \
    TORCH_NUM_INTEROP_THREADS=2

# Create working directory
WORKDIR /workspace

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data /workspace/results

# Download MegaDetector model (v5a by default)
# You can override this with a volume mount
RUN wget -q https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt \
    -O /workspace/models/md_v5a.0.0.pt && \
    wget -q https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt \
    -O /workspace/models/md_v5b.0.0.pt

# Copy benchmark scripts
COPY benchmark.py /workspace/
COPY utils.py /workspace/
COPY requirements.txt /workspace/

# Install MegaDetector package
RUN pip install git+https://github.com/agentmorris/MegaDetector.git

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import torch; import cv2; print('Health check passed')" || exit 1

# Set up non-root user for security
RUN useradd -m -u 1000 benchmark && \
    chown -R benchmark:benchmark /workspace
USER benchmark

# Default command runs benchmark
ENTRYPOINT ["python", "benchmark.py"]
CMD ["--help"]