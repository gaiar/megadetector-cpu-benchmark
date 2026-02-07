#!/bin/bash

# Download test images for MegaDetector benchmark
# This script downloads sample wildlife images from various sources

set -e

echo "📥 MegaDetector Test Image Downloader"
echo "======================================"

# Create directories
DATA_DIR="data/test_images"
mkdir -p "$DATA_DIR"
mkdir -p "models"

# Function to download with progress
download_with_progress() {
    local url=$1
    local output=$2
    echo "  Downloading: $(basename $output)"
    wget -q --show-progress -O "$output" "$url" || curl -L -o "$output" "$url"
}

echo ""
echo "1️⃣ Downloading sample wildlife images..."
echo "----------------------------------------"

# Download sample images from MegaDetector repo
SAMPLE_IMAGES=(
    "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/images/orinoquia-thumb-web.jpg"
    "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/images/nacti.jpg"
    "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/images/pheasant_web.jpg"
    "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/images/idaho-camera-traps.jpg"
    "https://raw.githubusercontent.com/agentmorris/MegaDetector/main/images/channel-islands-thumb.jpg"
)

for url in "${SAMPLE_IMAGES[@]}"; do
    filename=$(basename "$url")
    download_with_progress "$url" "$DATA_DIR/$filename"
done

echo ""
echo "2️⃣ Generating synthetic test images..."
echo "----------------------------------------"

# Create Python script to generate additional test images
cat > generate_test_images.py << 'EOF'
import sys
sys.path.append('.')
from bench_utils import generate_synthetic_images

# Generate images of different sizes for comprehensive testing
sizes = [
    (640, 480),    # VGA
    (1280, 720),   # HD
    (1920, 1080),  # Full HD
    (2560, 1440),  # 2K
    (3840, 2160),  # 4K
]

print("Generating synthetic test images...")
generate_synthetic_images(
    output_dir="data/test_images",
    num_images=50,
    sizes=sizes
)
EOF

python generate_test_images.py
rm generate_test_images.py

echo ""
echo "3️⃣ Downloading LILA dataset samples (optional)..."
echo "------------------------------------------------"
echo "For more comprehensive testing, you can download images from LILA datasets:"
echo ""
echo "Option A: Download Snapshot Serengeti samples"
echo "  wget -r -l 1 -nd -np -A 'S1_*.JPG' https://lilablobssc.blob.core.windows.net/snapshot-serengeti-bboxes/S1/R10/R10_R1/ -P data/test_images/"
echo ""
echo "Option B: Use the LILA downloader script"
echo "  pip install lila-tools"
echo "  python -c \"from lila_tools import download_dataset; download_dataset('Snapshot Serengeti', max_images=100)\""
echo ""

# Count images
IMAGE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)

echo ""
echo "4️⃣ Model files..."
echo "----------------------------------------"
echo "Models will be downloaded automatically by Docker build."
echo "To use custom models, place them in the 'models' directory."
echo ""

echo "✅ Setup Complete!"
echo "=================="
echo "📸 Total test images: $IMAGE_COUNT"
echo "📁 Images location: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Build Docker image: docker-compose build"
echo "2. Run quick test: docker-compose run benchmark-quick"
echo "3. Run full benchmark: docker-compose up"
echo ""