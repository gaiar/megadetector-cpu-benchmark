#!/bin/bash

# MegaDetector CPU Benchmark - Quick Setup Script
# This script automates the initial setup process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Header
echo "================================================"
echo "   MegaDetector CPU Benchmark Setup"
echo "================================================"
echo ""

# Check Docker installation
print_info "Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker found: $DOCKER_VERSION"
else
    print_error "Docker not found! Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
print_info "Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_success "Docker Compose found: $COMPOSE_VERSION"
else
    print_error "Docker Compose not found! Please install Docker Compose."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p data/test_images models results
print_success "Directories created"

# Create .gitkeep files
touch data/.gitkeep
touch models/.gitkeep
touch results/.gitkeep

# Download test images
print_info "Downloading test images..."
if [ -f "download_test_images.sh" ]; then
    chmod +x download_test_images.sh
    ./download_test_images.sh
    print_success "Test images downloaded"
else
    print_error "download_test_images.sh not found"
fi

# Build Docker image
print_info "Building Docker image (this may take several minutes)..."
if docker-compose build; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# System information
echo ""
echo "================================================"
echo "   System Information"
echo "================================================"
echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
echo "Memory: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))GB")"
echo ""

# Test run
echo "================================================"
echo "   Running Quick Test"
echo "================================================"
print_info "Running quick benchmark test..."

if docker-compose --profile quick up --exit-code-from benchmark-quick; then
    print_success "Quick test completed successfully!"
    
    # Check for results
    if ls results/benchmark_results_*.json 1> /dev/null 2>&1; then
        echo ""
        print_success "Results saved in results/ directory"
        
        # Display summary
        LATEST_RESULT=$(ls -t results/benchmark_results_*.json | head -1)
        if [ -f "$LATEST_RESULT" ]; then
            echo ""
            echo "Summary:"
            python3 -c "
import json
with open('$LATEST_RESULT', 'r') as f:
    data = json.load(f)
    if data:
        r = data[0]
        print(f'  • FPS: {r[\"fps\"]:.2f}')
        print(f'  • Latency P95: {r[\"latency_p95\"]:.1f}ms')
        print(f'  • CPU Usage: {r[\"cpu_usage_mean\"]:.1f}%')
        print(f'  • Memory: {r[\"memory_usage_peak\"]:.1f}GB')
" 2>/dev/null || echo "  Could not parse results"
        fi
    fi
else
    print_error "Quick test failed"
fi

# Next steps
echo ""
echo "================================================"
echo "   Setup Complete! 🎉"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Run full benchmark:    docker-compose up"
echo "2. View results:          ls results/"
echo "3. Compare with YOLO:     python3 bench_utils.py"
echo "4. Read documentation:    cat README.md"
echo ""
echo "For help: https://github.com/gaiar/megadetector-cpu-benchmark"
echo ""