#!/bin/bash

# Script to initialize and push to GitHub repository
# Usage: ./init-github-repo.sh

set -e

REPO_NAME="megadetector-cpu-benchmark"
GITHUB_USER="gaiar"

echo "================================================"
echo "   GitHub Repository Setup"
echo "   Repository: $GITHUB_USER/$REPO_NAME"
echo "================================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "→ Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Use README-GITHUB.md as the main README
if [ -f "README-GITHUB.md" ]; then
    echo "→ Setting up README..."
    mv README.md README-LOCAL.md 2>/dev/null || true
    cp README-GITHUB.md README.md
    echo "✓ README configured for GitHub"
fi

# Add all files
echo "→ Adding files to git..."
git add .
echo "✓ Files added"

# Create initial commit
echo "→ Creating initial commit..."
git commit -m "Initial commit: MegaDetector CPU Benchmark Suite

- CPU-optimized Docker container for Intel Xeon
- Comprehensive benchmarking scripts
- Support for 64GB RAM configurations
- Multiple benchmark modes (quick, full, threading)
- Automated performance visualization
- Production-ready deployment configs" || echo "✓ Already committed"

# Set main branch
git branch -M main

# Instructions for creating GitHub repo
echo ""
echo "================================================"
echo "   Create GitHub Repository"
echo "================================================"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Create repository with these settings:"
echo "   • Repository name: $REPO_NAME"
echo "   • Description: High-performance CPU benchmarking suite for MegaDetector wildlife detection"
echo "   • Public repository"
echo "   • DO NOT initialize with README, .gitignore, or license"
echo ""
echo "3. After creating, run these commands:"
echo ""
echo "git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "git push -u origin main"
echo ""
echo "================================================"
echo "   Alternative: Using GitHub CLI"
echo "================================================"
echo ""
echo "If you have GitHub CLI installed (gh), run:"
echo ""
echo "gh repo create $REPO_NAME --public --description \"High-performance CPU benchmarking suite for MegaDetector wildlife detection\" --source=. --remote=origin --push"
echo ""
echo "================================================"
echo "   Repository Structure"
echo "================================================"
echo ""
tree -L 2 -I '__pycache__|*.pyc|data/test_images' 2>/dev/null || find . -maxdepth 2 -type f -not -path "./.git/*" -not -path "./data/test_images/*" | sort

echo ""
echo "================================================"
echo "   Ready to Push!"
echo "================================================"
echo ""
echo "Your repository is ready. Follow the instructions above to push to GitHub."
echo ""
echo "After pushing, you can:"
echo "• Enable GitHub Actions for CI/CD"
echo "• Add topics: docker, benchmark, wildlife, conservation, cpu-optimization"
echo "• Create releases with docker images"
echo "• Add GitHub Pages for documentation"
echo ""