#!/bin/bash

# Manually download ResNet50 model to avoid SSL certificate issues

echo "Downloading ResNet50 model manually..."

# Create cache directory
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

# Model URL and filename
MODEL_URL="https://download.pytorch.org/models/resnet50-0676ba61.pth"
MODEL_FILE="$CACHE_DIR/resnet50-0676ba61.pth"

# Check if already exists
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at: $MODEL_FILE"
    echo "Skipping download."
    exit 0
fi

# Download using curl (bypasses Python SSL issues)
echo "Downloading from: $MODEL_URL"
echo "Saving to: $MODEL_FILE"

curl -L -k -o "$MODEL_FILE" "$MODEL_URL"

if [ $? -eq 0 ] && [ -f "$MODEL_FILE" ]; then
    echo "✓ Download successful!"
    echo "Model saved to: $MODEL_FILE"
    echo "File size: $(du -h "$MODEL_FILE" | cut -f1)"
else
    echo "✗ Download failed!"
    echo "Please try:"
    echo "1. Check your internet connection"
    echo "2. Download manually from: $MODEL_URL"
    echo "3. Save to: $MODEL_FILE"
    exit 1
fi

