#!/bin/bash

# Fix SSL Certificate Issues for Python on macOS
# This script helps resolve SSL certificate verification errors

echo "Fixing SSL Certificate Issues for Python..."
echo ""

# Find Python installation
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_PATH=$(which python3)

echo "Python version: $PYTHON_VERSION"
echo "Python path: $PYTHON_PATH"
echo ""

# Try to find and run Install Certificates command
CERT_SCRIPT="/Applications/Python $PYTHON_VERSION/Install Certificates.command"

if [ -f "$CERT_SCRIPT" ]; then
    echo "Found certificate installer: $CERT_SCRIPT"
    echo "Running certificate installer..."
    bash "$CERT_SCRIPT"
else
    echo "Certificate installer not found at: $CERT_SCRIPT"
    echo ""
    echo "Alternative solutions:"
    echo ""
    echo "1. Install certificates manually:"
    echo "   - Open Finder"
    echo "   - Go to Applications"
    echo "   - Find 'Python $PYTHON_VERSION' folder"
    echo "   - Run 'Install Certificates.command'"
    echo ""
    echo "2. Or manually download ResNet50 model:"
    echo "   mkdir -p ~/.cache/torch/hub/checkpoints"
    echo "   curl -L -o ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth \\"
    echo "     https://download.pytorch.org/models/resnet50-0676ba61.pth"
    echo ""
    echo "3. Or use pip to install certificates:"
    echo "   pip install --upgrade certifi"
    echo "   /Applications/Python\\ $PYTHON_VERSION/Install\\ Certificates.command"
fi

echo ""
echo "Done!"

