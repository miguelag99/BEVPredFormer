#!/bin/bash
# Create checkpoints directory if it doesn't exist
mkdir -p ./checkpoints

# Check user argument
if [ "$#" -eq 0 ]; then
    echo "Usage: ./download_weights.sh [resnet50|resnet101|efficientnet]"
    exit 1
fi

case "$1" in
    "resnet50")
        echo "Checking ResNet50 weights..."
        if [ -f "checkpoints/resnet50-0676ba61.pth" ]; then
            echo "ResNet50 weights already exist, skipping download."
        else
            echo "Downloading ResNet50 weights..."
            wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P checkpoints
        fi
        ;;
    "resnet101")
        echo "Checking ResNet101 weights..."
        if [ -f "checkpoints/resnet101-63fe2227.pth" ]; then
            echo "ResNet101 weights already exist, skipping download."
        else
            echo "Downloading ResNet101 weights..."
            wget https://download.pytorch.org/models/resnet101-63fe2227.pth -P checkpoints
        fi
        ;;
    "efficientnet")
        echo "Checking EfficientNet weights..."
        if [ -f "checkpoints/efficientnet-b4-6ed6700e.pth" ]; then
            echo "EfficientNet weights already exist, skipping download."
        else
            echo "Downloading EfficientNet weights..."
            wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth -P checkpoints
        fi
        ;;
    *)
        echo "Invalid argument: $1"
        echo "Valid options are: resnet50, resnet101, efficientnet"
        exit 1
        ;;
        
esac

echo "Weights downloaded successfully to ./checkpoints"