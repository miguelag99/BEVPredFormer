#!/bin/bash
# Create checkpoints directory if it doesn't exist
mkdir -p ./checkpoints

RELEASE_URL="https://github.com/miguelag99/BEVPredFormer/releases/download/v2.0.0"

# Stage-2 prediction models (release v2.0.0)
PREDICTION_CKPTS=(
    "effvit_SpUnet_2TripletTST_256_ps4_scale05.ckpt"
    "effvit_SpUnet_2TripletTST_256_ps4_scale03.ckpt"
    "effvit_SpUnet_2TripletTST_256_ps4_scale05_short_range.ckpt"
)

# Stage-1 segmentation backbones (release v2.0.0)
BACKBONE_CKPTS=(
    "BEVPredformer_Backbone_effvitL2_03.ckpt"
    "BEVPredformer_Backbone_effvitL2_05.ckpt"
    "BEVPredformer_Backbone_effvitL2_1.ckpt"
    "BEVPredformer_Backbone_effvitL2_05_short_range.ckpt"
)

download_release() {
    for name in "$@"; do
        if [ -f "checkpoints/$name" ]; then
            echo "$name already exists, skipping download."
        else
            echo "Downloading $name..."
            wget "$RELEASE_URL/$name" -P checkpoints || exit 1
        fi
    done
}

# Check user argument
if [ "$#" -eq 0 ]; then
    echo "Usage: ./download_weights.sh [prediction|backbone|all|resnet50|resnet101|efficientnet]"
    exit 1
fi

case "$1" in
    "prediction")
        download_release "${PREDICTION_CKPTS[@]}"
        ;;
    "backbone")
        download_release "${BACKBONE_CKPTS[@]}"
        ;;
    "all")
        download_release "${PREDICTION_CKPTS[@]}" "${BACKBONE_CKPTS[@]}"
        ;;
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
        echo "Valid options are: prediction, backbone, all, resnet50, resnet101, efficientnet"
        exit 1
        ;;

esac

echo "Weights downloaded successfully to ./checkpoints"
