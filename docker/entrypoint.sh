#!/bin/bash

clear

# Update uv project installation
if [ ! -d ".venv" ]; then
  echo " .venv not found — updating virtual environment..."
  uv sync --link-mode=copy
  echo " ✅ .venv updated"
fi

echo -e "\n-----------------------------------------------------------------------------------------\n"
figlet -c "BEVPredFormer"
echo -e "\n------------------------------------ System info ----------------------------------------\n"

# Get current username and store in a local variable
CURRENT_USER=$(whoami)

# Check if nuscenes dataset is available
echo -e "🔍 Checking Nuscenes dataset availability..."
if [ ! -d "/home/$CURRENT_USER/Datasets/nuscenes/samples" ]; then
    echo -e "❌ \033[91m\033[1mNuscenes dataset not found\033[0m"
    echo -e "   Please download it and place it in /home/$CURRENT_USER/Datasets/nuscenes or set the correspondign path in config files"
    exit 1
else
    echo -e "✅ \033[92m\033[1mNuscenes dataset found\033[0m"
    echo -e "📍 Dataset path:"
    echo -e "   /home/$CURRENT_USER/Datasets/nuscenes"
fi

# Check if CUDA is available
echo -e "\n🔍 Checking GPU and CUDA availability..."
if ! uv run python -c "import torch" 2>/dev/null; then
    echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
    echo -e "   Please check your PyTorch installation!"
else
    CUDA_AVAILABLE=$(uv run python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        echo -e "✅ \033[92m\033[1mPyTorch is working properly with the GPU.\033[0m"
        echo -e "📍 GPU Information:"
        uv run python -c "import torch; print(f'   - CUDA version:     {torch.version.cuda}')"
        uv run python -c "import torch; print(f'   - Device name:      {torch.cuda.get_device_name(0)}')"
        uv run python -c "import torch; print(f'   - Number of GPUs:   {torch.cuda.device_count()}')"
    else
        echo -e "❌ \033[91m\033[1mCUDA is not available!\033[0m"
        echo -e "   Check your PyTorch installation"
    fi
fi

echo -e "\n-----------------------------------------------------------------------------------------\n"

ulimit -n 65535
/bin/bash