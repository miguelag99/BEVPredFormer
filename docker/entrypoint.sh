#!/bin/bash

clear

# Define paths
OPS_PATH="/home/perception/workspace/bevpredformer/ops"

# Check if build directories exist
if [ ! -d "${OPS_PATH}/defattn/build" ] || \
   [ ! -d "${OPS_PATH}/defattn/dist" ] || \
   [ ! -d "${OPS_PATH}/gs/build" ] || \
   [ ! -d "${OPS_PATH}/gs/dist" ]; then
  
  echo -e "\033[93mBuilding and installing CUDA operations...\033[0m"
  
  # Grid Sampling
  cd "${OPS_PATH}/gs" && python setup.py build install --user && cd - || echo "Error building Grid Sampling"
  
  # Defformable Attention
  cd "${OPS_PATH}/defattn" && python setup.py build install --user && cd - || echo "Error building Defformable Attention"

else
  # Check if modules are already installed
  if ! python -c "import importlib.util; exit(0 if importlib.util.find_spec('MultiScaleDeformableAttention') and importlib.util.find_spec('sparse_gs') else 1)"; then
    echo -e "\033[93mAlready built CUDA ops, installing...\033[0m"

    # Grid Sampling
    cd "${OPS_PATH}/gs" && python setup.py install --user && cd - || echo "Error installing Grid Sampling"

    # Defformable Attention
    cd "${OPS_PATH}/defattn" && python setup.py install --user && cd - || echo "Error installing Defformable Attention"
  else
    echo -e "\033[92mCUDA Operations already installed.\033[0m"
  fi
fi

clear

figlet -c "BEVPredFormer"
echo -e "\n------------------------------------ System info ----------------------------------------\n"

# Get current username and store in a local variable
CURRENT_USER=$(whoami)

# Check if nuscenes dataset is available
echo -e "🔍 Checking Nuscenes dataset availability..."
if [ ! -d "/home/$CURRENT_USER/Datasets/nuscenes" ]; then
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
if ! python -c "import torch" 2>/dev/null; then
    echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
    echo -e "   Please check your PyTorch installation"
else
    python -c "import torch; print(f'✅ \033[92m\033[1mCUDA available: {torch.cuda.is_available()}\033[0m')"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo -e "✅ \033[92m\033[1mPyTorch is working properly with the GPU\033[0m"
        echo -e "📍 GPU Information:"
        python -c "import torch; print(f'   CUDA version: {torch.version.cuda}')"
        python -c "import torch; print(f'   GPU device: {torch.cuda.get_device_name(0)}')"
        python -c "import torch; print(f'   Available GPUs: {torch.cuda.device_count()}')"
    else
        echo -e "❌ \033[91m\033[1mCUDA is not available!\033[0m"
        echo -e "   Check your PyTorch installation"
    fi
fi

echo -e "\n-----------------------------------------------------------------------------------------\n"

/bin/bash