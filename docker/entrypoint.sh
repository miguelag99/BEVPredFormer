#!/bin/bash

# Define paths
OPS_PATH="/home/perception/workspace/bevpredformer/ops"

# Check if build directories exist
if [ ! -d "${OPS_PATH}/defattn/build" ] || \
   [ ! -d "${OPS_PATH}/defattn/dist" ] || \
   [ ! -d "${OPS_PATH}/gs/build" ] || \
   [ ! -d "${OPS_PATH}/gs/dist" ]; then
  
  echo -e "\033[93mBuilding and installing operations...\033[0m"
  
  # Grid Sampling
  cd "${OPS_PATH}/gs" && python setup.py build install --user && cd - || echo "Error building Grid Sampling"
  
  # Defformable Attention
  cd "${OPS_PATH}/defattn" && python setup.py build install --user && cd - || echo "Error building Defformable Attention"

else
  # Check if modules are already installed
  if ! python -c "import importlib.util; exit(0 if importlib.util.find_spec('MultiScaleDeformableAttention') and importlib.util.find_spec('sparse_gs') else 1)"; then
    echo -e "\033[93mAlready built ops, installing operations...\033[0m"

    # Grid Sampling
    cd "${OPS_PATH}/gs" && python setup.py install --user && cd - || echo "Error installing Grid Sampling"

    # Defformable Attention
    cd "${OPS_PATH}/defattn" && python setup.py install --user && cd - || echo "Error installing Defformable Attention"
  else
    echo -e "\033[92mOperations already installed.\033[0m"
  fi
fi

clear

figlet -c "BEVPredFormer"
echo -e "\n------------------------------------ System info ----------------------------------------\n"

# Get current username and store in a local variable
CURRENT_USER=$(whoami)

# Check if nuscenes dataset is available at /home/$CURRENT_USER/Datasets/nuscenes
if [ ! -d "/home/$CURRENT_USER/Datasets/nuscenes" ]; then
  echo -e "\033[91mWarning: Nuscenes dataset not found. Please download it and place it in /home/$CURRENT_USER/Datasets/nuscenes\033[0m"
  exit 1
else
  echo -e "\033[92mNuscenes dataset found at /home/$CURRENT_USER/Datasets/nuscenes\033[0m"
fi

# Check if CUDA is available
echo -e "\nChecking GPU and CUDA availability..."
if ! python -c "import torch" 2>/dev/null; then
  echo -e "\033[91mFailed to import torch. Please check your PyTorch installation\033[0m"
else
  python -c "import torch; print(f'\033[92mCUDA available: {torch.cuda.is_available()}\033[0m')"
  
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "\033[92mPyTorch is working properly with the GPU\033[0m"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'Number of available GPUs: {torch.cuda.device_count()}')"
  else
    echo -e "\033[91mCUDA is not available! Check your PyTorch installation\033[0m"
  fi
fi

echo -e "\n-----------------------------------------------------------------------------------------\n"

/bin/bash