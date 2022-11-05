#!/bin/bash

# This scirpt create a virtual environment on Linux.

echo "Creating venv using Python3.9"

# Only tested on Linux
python3.9 -m venv .venv  # Create virtual environment
source ./.venv/bin/activate  # Activate venv

pip install --upgrade pip

# Install PyTorch
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
pip install -r requirements.txt

# Install local package
pip install --upgrade .

# To check if PyTorch is correctly installed and GPU is working.
python ./examples/check_gpu.py
