# BAARD: Blocking Adversarial examples by testing for Applicability, Reliability and Decidability

## Install

All scripts are tested on Python `3.9.15` with PyTorch `1.12.1+cu116` on Ubuntu `20.04.5 LTS`.

```bash
# Only tested on Linux
python3.9 -m venv venv  # Create virtual environment
source ./venv/bin/activate  # Activate venv

pip install --upgrade pip  

# Install PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
pip install -r requirements.txt

# To check if PyTorch is correctly installed and GPU is working.
python ./examples/test_gpu.py
```
