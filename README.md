# BAARD: Blocking Adversarial examples by testing for Applicability, Reliability and Decidability

## Install

All scripts are tested on Python `3.9.15` with PyTorch `1.12.1+cu116` on Ubuntu `20.04.5 LTS`.

```bash
# Only tested on Linux
python3.9 -m venv venv  # Create virtual environment
source ./venv/bin/activate  # Activate venv

pip install --upgrade pip

# Install PyTorch
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
pip install -r requirements.txt

# To check if PyTorch is correctly installed and GPU is working.
python ./examples/check_gpu.py
```

## Train clasifiers

To train CNN for MNIST:

```bash
# By default this code check if GPU is available.
# Default seed is 1234, use `--seed=1234` to change it.
python ./classifiers/mnist_cnn.py

# To test the environment, run:
python ./classifiers/mnist_cnn.py --fast_dev_run=true

```
