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

The Python script for training the classifier takes command line arguments and passes them to `PyTorch-Lightning`'s `Trainer` class.
A full list of parameters can be found [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

To train `CNN` for `MNIST`:

```bash
# Usage:
python ./baard/classifiers/mnist_cnn.py [--max_epochs MAX_EPOCHS] [--seed SEED] ...

python ./baard/classifiers/cifar10_resnet18.py [--max_epochs MAX_EPOCHS] [--seed SEED] ...
```

### Classifier Example

```bash
# To test the environment, run:
python ./baard/classifiers/mnist_cnn.py --fast_dev_run=true

# To train the model, run:
# By default this code check if GPU is available.
# Default seed is 1234, use `--seed=1234` to change it.
python ./baard/classifiers/mnist_cnn.py

```

To train `ResNet-18` for `CIFAR-10`:

```bash
# This Python script takes the same arguments as `mnist_cnn.py`
python ./baard/classifiers/cifar10_resnet18.py
```

To check log files from `TensorBoard`, run:

```bash
tensorboard --logdir logs
```

## Generate adversarial examples

### Basic usage

Generating 100 adversarial examples on `MNIST` using FGSM on L2 norm with Epsilon=0.06

```bash
# Usage
python ./experiments/train_adv_examples.py [-s SEED] [-d DATASET_NAME] [--n_att NB_ADV_EXAMPLES] [--n_val NB_VAL_EXAMPLES] [-a ATTACK_NAME]  [--eps LIST_OF_EPSILON] [--params ATTACK_PARAMS]
```

### Available options

- '-s', '--seed': Seed value. The result folder name is based on the seed value.
- '-d', '--data': Dataset. Either `MNIST` or `CIFAR10`. (TODO: Add SVHN, and tabular datasets.)
- '--n_att': Number of adversarial examples want to generate. Default is `100`. Use `1000` for the actual experiment.
- '--n_val': Number of validation examples. The validation set comes from the correctly classified test set.
  This set will be used by the defense. In the experiment, use `1000`. `n_att + n_val` must be smaller than test set. Default is `1000`.
- '-a', '--attack': Adversarial attack. One of `FGSM`, `PGD`, `APGD`, `CW2`.
- '--eps': A list of epsilons as a JSON string. e.g., --eps="[0.06, 0.13, 0.25]". In C&W attack, this controls the confidence parameter c. Default is "[0.06]".
- '--params': Parameters for the adversarial attack as a JSON string. e.g., `{"norm":"inf", "clip_min":0, "clip_max":1}`.
  This JSON string will be converted into a dictionary and pass directly to the attack. Check `./baard/attacks` to see the specific parameters for each attack.

### Attack Example

```bash
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf", "clip_min":0, "clip_max":1}' --eps="[0.06]"
```

