# BAARD: Blocking Adversarial examples by testing for Applicability, Reliability and Decidability

BAARD is a framework for detecting adversarial examples.

This is the repository for the paper -- _BAARD: Blocking Adversarial examples by testing for Applicability, Reliability and Decidability_, which is accepted by PAKDD 2023.

(TODO: Paper link will be updated.)

## Abstract

Adversarial defenses protect machine learning models from adversarial attacks, but are often tailored to one type of model or attack. The lack of information on unknown potential attacks makes detecting adversarial examples challenging. Additionally, attackers do not need to follow the rules made by the defender. To address this problem, we take inspiration from the concept of Applicability Domain in cheminformatics. Cheminformatics models struggle to make accurate predictions because only a limited number of compounds are known and available for training. Applicability Domain defines a domain based on the known compounds and rejects any unknown compound that falls outside the domain. Similarly, adversarial examples start as harmless inputs, but can be manipulated to evade reliable classification by moving outside the domain of the classifier. We are the first to identify the similarity between Applicability Domain and adversarial detection. Instead of focusing on unknown attacks, we focus on what is known, the training data. We propose a simple yet robust triple-stage data-driven framework that checks the input globally and locally, and confirms that they are coherent with the model's output. This framework can be applied to any classification model and is not limited to specific attacks. We demonstrate these three stages work as one unit, effectively detecting various attacks, even for a white-box scenario.

## Install

Required packages:

- adversarial-robustness-toolbox
- jupyterlab
- lightning-bolts
- matplotlib
- opencv-python
- pandas
- pytorch-lightning
- scikit-learn
- seaborn
- torch
- torchmetrics
- torchvision
- tqdm

All scripts are tested on Python `3.9.15` with PyTorch `1.12.1+cu116` on Ubuntu `20.04.5 LTS`.

```bash
# Only tested on Linux
python3.9 -m venv .venv  # Create virtual environment
source ./.venv/bin/activate  # Activate venv

python -m pip install --upgrade pip

# Install PyTorch
python -m pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
python -m pip install -r requirements.txt

# Install local package
python -m pip install --upgrade .

# To check if PyTorch is correctly installed and GPU is working.
python ./examples/check_gpu.py
```

### Shortcut for creating `venv`

Or you can run bash `create_venv.sh` which contains the script above.

```bash
bash ./create_venv.sh
```

### NOTE

- All bash script and terminal commands are prepared for **Linux** only. Changing the script according when running on a Windows machine.
- If an alternative version of `PyTorch` is installed, remove all `PyTorch` related packages from `requirements.txt` file,
  and install them manually, including: `pytorch-lightning`, `torch`, `torch-tb-profiler`, `torchinfo`, `torchmetrics`,
  and `torchvision`.
- `OpenCV` is required for `Feature Squeezing` detector. The script from `requirements.txt` will try to install
  a pre-build **CPU-only** version. Check [here](https://pypi.org/project/opencv-python/) for more details.

## Train Classifiers

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

### NOTE

- All pre-trained models are saved under `./pretrained_clf/`.

## Generate Adversarial Examples

### Basic usage

```bash
# Usage
python ./experiments/train_adv_examples.py [-s SEED] [-d DATASET_NAME] [--n_att NB_ADV_EXAMPLES] [--n_val NB_VAL_EXAMPLES] [-a ATTACK_NAME]  [--eps LIST_OF_EPSILON] [--params ATTACK_PARAMS]
```

### Available options

- '-s', '--seed': Seed value. The **output folder name** is based on the seed value.
- '-d', '--data': Dataset. Either `MNIST` or `CIFAR10`.
- '--n_att': Number of adversarial examples want to generate. Default is `100`. Use `1000` for the actual experiment.
- '--n_val': Number of validation examples. The validation set comes from the correctly classified test set.
  This set will be used by the defense. In the experiment, use `1000`. `n_att + n_val` must be smaller than test set. Default is `1000`.
- '-a', '--attack': Adversarial attack. One of `FGSM`, `PGD`, `APGD`, `CW2`.
- '--eps': A list of epsilons as a JSON string. e.g., --eps="[0.06, 0.13, 0.25]". In C&W attack, this controls the confidence parameter c. Default is "[0.06]".
- '--params': Parameters for the adversarial attack as a JSON string. e.g., `{"norm":"inf", "clip_min":0, "clip_max":1}`.
  This JSON string will be converted into a dictionary and pass directly to the attack. Check `./baard/attacks` to see the specific parameters for each attack.

### List of tabular datasets

We prepared several tabular data for testing. The code [here](experiments/tabular_preprocessing.ipynb) shows the preprocessing process.

- `abalone`
- `banknote`
- `BC`
- `CMC`
- `HTRU2`
- `phoneme`
- `ringnorm`
- `texture`
- `yeast`

### NOTE

- Windows OS cannot cannot escape double quote `"`, e.g., `'{"norm":"inf"}'`. Use `\` to escape the double quote `"` symbol, e.g., `'{\"norm\":\"inf\"}'`.

### Attack example

Generating 100 adversarial examples on `MNIST` using FGSM on L2 norm with Epsilon=0.06

```bash
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf", "clip_min":0, "clip_max":1}' --eps="[0.06]" --n_att=100 --n_val=1000
```

## Code demo

Code demo can be found under `./examples/`.

## Run experiments from terminal

Under `./linux` folder, there are scripts for running the experiment on a Linux machine.
