#!/bin/bash

# This script runs linear search for fining the minimal epsilon. The default number of adversarial examples is 100, which is too small for testing the detector.

source ./.venv/bin/activate
python -m pip install --upgrade .

# For quick develop only. Set `n_att` to a larger value when running the experiment!
# Data:MNIST,Attack:FGSM
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf"}' --eps="[0.03,0.09,0.16,0.22,0.28,0.34,0.41,0.47,0.53,0.59,0.66,0.72,0.78,0.84,0.91,0.97,1]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":2}' --eps="[0.5,1,2,4,6,8,10,12]"

# Data:MNIST,Attack:PGD
python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.03,0.09,0.16,0.22,0.28,0.34,0.41,0.47,0.53,0.59,0.66,0.72,0.78,0.84,0.91,0.97,1]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":2,"eps_iter":0.1}' --eps="[0.5,1,2,4,6,8,10,12]"

# Data:MNIST,Attack:APGD
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.03,0.09,0.16,0.22,0.28,0.34,0.41,0.47,0.53,0.59,0.66,0.72,0.78,0.84,0.91,0.97,1]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":2,"eps_iter":0.1}' --eps="[0.5,1,2,4,6,8,10,12]"

# Data:MNIST,Attack:CW2
python ./experiments/train_adv_examples.py -d=MNIST --attack=CW2 --params='{"max_iterations":200}' --eps="[0,1,10]"

# Data:CIFAR10,Attack:FGSM
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":"inf"}' --eps="[0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":2}' --eps="[0.1,0.2,0.3,0.4,0.5,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]"

Data:CIFAR10,Attack:PGD
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":2,"eps_iter":0.05}' --eps="[0.1,0.2,0.3,0.4,0.5,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]"

# Data:CIFAR10,Attack:APGD
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":2,"eps_iter":0.05}' --eps="[0.1,0.2,0.3,0.4,0.5,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]"

# Data:CIFAR10,Attack:CW2
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=CW2 --params='{"max_iterations":200}' --eps="[0,1,10]"
