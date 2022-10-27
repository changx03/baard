#!/bin/bash

source ./venv/bin/activate

# For quick develop only. Set `n_att` to a larger value when running the experiment!
# Data: MNIST, Attack: FGSM
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":"inf"}' --eps="[0.03,0.06,0.09,0.12,0.16,0.19,0.22,0.25,0.28,0.31]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=FGSM --params='{"norm":2}' --eps="[1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]"

# Data: MNIST, Attack: PGD
python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.06,0.09,0.12,0.16,0.19,0.22,0.25,0.28,0.31]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1, 2, 4, 8, 16, 32, 48, 64]"

# Data: MNIST, Attack: APGD
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.06,0.09,0.12,0.16,0.19,0.22,0.25,0.28,0.31]"
python ./experiments/train_adv_examples.py -d=MNIST --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[1, 2, 4, 8, 16, 32, 48, 64]"

# Data: MNIST, Attack: CW2
python ./experiments/train_adv_examples.py -d=MNIST --attack=CW2 --params='{"max_iterations": 200}' --eps="[0, 1, 10]"

# Data: CIFAR10, Attack: FGSM
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":"inf"}' --eps="[0.03,0.06,0.09,0.12,0.16,0.19,0.22,0.25,0.28,0.31]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=FGSM --params='{"norm":2}' --eps="[1, 2, 4, 8, 16, 32, 48, 64]"

# Data: CIFAR10, Attack: PGD
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.06,0.09,0.12,0.16]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=PGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.5, 1, 2, 4, 8, 16, 32]"

# Data: CIFAR10, Attack: APGD
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":"inf", "eps_iter":0.03}' --eps="[0.03,0.06,0.09,0.12,0.16]"
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=APGD --params='{"norm":2, "eps_iter":0.1}' --eps="[0.5, 1, 2, 4, 8, 16, 32]"

# Data: CIFAR10, Attack: CW2
python ./experiments/train_adv_examples.py -d=CIFAR10 --attack=CW2 --params='{"max_iterations": 200}' --eps="[0, 1, 10]"
