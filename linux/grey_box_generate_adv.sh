#!/bin/bash

# This script generates adversaral examples for grey-box benchmark.

source ./.venv/bin/activate
python -m pip install --upgrade .

# SEEDS=(188283 292478 382347 466364 543597)
SEEDS=(188283) # TODO: Run 1 repeatation first
SIZE=1000

for SEED in ${SEEDS[@]}; do
    echo "Running MNIST #######################################################"
    DATA=MNIST
    # Attack:APGD
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.22,0.66]"
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.1}' --eps="[4,8]"
    # Attack:CW2
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=CW2 --params='{"max_iterations":200}' --eps="[0]"

    echo "Running CIFAR10 #####################################################"
    DATA=CIFAR10
    # Attack:APGD
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01,0.1]"
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.05}' --eps="[0.3,3]"
    # Attack:CW2
    python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=CW2 --params='{"max_iterations":200}' --eps="[0]"
done