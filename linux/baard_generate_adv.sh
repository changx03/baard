#!/bin/bash

# This script generates adversarial examples for the ablation study for BAARD.

source ./.venv/bin/activate
python -m pip install --upgrade .

# A seed starts with 6 to indicate this is for ablation study.
# 1-5 are reserved for repeated experiment for grey-box evaluation.
# SEED=643896
SEED=945269
SIZE=1000

# 1. More espilon on both L2 and Linf
echo "Running MNIST #######################################################"
DATA="MNIST"
# Attack:APGD
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.03,0.09,0.16,0.22,0.28,0.34,0.4,0.6,0.8,1.0]"
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.1}' --eps="[0.5,1.0,2.0,3.0,4.0,6.0,8.0,10.0]"

echo "Running CIFAR10 #####################################################"
DATA="CIFAR10"
# Attack:APGD
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01,0.03,0.05,0.1,0.15,0.2,0.3]"
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.05}' --eps="[0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,4.0]"
