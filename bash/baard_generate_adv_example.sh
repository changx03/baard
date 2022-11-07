#!/bin/bash

# This script generates adversarial examples for the ablation study for BAARD.

source ./.venv/bin/activate
pip install --upgrade .

# A seed starts with 6 to indicate this is for ablation study. 
# 1-5 are reserved for repeated experiment for grey-box evaluation.
SEED=643896
SIZE=1000

# 1. More espilon on both L2 and Linf
echo "Running MNIST #######################################################"
DATA="MNIST"
# Attack:APGD
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.03,0.09,0.16,0.22,0.28,0.34,0.41,0.47,0.53,0.59,0.66,0.72,0.78,0.84,0.91,0.97,1]"
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.1}' --eps="[0.5,1,2,3,4,5,6,7,8,9,10]"

echo "Running CIFAR10 #####################################################"
DATA="CIFAR10"
# Attack:APGD
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01,0.02,0.03,0.05,0.06,0.08,0.09,0.11,0.12,0.14,0.16,0.17,0.19,0.2,0.22,0.23,0.25,0.27,0.28,0.3]"
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":2,"eps_iter":0.05}' --eps="[0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6,3.9]"
