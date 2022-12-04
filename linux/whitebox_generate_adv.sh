#!/bin/bash

# This script generates adversarial examples for the ablation study for BAARD.

source ./.venv/bin/activate
python -m pip install --upgrade .

# A seed starts with 7 to indicate this is for whitebox.
# SEED=727328
SEED=829257
SIZE=1000

# 1. More espilon on both L2 and Linf
echo "Running MNIST ###########################################################"
DATA="MNIST"
# Attack:APGD
# This is for creating the initial test and validation sets.
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.03}' --eps="[0.22]"

echo "Running MNIST Linf ######################################################"
EPS_LIST=(0.03 0.09 0.16 0.22 0.28 0.34 0.4 0.6 0.8 1.0)
for EPS in ${EPS_LIST[@]}; do
    python ./experiments/baard_whitebox.py --data MNIST --seed $SEED --norm inf --epsiter 0.03 --eps $EPS
    python ./experiments/baard_whitebox.py --data MNIST --seed $SEED --norm inf --epsiter 0.03 --eps $EPS --clean "ValClean-1000.pt" --validation 1
done

echo "Running MNIST L2 ########################################################"
EPS_LIST=(0.5 1.0 2.0 3.0 4.0 6.0 8.0 10.0)
for EPS in ${EPS_LIST[@]}; do
    python ./experiments/baard_whitebox.py --data MNIST --seed $SEED --norm 2 --epsiter 0.1 --eps $EPS
    python ./experiments/baard_whitebox.py --data MNIST --seed $SEED --norm 2 --epsiter 0.1 --eps $EPS --clean "ValClean-1000.pt" --validation 1
done

echo "Running CIFAR10 #########################################################"
DATA="CIFAR10"
# Attack:APGD
python ./experiments/train_adv_examples.py --seed=$SEED --n_att=$SIZE --n_val=$SIZE -d=$DATA --attack=APGD --params='{"norm":"inf","eps_iter":0.01}' --eps="[0.01]"

echo "Running CIFAR10 Linf ####################################################"
EPS_LIST=(0.01 0.03 0.05 0.1 0.15 0.2 0.3)
for EPS in ${EPS_LIST[@]}; do
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed $SEED --norm inf --epsiter 0.01 --eps $EPS
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed $SEED --norm inf --epsiter 0.01 --eps $EPS --clean "ValClean-1000.pt" --validation 1
done

echo "Running CIFAR10 L2 ######################################################"
EPS_LIST=(0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0 4.0)
for EPS in ${EPS_LIST[@]}; do
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed $SEED --norm 2 --epsiter 0.05 --eps $EPS
    python ./experiments/baard_whitebox.py --data CIFAR10 --seed $SEED --norm 2 --epsiter 0.05 --eps $EPS --clean "ValClean-1000.pt" --validation 1
done
