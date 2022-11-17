#!/bin/bash

# This script generates adversaral examples for grey-box benchmark.

source ./.venv/bin/activate
python -m pip install --upgrade .

SEEDS=(188283 292478 382347 466364 543597)

for SEED in ${SEEDS[@]}; do
    python ./experiments/svm_train_adv.py --seed=$SEED -d=banknote --eps="[0.2,0.6]"
    python ./experiments/svm_train_adv.py --seed=$SEED -d=BC --eps="[0.2,0.6]"
    python ./experiments/tree_train_adv.py --seed=$SEED -d=banknote
    python ./experiments/tree_train_adv.py --seed=$SEED -d=BC
done
