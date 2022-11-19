#!/bin/bash

# This script runs the ablation study for BAARD.
# It tests BAARD on a wide range of epsilon.
# NOTE: The adversarial examples MUST be pre-trained!

source ./.venv/bin/activate
python -m pip install --upgrade .

echo "Compute accuracy on adversarial examples. ###############################"
python ./experiments/whitebox_save_acc.py

SEED=727328
ATTACK="whitebox"
NORMS=("inf" "2")
DETECTORS=("BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD")
DATASETS=("MNIST" "CIFAR10")

for DATA in ${DATASETS[@]}; do
    for DETECTOR in ${DETECTORS[@]}; do
        for NORM in ${NORMS[@]}; do
            echo "Running $DETECTOR on $DATA with $ATTACK L$NORM ##############"
            python ./experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
        done
    done
done
