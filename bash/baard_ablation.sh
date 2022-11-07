#!/bin/bash

# This script runs the ablation study for BAARD.
# This script run BAARD on a wide range of epsilon.
# NOTE: The adversarial examples MUST be pre-trained!

source ./.venv/bin/activate
pip install --upgrade .

SEED=643896
SIZE=1000
ATTACK="APGD"
NORMS=("inf" "2")
DETECTORS=("BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD")
DATASETS=("MNIST" "CIFAR10")

for DATA in ${DATASETS[@]}; do
    for DETECTOR in ${DETECTORS[@]}; do
        ATTACK="APGD"
        NORMS=("inf" "2")
        for NORM in ${NORMS[@]}; do
            echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
            python ./experiments/detectors_extract_features.py --s $SEEDS --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
        done
    done
done
