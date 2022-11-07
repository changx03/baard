#!/bin/bash

# This script runs the grey-box evaluation.

source ./.venv/bin/activate
pip install --upgrade .

# SEEDS=(188283 292478 382347 466364 543597)
SEEDS=(1234) # TODO: Run 1 repeatation first
DETECTORS=("FS" "LID" "ML-LOO" "Odds" "PN" "RC" "BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD")
DATASETS=("MNIST" "CIFAR10")

for DATA in ${DATASETS[@]}; do
    for DETECTOR in ${DETECTORS[@]}; do
        ATTACK="APGD"
        NORMS=("inf" "2")
        for NORM in ${NORMS[@]}; do
            echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
            python ./experiments/detectors_extract_features.py --s $SEEDS --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
        done

        ATTACK="CW2"
        NORM="2"
        echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
        python ./experiments/detectors_extract_features.py --s $SEEDS --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
    done
done
