#!/bin/bash

# This script runs the grey-box evaluation.
# NOTE: Adversarial examples need to be pre-trained.

source ./.venv/bin/activate
python -m pip install --upgrade .

SEEDS=(188283 292478 382347 466364 543597)
# SEEDS=(188283) # TODO: Run 1 repeatation first
# SEEDS=(1234) # Use this for testing!
DETECTORS=("BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD" "FS" "LID" "ML-LOO" "Odds" "PN" "RC")
DATASETS=("MNIST" "CIFAR10")

for SEED in ${SEEDS[@]}; do
    for DATA in ${DATASETS[@]}; do
        for DETECTOR in ${DETECTORS[@]}; do
            ATTACK="APGD"
            NORMS=("inf" "2")
            for NORM in ${NORMS[@]}; do
                echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
                python ./experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
            done

            ATTACK="CW2"
            NORM="2"
            echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
            python ./experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
        done
    done
done
