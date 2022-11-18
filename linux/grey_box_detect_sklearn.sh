#!/bin/bash

# This script runs the grey-box evaluation.
# NOTE: Adversarial examples need to be pre-trained.

source ./.venv/bin/activate
python -m pip install --upgrade .

# SEEDS=(188283) # Test 1 repeatation 1st!
SEEDS=(188283 292478 382347 466364 543597)
DETECTORS=("BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD" "RC")
DATASETS=("banknote" "BC")

for SEED in ${SEEDS[@]}; do
    for DATA in ${DATASETS[@]}; do
        for DETECTOR in ${DETECTORS[@]}; do
            python ./experiments/extract_features_sklearn.py -s $SEEDS --data $DATA --model="SVM" --detector $DETECTOR -a "PGD-Linf"
            python ./experiments/extract_features_sklearn.py -s $SEEDS --data $DATA --model="DecisionTree" --detector $DETECTOR -a "DecisionTreeAttack"
        done
    done
done
