#!/bin/bash

# This script tunes the parameter `scale` for Stage 2 and 3 in BAARD.
# Tuning K first, before tuning the parameter `scale`.

source ./.venv/bin/activate
python -m pip install --upgrade .

SEED=643896
ATTACK="APGD"
DETECTORS=("BAARD-S2" "BAARD-S3") # Only Stage 2 and 3 require tuning.

DATA="MNIST"
NORM="inf"
EPS=0.22 # Min. epsilon to achieve 95 success rate.
DETECTOR="BAARD-S2"
K=5
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

DETECTOR="BAARD-S3"
K=100
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

NORM=2
EPS=4.0 # Min. epsilon to achieve 95 success rate.

DETECTOR="BAARD-S2"
K=5
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

DETECTOR="BAARD-S3"
K=100
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

DATA="CIFAR10"
NORM="inf"
EPS=0.01 # Min. epsilon to achieve 95 success rate.
DETECTOR="BAARD-S2"
K=5
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

DETECTOR="BAARD-S3"
K=100
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

NORM=2
EPS=0.3 # Min. epsilon to achieve 95 success rate.

DETECTOR="BAARD-S2"
K=5
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K

DETECTOR="BAARD-S3"
K=100
echo "Running $DETECTOR on $DATA with $ATTACK L$NORM eps=$EPS"
python ./experiments/baard_tune.py -s $SEED --data $DATA --detector $DETECTOR -l $NORM --eps $EPS -k $K
