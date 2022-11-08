#!/bin/bash

# This file tests the bash scripts.

python ./experiments/baard_tune.py -s 643896 --data MNIST --detector "BAARD-S3" -l 2 --eps 4.0

python ./experiments/extract_features.py -s 643896 --data MNIST --attack APGD -l 2 --detector "BAARD-S2"

python ./experiments/train_adv_examples.py -s 3456 -d=MNIST --attack=FGSM --params='{"norm":2}' --eps="[1,4,8]" --n_val=1000