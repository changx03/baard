#!/bin/bash -e
#SBATCH --job-name=graybox_test
#SBATCH --output=logs/graybox_test_%a.out
#SBATCH --time=10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03637/baard_v4/.venv/bin/activate

SEED=382347
DATA="MNIST"
ATTACK="APGD"
NORM="inf"
DETECTOR="BAARD-S1"

python /nesi/project/uoa03637/baard_v4/experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector $DETECTOR
