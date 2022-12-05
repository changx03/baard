#!/bin/bash -e
#SBATCH --job-name=graybox543
#SBATCH --output=logs/graybox543_%a.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-9

module load CUDA/11.6.2
module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03637/baard_v4/.venv/bin/activate

SEEDS=(543597 466364 382347)
DETECTORS=("BAARD-S1" "BAARD-S2" "BAARD-S3" "BAARD" "FS" "LID" "ML-LOO" "Odds" "PN" "RC")
DATASETS=("MNIST" "CIFAR10")

for SEED in ${SEEDS[@]}; do
    for DATA in ${DATASETS[@]}; do
        ATTACK="APGD"
        NORMS=("inf" "2")
        for NORM in ${NORMS[@]}; do
            echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
            python /nesi/project/uoa03637/baard_v4/experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector ${DETECTORS[$SLURM_ARRAY_TASK_ID]}
        done

        ATTACK="CW2"
        NORM="2"
        echo "Running $DETECTOR on $DATA with $ATTACK L$NORM"
        python /nesi/project/uoa03637/baard_v4/experiments/extract_features.py -s $SEED --data $DATA --attack $ATTACK -l $NORM --detector ${DETECTORS[$SLURM_ARRAY_TASK_ID]}
    done
done
