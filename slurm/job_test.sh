#!/bin/bash -e
#SBATCH --job-name=test_job
#SBATCH --output=logs/test_job_%a.out
#SBATCH --time=5:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module load CUDA
module load Python/3.9.9-gimkl-2020a
source /nesi/project/uoa03637/baard_v4/.venv/bin/activate

python /nesi/project/uoa03637/baard_v4/examples/check_gpu.py
