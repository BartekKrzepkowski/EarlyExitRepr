#!/bin/bash
##DGX
#SBATCH --job-name=ncollapse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=batch
#SBATCH --nodelist=login01
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate ncollapse
WANDB__SERVICE_WAIT=300 python3 -m scripts.python.$1 $2 $3 $4