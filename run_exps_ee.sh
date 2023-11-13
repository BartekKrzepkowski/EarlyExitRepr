#!/bin/bash
##ATHENA
#SBATCH --job-name=ncollapse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

eval "$(conda shell.bash hook)"
conda activate ncollapse
nvidia-smi
CHECKPOINT1="/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2/just_run, sgd, cifar10, resnet_tunnel_lr_0.1_momentum_0.0_wd_0.0_lr_lambda_1.0, phase1/2023-11-12_16-54-40/checkpoints/model_step_epoch_160_global_step_64000.pth"
CHECKPOINT2="/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2/just_run, sgd, cifar10, resnet_tunnel_lr_0.1_momentum_0.0_wd_0.0_lr_lambda_1.0, phase1/2023-11-12_16-54-40/checkpoints/model_step_epoch_180_global_step_72000.pth"
PHASE1=160
PHASE2=180
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 1e-1 0.0 0 "${CHECKPOINT1}" $PHASE1 &
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python -m scripts.python.$1 1e-1 0.0 0 "${CHECKPOINT2}" $PHASE2 &
wait