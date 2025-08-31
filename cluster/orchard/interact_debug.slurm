#!/bin/bash
#SBATCH --job-name=interactive_debug
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}@andrew.cmu.edu

source /project/flame/$USER/miniconda3/etc/profile.d/conda.sh

echo "Job started on $(hostname) at $(date)"
nvidia-smi

export HYDRA_FULL_ERROR=1

# only have to run this once
# python /project/flame/$USER/miniconda3/envs/equidiff/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py

sleep 3600
