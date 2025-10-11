#!/bin/bash
#SBATCH --job-name=train_lerobot
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}@andrew.cmu.edu

echo "Job started on $(hostname) at $(date)"
export HYDRA_FULL_ERROR=1

pixi run python scripts/train.py \
	model=articubot \
	dataset=liberoLerobot \
	dataset.repo_id=sriramsk/libero_lerobot_all_heatmapGoal \
	dataset.cache_dir=/tmp/libero_all_cache \
	model.use_rgb=True \
	model.in_channels=7 \
	training.batch_size=48 \
	resources.num_workers=48 \
	training.check_val_every_n_epochs=1
