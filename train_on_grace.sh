#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 1-
#SBATCH --job-name batch1000_pred10_300epochs_sparse
#SBATCH --output logs/grace_batchlim1000_pred10_300epochs_sparse.log
#SBATCH --mem 12GB

source ~/.bashrc
conda activate pytorch_geo
python -u train.py --name="grace_batchlim1000_pred10_300epochs_sparse" --epochs=300 --batch_size=2 --seq_len=49 --predicted_timesteps=10 --lr=1e-6 --batch_limit=1000 --sparsity_prior 
