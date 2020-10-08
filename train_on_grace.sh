#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --time 10:00:00
#SBATCH --job-name batch1000_pred10_200epochs_sparse_traincheckpoint
#SBATCH --output logs/grace_batchlim1000_pred10_200epochs_sparse_traincheckpoint.log
#SBATCH --mem 12GB

source ~/.bashrc
conda activate pytorch_geo
python -u train.py --name="grace_batchlim1000_pred10_300epochs_sparse_traincheckpoint" --epochs=200 --batch_size=2 --seq_len=49 --predicted_timesteps=10 --lr=5e-7 --batch_limit=1000 --sparsity_prior 
