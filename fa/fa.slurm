#!/bin/bash

#SBATCH --job-name=fa_bp_test  
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=02-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=fa_opt.txt
#SBATCH --error=fa_err.txt

python train_fa_vs_bp_linear_model.py

