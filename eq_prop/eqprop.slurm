#!/bin/bash

#SBATCH --job-name=ep_test  
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=02-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=ep_opt.txt
#SBATCH --error=ep_err.txt

python train.py --num-epochs 50 --free-iters 20 --clamped-iters 4 --beta 0.5 --dt 0.5 --layer-sizes 784 500 10 --learning-rates 0.1 0.05 --num-workers 1

