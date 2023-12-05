#!/bin/bash

#SBATCH --job-name=hsic_test  
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=02-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=hsic_opt.txt
#SBATCH --error=hsic_err.txt

python torch_hsic.py



