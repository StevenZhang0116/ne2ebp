#!/bin/bash

#SBATCH --job-name=tp_test  
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=02-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=tp_opt.txt
#SBATCH --error=tp_err.txt

/gscratch/deepthought/zihan/miniconda3/envs/playground/bin/python3.8 main.py --batch_size 32 --action train --epochs 50


