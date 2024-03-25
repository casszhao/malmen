#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --qos=gpu 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=2
#SBATCH --mem=150G 
#SBATCH --time 3-00:00:00 
#SBATCH --job-name=ed10000

# Load modules & activate env

module load Anaconda3/2022.05
module load CUDA/11.8.0

# Activate env
source activate malmen


python main.py data=zsre model=gpt-j  editor=malmen