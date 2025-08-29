#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o training.log-%A-%a
#SBATCH -t 6:00:00

# Set up environment
module load miniforge/24.3.0-0

conda activate pde-opt-env

# Run your application
python -u optimize_nn_script.py