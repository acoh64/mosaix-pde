#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -o diffusion_64.log-%A-%a
#SBATCH -t 6:00:00
#SBATCH --job-name=diffusion_64

module load miniforge/24.3.0-0

conda activate pde-opt-env

python -u scripts/run_training.py