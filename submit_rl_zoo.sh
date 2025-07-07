#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH -o rl_zoo.log-%A-%a
#SBATCH -t 6:00:00
#SBATCH --job-name=rl_zoo

module load miniforge/24.3.0-0

conda activate pde-opt-env

python -u scripts/rl_zoo_training.py --algo dqn --env AdvectionDiffusion-v0 --eval-freq 10000 --save-freq 50000