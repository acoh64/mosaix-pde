#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o log_files/timing_solver_%j.out
#SBATCH -t 6:00:00

# Set up environment
# Activate an environment with this repository installed before submitting.
# Run sbatch from this script's directory.
set -euo pipefail
mkdir -p generated log_files

nvidia-smi

# Run your application
echo "Running on GPU with float32 precision"
python run_timing.py --device gpu --dtype float32

echo "Running on GPU with float64 precision"
python run_timing.py --device gpu --dtype float64

echo "Running on CPU with float32 precision"
python run_timing.py --device cpu --dtype float32

echo "Running on CPU with float64 precision"
python run_timing.py --device cpu --dtype float64
