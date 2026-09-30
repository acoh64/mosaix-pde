#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o log_files/timing_gradients_%j.out
#SBATCH -t 6:00:00

# Set up environment
# Activate an environment with this repository installed before submitting.
# Run sbatch from this script's directory.
set -euo pipefail
mkdir -p generated log_files

nvidia-smi

# Run your application
echo "Running forward on GPU with float32 precision"
python run_timing_gradients.py --mode forward --spacing log --dtype float32 --device gpu

echo "Running forward on GPU with float64 precision"
python run_timing_gradients.py --mode forward --spacing log --dtype float64 --device gpu

echo "Running reverseon GPU with float32 precision"
python run_timing_gradients.py --mode reverse --spacing log --dtype float32 --device gpu

echo "Running reverse on GPU with float64 precision"
python run_timing_gradients.py --mode reverse --spacing log --dtype float64 --device gpu
