#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o log_files/timing_timestepping_%j.out
#SBATCH -t 6:00:00

# Set up environment
# Activate an environment with this repository installed before submitting.
# Run sbatch from this script's directory.
set -euo pipefail
mkdir -p generated log_files

nvidia-smi

# Run your application
echo "Running Cahn-Hilliard on GPU with float64 precision and Tsit5 solver"
python run_timing_timestepping.py --device gpu --dtype float64 --solver Tsit5 --equation cahn_hilliard

echo "Running Cahn-Hilliard on GPU with float64 precision and ROCK2JAX solver"
python run_timing_timestepping.py --device gpu --dtype float64 --solver ROCK2JAX --equation cahn_hilliard

echo "Running Cahn-Hilliard on GPU with float64 precision and SemiImplicitFourierSpectral solver"
python run_timing_timestepping.py --device gpu --dtype float64 --solver SemiImplicitFourierSpectral --equation cahn_hilliard

echo "Running Allen-Cahn on GPU with float64 precision and Tsit5 solver"
python run_timing_timestepping.py --device gpu --dtype float64 --solver Tsit5 --equation allen_cahn

echo "Running Allen-Cahn on GPU with float64 precision and ROCK2JAX solver"
python run_timing_timestepping.py --device gpu --dtype float64 --solver ROCK2JAX --equation allen_cahn
