# Paper figure reproduction

Code and data for Figures 2 and 3. Install the package from the repository root:

```sh
python -m pip install -e .
```

The scripts support CPU execution. GPU execution requires a CUDA-enabled JAX
installation and a compatible NVIDIA GPU. Timings depend on hardware and
software versions.

## Figure 2: solver and gradient benchmarks

From `paper/figure_reproduction/figure2`, replot the paper's saved measurements:

```sh
python plot_figure2.py
```

This writes `generated/figure2a.png`, `figure2b.png`, and `figure2c.png`.
`timing_figure.ipynb` provides the same plotting entry point. No GPU is needed.
The first measurement in each timing series is excluded as warm-up; subsequent
measurements are averaged. The paper combines these three panels into one figure.

The benchmark programs are:

| Panel | Program | Configuration |
| --- | --- | --- |
| a | `run_timing.py` | 1,000 semi-implicit Fourier steps, CPU/GPU, Float32/Float64 |
| b | `run_timing_gradients.py` | Forward/reverse differentiation through 1,000 steps, 64 × 64 grid, GPU, Float32/Float64 |
| c | `run_timing_timestepping.py` | Tsit5, ROCK2, and semi-implicit Fourier with PID step control, GPU, Float64 |

The paper's benchmarks used Intel Xeon Platinum 8562Y+ CPUs and an NVIDIA L40S
GPU, with one allocated CPU core per job. No explicit software thread limit
was set. Panel c integrates the Cahn–Hilliard equation to `t = 0.01` with
`rtol = 1e-7`, `atol = 1e-9`, and PID coefficients `(0.4, 0.3, 0)`.

Small CPU checks, retaining the full integration intervals:

```sh
python run_timing.py --sizes 16 --num-runs 2
python run_timing_gradients.py --sizes 2 --grid-size 16 --num-runs 2 --mode forward
python run_timing_gradients.py --sizes 2 --grid-size 16 --num-runs 2 --mode reverse
python run_timing_timestepping.py --sizes 16 --num-runs 2 --dtype float64 --solver Tsit5
python run_timing_timestepping.py --sizes 16 --num-runs 2 --dtype float64 --solver ROCK2JAX
python run_timing_timestepping.py --sizes 16 --num-runs 2 --dtype float64 --solver SemiImplicitFourierSpectral
```

Omit `--sizes`, `--grid-size`, and `--num-runs` to use the complete benchmark
sweeps. Use `--device gpu` for GPU runs and `--dtype float64` for double precision.
For panel b use `--spacing log`, as in the submission scripts. The full sweeps
can require substantial time and memory.

`submit_*.sh` contains the Slurm commands for the full sweeps. Activate an
environment with this repository installed, select a suitable partition/GPU,
and create the log directory **before** submission:

```sh
mkdir -p log_files
sbatch submit_solver_timing.sh
sbatch submit_timing_gradients.sh
sbatch submit_timing_timestepping.sh
```

New measurements go to `generated/`, preserving the paper's data in `results/`.
Each benchmark waits for JAX results before stopping its timer and reuses the
compiled callable across repetitions. The first repetition includes compilation.
To plot a complete new sweep, run `python plot_figure2.py --data-dir generated`.
The solver-comparison script also supports the supplementary Allen–Cahn cases
included in its submission file.

## Figure 3: random-action PDE environment

From `paper/figure_reproduction/figure3`:

```sh
python figure3.py
```

This runs the Gross–Pitaevskii environment on a 256 × 256 grid for 100 actions
and saves the panel sequence for steps 0, 10, …, 60 to `generated/figure3.png`.
The accompanying `generated/figure3.npz` stores the states, observations,
actions, trajectory, rewards, seed, and grid size. `figure3.ipynb` runs the
same function and displays the image.

For a short CPU check:

```sh
python figure3.py --grid-size 32 --steps 2
```

The default action seed is 0; change it with `--seed`. The paper used unseeded
random actions, so this reproduces the demonstration setup rather than its
exact trajectory. Reduced-grid checks are for exercising the code and do not
represent the paper's spatial resolution. Open each notebook from its own
folder so that it can import the adjacent Python script.
