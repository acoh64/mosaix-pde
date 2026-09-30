import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from mosaix_pde.numerics.domains import Domain
from mosaix_pde.numerics.equations.cahn_hilliard import CahnHilliard2DPeriodic
from mosaix_pde.numerics.solvers import SemiImplicitFourierSpectral
from mosaix_pde.pde_model import PDEModel


def run_timing(Ns, num_runs=10, device="cpu", dtype="float32"):

    # Configure JAX device
    if device == "gpu":
        jax.config.update("jax_platform_name", "gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")

    # Configure JAX dtype
    jax.config.update("jax_enable_x64", dtype == "float64")

    solve_times = {N: [] for N in Ns}

    for N in Ns:
        Nx, Ny = N, N
        Lx = 0.01 * Nx
        Ly = 0.01 * Ny
        domain = Domain(
            (Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless"
        )

        model = PDEModel(CahnHilliard2DPeriodic, domain, SemiImplicitFourierSpectral)

        t_start = 0.0
        t_final = 0.001
        dt = 0.000001
        ts_save = jnp.linspace(t_start, t_final, 2)

        pde_parameters = {
            "kappa": 0.002,
            "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
            "D": lambda c: (1.0 - c) * c,
            "derivs": "fd",
        }

        solver_parameters = {
            "A": 0.5,
        }

        key = random.PRNGKey(0)
        u0 = 0.5 * jnp.ones((Nx, Ny)) + 0.01 * random.normal(key, (Nx, Ny))

        @eqx.filter_jit
        def solve_once(
            model=model,
            pde_parameters=pde_parameters,
            u0=u0,
            ts_save=ts_save,
            solver_parameters=solver_parameters,
            dt=dt,
        ):
            return model.solve(
                pde_parameters,
                u0,
                ts_save,
                solver_parameters,
                dt0=dt,
            )

        for i in range(num_runs):
            print(f"Running run {i} of {num_runs} for N = {N}")
            start_time = time.perf_counter()
            solution = solve_once()
            jax.block_until_ready(solution)
            end_time = time.perf_counter()
            if not np.isfinite(np.asarray(solution)).all():
                raise RuntimeError("Solver returned non-finite values")
            solve_times[N].append(end_time - start_time)

    return solve_times


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="generated/solve_times",
        help="Path to save timing results JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run on: cpu or gpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision: float32 or float64",
    )
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+", help="Grid sizes")
    args = parser.parse_args()
    if args.num_runs < 2 or (args.sizes and min(args.sizes) < 2):
        parser.error("Use at least two runs (including warm-up) and sizes >= 2")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    Ns = args.sizes or [16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"Running on {args.device} with {args.dtype} precision")
    solve_times = run_timing(
        Ns, num_runs=args.num_runs, device=args.device, dtype=args.dtype
    )
    print(solve_times)

    with open(args.output + f"_{args.device}_{args.dtype}.json", "w") as f:
        json.dump(solve_times, f)
