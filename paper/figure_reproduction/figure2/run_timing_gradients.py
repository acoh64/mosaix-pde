import time
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from mosaix_pde.numerics.domains import Domain
from mosaix_pde.numerics.equations.cahn_hilliard import CahnHilliard2DPeriodic
from mosaix_pde.numerics.functions.legendre import TestChemicalPotential
from mosaix_pde.numerics.solvers import SemiImplicitFourierSpectral
from mosaix_pde.pde_model import PDEModel


def run_timing_fwd(num_params, num_runs=10, N=128, device="cpu", dtype="float32"):

    # Configure JAX device
    if device == "gpu":
        jax.config.update("jax_platform_name", "gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")

    # Configure JAX dtype
    jax.config.update("jax_enable_x64", dtype == "float64")

    solve_times = {n_params: [] for n_params in num_params}

    for n_params in num_params:
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

        chem_pot_model = TestChemicalPotential(
            jnp.zeros(n_params), lambda x: jnp.log(x / (1.0 - x))
        )

        solver_parameters = {
            "A": 0.5,
        }

        key = random.PRNGKey(0)
        u0 = 0.5 * jnp.ones((Nx, Ny)) + 0.01 * random.normal(key, (Nx, Ny))

        # @eqx.filter_value_and_grad
        @eqx.filter_jacfwd
        def loss(
            chem_pot_model,
            *,
            model=model,
            u0=u0,
            ts_save=ts_save,
            solver_parameters=solver_parameters,
            dt=dt,
        ):
            pde_parameters = {
                "kappa": 0.002,
                "mu": chem_pot_model,
                "D": lambda c: (1.0 - c) * c,
                "derivs": "fd",
            }
            solution = model.solve(
                pde_parameters,
                u0,
                ts_save,
                solver_parameters,
                dt0=dt,
            )
            return jnp.mean(solution**2)

        @eqx.filter_jit
        def loss_jit(chem_pot_model):
            return loss(chem_pot_model)

        for i in range(num_runs):
            start_time = time.perf_counter()
            res = loss_jit(chem_pot_model)
            jax.block_until_ready(res)
            end_time = time.perf_counter()
            if not all(
                np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(res)
            ):
                raise RuntimeError("Gradient returned non-finite values")
            solve_times[n_params].append(end_time - start_time)

    return solve_times


def run_timing_rev(num_params, num_runs=10, N=128, device="cpu", dtype="float32"):

    # Configure JAX device
    if device == "gpu":
        jax.config.update("jax_platform_name", "gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")

    # Configure JAX dtype
    jax.config.update("jax_enable_x64", dtype == "float64")

    solve_times = {n_params: [] for n_params in num_params}

    for n_params in num_params:
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

        chem_pot_model = TestChemicalPotential(
            jnp.zeros(n_params), lambda x: jnp.log(x / (1.0 - x))
        )

        solver_parameters = {
            "A": 0.5,
        }

        key = random.PRNGKey(0)
        u0 = 0.5 * jnp.ones((Nx, Ny)) + 0.01 * random.normal(key, (Nx, Ny))

        @eqx.filter_value_and_grad
        def loss(
            chem_pot_model,
            *,
            model=model,
            u0=u0,
            ts_save=ts_save,
            solver_parameters=solver_parameters,
            dt=dt,
        ):
            pde_parameters = {
                "kappa": 0.002,
                "mu": chem_pot_model,
                "D": lambda c: (1.0 - c) * c,
                "derivs": "fd",
            }
            solution = model.solve(
                pde_parameters,
                u0,
                ts_save,
                solver_parameters,
                dt0=dt,
                adjoint=dfx.RecursiveCheckpointAdjoint(),
            )
            return jnp.mean(solution**2)

        @eqx.filter_jit
        def loss_jit(chem_pot_model):
            return loss(chem_pot_model)

        for i in range(num_runs):
            start_time = time.perf_counter()
            res = loss_jit(chem_pot_model)
            jax.block_until_ready(res)
            end_time = time.perf_counter()
            if not all(
                np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(res)
            ):
                raise RuntimeError("Gradient returned non-finite values")
            solve_times[n_params].append(end_time - start_time)

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
        "--mode",
        type=str,
        choices=["forward", "reverse"],
        default="reverse",
        help="Mode of automatic differentiation to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run on: cpu or gpu",
    )
    parser.add_argument(
        "--spacing",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Use linear or logarithmic spacing for parameter sizes",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision: float32 or float64",
    )
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", help="Parameter counts")
    parser.add_argument("--grid-size", type=int, default=64)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.spacing == "linear":
        Ns = np.linspace(1, 10000, 10)
    else:
        Ns = np.logspace(0, 4, 10)
    Ns = args.sizes or [int(n) for n in Ns]

    if args.mode == "forward":
        solve_times = run_timing_fwd(
            Ns,
            num_runs=args.num_runs,
            N=args.grid_size,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        solve_times = run_timing_rev(
            Ns,
            num_runs=args.num_runs,
            N=args.grid_size,
            device=args.device,
            dtype=args.dtype,
        )
    print(solve_times)

    with open(
        args.output + f"_{args.mode}_{args.spacing}_{args.device}_{args.dtype}.json",
        "w",
    ) as f:
        json.dump(solve_times, f)
