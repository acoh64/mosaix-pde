"""Run the Cahn--Hilliard solving and training example from the README.

This example first generates a trajectory and then fits a periodic convolutional
neural network for the chemical-potential term. Training may take several minutes.
"""

import jax
import jax.numpy as jnp

from mosaix_pde import (
    CahnHilliard2DPeriodic,
    Domain,
    PDEModel,
    PeriodicCNN,
    SemiImplicitFourierSpectral,
)


def main():
    nx = ny = 128
    lx = ly = 0.01 * nx
    domain = Domain((nx, ny), ((-lx / 2, lx / 2), (-ly / 2, ly / 2)), "dimensionless")
    opt_model = PDEModel(
        equation_type=CahnHilliard2DPeriodic,
        domain=domain,
        solver_type=SemiImplicitFourierSpectral,
    )

    params = {
        "kappa": 0.002,
        "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
        "D": lambda c: c * (1.0 - c),
    }
    solver_parameters = {"A": 0.5}

    key = jax.random.PRNGKey(0)
    y0 = jnp.clip(0.01 * jax.random.normal(key, (nx, ny)) + 0.5, 0.0, 1.0)
    ts = jnp.linspace(0.0, 0.02, 100)
    solution = opt_model.solve(
        params,
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )

    data = {"ys": solution, "ts": ts}
    model = PeriodicCNN(
        in_channels=1,
        hidden_channels=(32, 64, 64),
        out_channels=1,
        kernel_size=3,
        key=jax.random.PRNGKey(0),
    )
    init_params = {"mu": model}
    static_params = {
        "kappa": 0.002,
        "D": lambda c: c * (1.0 - c),
    }
    weights = {"mu": None}
    inds = [[30, 40, 50], [50, 60, 70], [70, 80, 90]]

    result = opt_model.train(
        data,
        inds,
        init_params,
        static_params,
        solver_parameters,
        weights,
        0.0,
        method="mse",
        max_steps=100,
    )
    print(result)


if __name__ == "__main__":
    main()
