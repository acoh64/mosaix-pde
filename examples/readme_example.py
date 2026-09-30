"""Run the Cahn--Hilliard solving and training example from the README.

This example first generates a trajectory and then fits a periodic convolutional
neural network for the chemical-potential term. A small grid and 13-parameter
pointwise network keep the CPU workload modest. The script reports trajectory
prediction error before and after a bounded least-squares fit.
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
    nx = ny = 32
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
    y0 = jnp.clip(0.1 * jax.random.normal(key, (nx, ny)) + 0.5, 0.01, 0.99)
    ts = jnp.linspace(0.0, 0.002, 5)
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
        hidden_channels=(4,),
        out_channels=1,
        kernel_size=1,
        key=jax.random.PRNGKey(0),
    )
    init_params = {"mu": model}
    static_params = {
        "kappa": 0.002,
        "D": lambda c: c * (1.0 - c),
    }
    weights = {"mu": None}
    inds = [[0, 1, 2], [2, 3, 4]]

    def trajectory_mse(parameters):
        predicted = opt_model.solve(parameters, y0, ts, solver_parameters)
        return jnp.mean((predicted[1:] - solution[1:]) ** 2)

    initial_mse = float(trajectory_mse({**init_params, **static_params}))
    result = opt_model.train(
        data,
        inds,
        init_params,
        static_params,
        solver_parameters,
        weights,
        0.0,
        method="least_squares",
        max_steps=20,
    )
    final_mse = float(trajectory_mse(result))
    print(f"Trajectory MSE before training: {initial_mse:.3e}")
    print(f"Trajectory MSE after training:  {final_mse:.3e}")
    print(f"Error reduction: {initial_mse / final_mse:.1f}x")


if __name__ == "__main__":
    main()
