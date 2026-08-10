"""Learn a Cahn--Hilliard chemical potential with a periodic CNN.

This is the standalone script counterpart to
``docs/notebooks/optimization_neural_network.ipynb``. It performs repeated PDE
solves during training and may take several minutes, especially on CPU.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mosaix_pde import (
    CahnHilliard2DPeriodic,
    Domain,
    PDEModel,
    PeriodicCNN,
    SemiImplicitFourierSpectral,
)


def plot_snapshots(solution, indices, *, row=None, label=None):
    """Plot selected snapshots in either a new figure or an existing row."""
    if row is None:
        _, row = plt.subplots(1, len(indices))
    for axis, index in zip(row, indices):
        axis.imshow(solution[index], vmin=0.0, vmax=1.0)
    if label is not None:
        row[0].set_ylabel(label)
    return row


def main():
    nx = ny = 32
    lx = ly = 0.01 * nx
    domain = Domain((nx, ny), ((-lx / 2, lx / 2), (-ly / 2, ly / 2)), "dimensionless")
    opt_model = PDEModel(
        equation_type=CahnHilliard2DPeriodic,
        domain=domain,
        solver_type=SemiImplicitFourierSpectral,
    )

    parameters = {
        "kappa": 0.002,
        "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
        "D": lambda c: jnp.ones_like(c),
    }
    solver_parameters = {"A": 0.5}

    key = jax.random.PRNGKey(0)
    y0 = jnp.clip(0.01 * jax.random.normal(key, (nx, ny)) + 0.5, 0.0, 1.0)
    ts = jnp.linspace(0.0, 0.02, 100)
    solution = opt_model.solve(
        parameters,
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )

    snapshot_indices = [0, 10, 20, 50, -1]
    axes = plot_snapshots(solution, snapshot_indices)
    for axis, index in zip(axes, snapshot_indices):
        axis.set_title(f"t = {ts[index]:.3f}")
    plt.tight_layout()
    plt.show()

    model = PeriodicCNN(
        in_channels=1,
        hidden_channels=(32, 64, 64),
        out_channels=1,
        kernel_size=3,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.filter_jit(model)

    data = {"ys": solution, "ts": ts}
    inds = [[30, 40, 50], [50, 60, 70], [70, 80, 90]]
    init_params = {"mu": model}
    static_params = {
        "kappa": 0.002,
        "D": lambda c: jnp.ones_like(c),
    }
    weights = {"mu": None}

    initial_solution = opt_model.solve(
        {**init_params, **static_params},
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )
    plot_snapshots(initial_solution, snapshot_indices)
    plt.tight_layout()
    plt.show()

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

    concentrations = jnp.linspace(0.01, 0.99, 100)
    ground_truth = parameters["mu"](concentrations)
    initial_values = []
    learned_values = []
    for concentration in concentrations:
        field = concentration * jnp.ones_like(y0)
        initial_values.append(jnp.mean(init_params["mu"](field)))
        learned_values.append(jnp.mean(result["mu"](field)))

    initial_values = jnp.array(initial_values)
    learned_values = jnp.array(learned_values)
    midpoint = jnp.argmin(jnp.abs(concentrations - 0.5))
    initial_values -= initial_values[midpoint]
    learned_values -= learned_values[midpoint]

    _, axis = plt.subplots()
    axis.plot(concentrations, ground_truth, label="Ground truth")
    axis.plot(concentrations, initial_values, label="Initial")
    axis.plot(concentrations, learned_values, label="Learned")
    axis.set_xlabel("c")
    axis.set_ylabel("mu")
    axis.legend()
    plt.show()

    optimized_solution = opt_model.solve(
        result,
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )
    _, axes = plt.subplots(2, 5)
    plot_snapshots(solution, snapshot_indices, row=axes[0], label="Ground truth")
    plot_snapshots(optimized_solution, snapshot_indices, row=axes[1], label="Learned")
    for axis, index in zip(axes[0], snapshot_indices):
        axis.set_title(f"t = {ts[index]:.3f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
