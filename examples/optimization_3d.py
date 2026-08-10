"""Fit constitutive functions in a three-dimensional Cahn--Hilliard model.

This is the standalone script counterpart to ``docs/notebooks/optimization_3D.ipynb``.
The three-dimensional optimization performs repeated PDE solves and can be
computationally expensive; a GPU is recommended.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mosaix_pde import (
    CahnHilliard3DPeriodic,
    ChemicalPotentialLegendrePolynomials,
    DiffusionLegendrePolynomials,
    Domain,
    PDEModel,
    SemiImplicitFourierSpectral,
)


def visualize_3d_solution(solution, ts, time_slices, z_slices):
    """Plot two-dimensional slices through a three-dimensional solution."""
    _, axes = plt.subplots(len(z_slices), len(time_slices))
    for column, time_index in enumerate(time_slices):
        axes[0, column].set_title(f"t = {ts[time_index]:.3f}")
    for row, z_index in enumerate(z_slices):
        axes[row, 0].set_ylabel(f"z = {z_index}")
        for column, time_index in enumerate(time_slices):
            axes[row, column].imshow(
                solution[time_index][:, :, z_index], vmin=0.0, vmax=1.0
            )
    plt.tight_layout()
    plt.show()


def main():
    nx = ny = nz = 32
    lx = ly = lz = 0.01 * nx
    domain = Domain(
        (nx, ny, nz),
        (
            (-lx / 2, lx / 2),
            (-ly / 2, ly / 2),
            (-lz / 2, lz / 2),
        ),
        "dimensionless",
    )
    opt_model = PDEModel(
        equation_type=CahnHilliard3DPeriodic,
        domain=domain,
        solver_type=SemiImplicitFourierSpectral,
    )

    parameters = {
        "kappa": 0.002,
        "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
        "D": lambda c: 0.15 * jnp.ones_like(c),
    }
    solver_parameters = {"A": 0.5}

    key = jax.random.PRNGKey(0)
    y0 = jnp.clip(0.01 * jax.random.normal(key, (nx, ny, nz)) + 0.5, 0.0, 1.0)
    ts = jnp.linspace(0.0, 0.2, 100)
    solution = opt_model.solve(
        parameters,
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )

    z_slices = [0, 10, 25]
    time_slices = [0, 10, 20, 50, -1]
    visualize_3d_solution(solution, ts, time_slices, z_slices)

    chemical_potential = ChemicalPotentialLegendrePolynomials(
        jnp.zeros(6), lambda x: jnp.log(x / (1.0 - x))
    )
    diffusivity = DiffusionLegendrePolynomials(jnp.log(0.05) * jnp.ones(1))
    data = {"ys": solution, "ts": ts}
    inds = [[30, 40, 50], [50, 60, 70], [70, 80, 90]]
    init_params = {"mu": chemical_potential, "D": diffusivity}
    static_params = {"kappa": 0.002}
    weights = {
        "mu": ChemicalPotentialLegendrePolynomials(jnp.array([0, 2, 6, 12, 20, 30])),
        "D": DiffusionLegendrePolynomials(jnp.array([0])),
    }

    initial_solution = opt_model.solve(
        {**init_params, **static_params},
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )
    visualize_3d_solution(initial_solution, ts, time_slices, z_slices)

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

    concentrations = jnp.linspace(0.01, 0.99, 100)
    mu_ground_truth = parameters["mu"](concentrations)
    diffusivity_ground_truth = parameters["D"](concentrations)
    mu_initial = init_params["mu"](concentrations)
    mu_learned = result["mu"](concentrations)
    diffusivity_initial = init_params["D"](concentrations)
    diffusivity_learned = result["D"](concentrations)

    midpoint = jnp.argmin(jnp.abs(concentrations - 0.5))
    mu_initial -= mu_initial[midpoint]
    mu_learned -= mu_learned[midpoint]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("Chemical Potential")
    axes[1].set_title("Diffusivity")
    axes[0].plot(concentrations, mu_ground_truth, label="Ground truth")
    axes[0].plot(concentrations, mu_initial, label="Initial")
    axes[0].plot(concentrations, mu_learned, label="Learned", linestyle="--")
    axes[1].plot(concentrations, diffusivity_ground_truth, label="Ground truth")
    axes[1].plot(concentrations, diffusivity_initial, label="Initial")
    axes[1].plot(concentrations, diffusivity_learned, label="Learned", linestyle="--")
    axes[0].set_xlabel("c")
    axes[1].set_xlabel("c")
    axes[0].set_ylabel("mu")
    axes[1].set_ylabel("D")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    optimized_solution = opt_model.solve(
        result,
        y0,
        ts,
        solver_parameters,
        dt0=0.000001,
        max_steps=1000000,
    )
    print("Ground truth solution:")
    visualize_3d_solution(solution, ts, time_slices, z_slices)
    print("Optimized solution:")
    visualize_3d_solution(optimized_solution, ts, time_slices, z_slices)


if __name__ == "__main__":
    main()
