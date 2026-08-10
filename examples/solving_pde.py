"""Solve the periodic two-dimensional Cahn--Hilliard equation.

This is the standalone script counterpart to ``docs/notebooks/solving_pde.ipynb``.
The example opens an animated Matplotlib visualization after the solve completes.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib import animation

from mosaix_pde import (
    CahnHilliard2DPeriodic,
    Domain,
    PDEModel,
    SemiImplicitFourierSpectral,
)


def main():
    nx = ny = 128
    lx = 0.01 * nx
    ly = 0.01 * ny
    domain = Domain((nx, ny), ((-lx / 2, lx / 2), (-ly / 2, ly / 2)), "dimensionless")

    model = PDEModel(CahnHilliard2DPeriodic, domain, SemiImplicitFourierSpectral)
    ts_save = jnp.linspace(0.0, 0.2, 200)

    pde_parameters = {
        "kappa": 0.002,
        "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
        "D": lambda c: (1.0 - c) * c,
        "derivs": "fd",
    }
    solver_parameters = {"A": 0.5}

    key = random.PRNGKey(0)
    u0 = 0.5 * jnp.ones((nx, ny)) + 0.01 * random.normal(key, (nx, ny))
    solution = model.solve(pde_parameters, u0, ts_save, solver_parameters)

    fig, ax = plt.subplots(figsize=(4, 4))
    frames = []
    for i in range(0, len(solution), 2):
        image = ax.imshow(
            solution[i],
            animated=True,
            cmap="RdBu",
            vmin=0.0,
            vmax=1.0,
            extent=[
                domain.box[0][0],
                domain.box[0][1],
                domain.box[1][0],
                domain.box[1][1],
            ],
        )
        frames.append([image])

    # Keep blitting disabled for compatibility with native GUI backends on macOS.
    _animation = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
    ax.set_title("Cahn--Hilliard Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
