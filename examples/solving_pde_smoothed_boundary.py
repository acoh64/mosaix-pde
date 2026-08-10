"""Solve an Allen--Cahn equation on a smoothed, nonrectangular boundary.

This is the standalone script counterpart to
``docs/notebooks/solving_pde_smoothed_boundary.ipynb``. It uses the smile image
shipped with the documentation and opens Matplotlib visualizations.
"""

from pathlib import Path

import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib import animation
from PIL import Image

from mosaix_pde import AllenCahn2DSmoothedBoundary, Domain, PDEModel, Shape


def main():
    image_path = Path(__file__).resolve().parents[1] / "docs" / "cool_smile.png"
    image = Image.open(image_path).convert("L")
    binary_mask = jnp.array((np.array(image) > 128).astype(np.float32))

    shape = Shape(
        binary=binary_mask,
        dx=(1.0, 1.0),
        smooth_epsilon=3.0,
        smooth_curvature=0.008,
        smooth_dt=0.01,
        smooth_tf=100.0,
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    ax1.imshow(shape.binary, cmap="gray")
    ax1.set_title("Binary Shape")
    ax1.axis("equal")
    ax2.imshow(shape.smooth, cmap="gray")
    ax2.set_title("Smoothed Shape")
    ax2.axis("equal")
    ax3.imshow(shape.binary[480:520, 180:220], cmap="gray")
    ax3.set_title("Binary Shape (Zoomed)")
    ax3.axis("equal")
    ax4.imshow(shape.smooth[480:520, 180:220], cmap="gray")
    ax4.set_title("Smoothed Shape (Zoomed)")
    ax4.axis("equal")
    plt.tight_layout()
    plt.show()

    shape.get_shape_modes(N=36)
    fig, axes = plt.subplots(6, 6, figsize=(10, 10))
    for i, axis in enumerate(axes.flatten()):
        axis.imshow(shape.shape_basis[:, :, i], cmap="RdBu")
        axis.axis("off")
    plt.tight_layout()
    plt.show()

    nx, ny = binary_mask.shape
    lx = 0.01 * nx
    ly = 0.01 * ny
    domain = Domain(
        (nx, ny),
        ((-lx / 2, lx / 2), (-ly / 2, ly / 2)),
        "dimensionless",
        shape,
    )
    model = PDEModel(AllenCahn2DSmoothedBoundary, domain, dfx.Tsit5)
    ts_save = jnp.linspace(0.0, 20.0, 200)

    pde_parameters = {
        "kappa": 0.002,
        "f": lambda c: (
            c * jnp.log(c) + (1.0 - c) * jnp.log(1.0 - c) + 3.0 * c * (1.0 - c) + 0.059
        ),
        "mu": lambda c: jnp.log(c / (1.0 - c)) + 3.0 * (1.0 - 2.0 * c),
        "R": lambda c: (1.0 - c) * c,
        "theta": lambda t: jnp.pi / 2.0,
        "derivs": "fd",
    }

    key = random.PRNGKey(0)
    u0 = 0.5 * jnp.ones((nx, ny)) + 0.1 * random.normal(key, (nx, ny))
    solution = model.solve(
        pde_parameters,
        u0,
        ts_save,
        stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    frames = []
    for i in range(0, len(solution), 10):
        image = ax.imshow(
            solution[i] * domain.geometry.binary,
            animated=True,
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
    _animation = animation.ArtistAnimation(fig, frames, interval=200, blit=False)
    ax.set_title("Allen--Cahn Evolution with Smoothed Boundary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
