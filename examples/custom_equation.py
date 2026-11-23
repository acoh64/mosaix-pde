"""Define and solve a custom periodic diffusion equation."""

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mosaix_pde import BaseEquation, Domain, PDEModel


class PeriodicDiffusion1D(BaseEquation):
    """Diffusion on a uniform one-dimensional periodic grid."""

    def __init__(self, domain, diffusivity):
        self.dx = domain.dx[0]
        self.diffusivity = diffusivity

    def rhs(self, state, t):
        left = jnp.roll(state, 1)
        right = jnp.roll(state, -1)
        laplacian = (left - 2 * state + right) / self.dx**2
        return self.diffusivity * laplacian


def main():
    domain = Domain(points=(64,), box=((0.0, 1.0),), units="dimensionless")
    (x,) = domain.axes()
    initial = jnp.exp(-0.5 * ((x - 0.5) / 0.06) ** 2)
    diffusivity = 0.05
    times = jnp.array([0.0, 0.02, 0.05])

    model = PDEModel(PeriodicDiffusion1D, domain, diffrax.Tsit5)
    solution = model.solve(
        parameters={"diffusivity": diffusivity},
        y0=initial,
        ts=times,
        dt0=0.2 * domain.dx[0] ** 2 / diffusivity,
    )

    means = solution.mean(axis=1)
    print("Mean concentration at each saved time:", means)
    assert jnp.allclose(means, initial.mean(), atol=1e-6, rtol=0)
    assert solution[-1].max() < solution[0].max()

    _, ax = plt.subplots()
    for time, profile in zip(times, solution):
        ax.plot(x, profile, label=f"t = {float(time):.2f}")
    ax.set(xlabel="Position", ylabel="Concentration")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
