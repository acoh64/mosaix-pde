"""
This module contains the Domain class, which is used to set up a simulation domain for the model.
"""

import dataclasses
from typing import Optional, Tuple
import jax
import jax.numpy as jnp

from .shapes import Shape


@dataclasses.dataclass
class Domain:
    """Sets up a simulation domain for the model.

    The following information is stored in a Domain:
    -- `points[i]` is the number of collocation points in the i'th dimension
    -- `dx[i]`     is the spacing between each collocation point in the i'th dimension
    -- `box[i]`    is the bounds of the simulation box in the i'th dimension
    -- `units`     are the length units these. values are stored in
    """

    points: Tuple[int, ...]
    box: Tuple[Tuple[float, float], ...]
    units: str
    geometry: Optional[Shape] = None

    def __post_init__(self):
        self.dx = tuple(
            (up_bound - low_bound) / points
            for (low_bound, up_bound), points in zip(self.box, self.points)
        )
        self.L = tuple((up_bound - low_bound) for (low_bound, up_bound) in self.box)

    def axes(self) -> Tuple[jax.Array, ...]:
        return tuple(
            jnp.linspace(low_bound + step / 2, up_bound - step / 2, num=points)
            for (low_bound, up_bound), points, step in zip(
                self.box, self.points, self.dx
            )
        )

    def fft_axes(self) -> Tuple[jax.Array, ...]:
        return tuple(
            jnp.fft.fftfreq(points, step) for points, step in zip(self.points, self.dx)
        )

    def rfft_axes(self) -> Tuple[jax.Array, ...]:
        return tuple(
            jnp.fft.rfftfreq(points, step) for points, step in zip(self.points, self.dx)
        )

    def mesh(self) -> Tuple[jax.Array, ...]:
        axes = self.axes()
        return tuple(jnp.meshgrid(*axes, indexing="ij"))

    def fft_mesh(self) -> Tuple[jax.Array, ...]:
        fft_axes = self.fft_axes()
        return tuple(jnp.meshgrid(*fft_axes, indexing="ij"))

    def rfft_mesh(self) -> Tuple[jax.Array, ...]:
        rfft_axes = self.rfft_axes()
        return tuple(jnp.meshgrid(*rfft_axes, indexing="ij"))

    def __str__(self):
        return f"Domain with bounds {self.box} with units of {self.units} and {self.points} collocation points."
