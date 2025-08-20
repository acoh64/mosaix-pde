import dataclasses

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import dataclasses
import numpy as np
import diffrax as dfx

from .utils.derivatives import _gradx_c, _grady_c, _grad2x_c, _grad2y_c, _grad2xy_c

Array = jax.Array

@dataclasses.dataclass
class Shape:
    """Sets up a geometry/shape for solving PDE on with smoothed boundary method.

    The user creates a shape by providing a binary representation and an optional smoothing parameter.
    """
    
    binary: Array
    dx: Optional[Tuple[float, float]] = (1.0, 1.0)
    smooth_params: Optional[Tuple[float, float, float]] = (1.0, 0.1, 1.0)
    
    def __post_init__(self):
        self.smooth = self.smooth_shape()
        self.smooth = jnp.where(self.smooth < 0.001, 0.001, self.smooth)
        self.smooth = jnp.where(self.smooth > 0.99, 1.0, self.smooth)
        
    def smooth_shape(self) -> Array:
        """Smooths the shape using the Allen-Cahn equation with curvature minimization."""

        epsilon, dt, tf = self.smooth_params

        potential = lambda u: 18. / epsilon * u * (1. - u) * (1. - 2.*u)

        @jax.jit
        def rhs(t, u, args):
            gradx = _gradx_c(u, self.dx[0])
            grady = _grady_c(u, self.dx[1])
            grad2x = _grad2x_c(u, self.dx[0])
            grad2y = _grad2y_c(u, self.dx[1])
            grad2xy = _grad2xy_c(u, self.dx[0], self.dx[1])
            grad_norm_sq = gradx**2 + grady**2
            grad_norm_sq = jnp.where(grad_norm_sq < 1e-7, 1.0, grad_norm_sq)
            norm_laplace = (grad2x * gradx**2 + 2.0 * grad2xy * gradx * grady + grad2y * grady**2) / grad_norm_sq
            return 2.0 * norm_laplace - potential(u) / epsilon
        
        solution = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            t0=0.,
            t1=tf,
            dt0=dt,
            y0=self.binary,
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            saveat=dfx.SaveAt(t1=True),
            max_steps=1000000,
        )

        return solution.ys[-1]