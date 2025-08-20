import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Callable, Dict, Any


def _lap_2nd_2D(u, hx, hy):
    return (jnp.roll(u, -1, 0) - 2 * u + jnp.roll(u, 1, 0)) / hx**2 + (
        jnp.roll(u, -1, 1) - 2 * u + jnp.roll(u, 1, 1)
    ) / hy**2


def _lap_2nd_3D(u, hx, hy, hz):
    return (
        (jnp.roll(u, -1, 0) - 2 * u + jnp.roll(u, 1, 0)) / hx**2
        + (jnp.roll(u, -1, 1) - 2 * u + jnp.roll(u, 1, 1)) / hy**2
        + (jnp.roll(u, -1, 2) - 2 * u + jnp.roll(u, 1, 2)) / hz**2
    )


def _gradx_c2f(a, hy):  # center -> y-face (i,j+1/2)
    return (jnp.roll(a, -1, 0) - a) / hy


def _grady_c2f(a, hx):  # center -> x-face (i+1/2,j), 2nd-order at the face
    return (jnp.roll(a, -1, 1) - a) / hx


def _gradz_c2f(a, hz):  # center -> z-face (i,j+1/2,k+1/2)
    return (jnp.roll(a, -1, 2) - a) / hz


def _avgx_c2f(a):  # average to y-face
    return 0.5 * (a + jnp.roll(a, -1, 0))


def _avgy_c2f(a):  # average to x-face
    return 0.5 * (a + jnp.roll(a, -1, 1))


def _avgz_c2f(a):  # average to z-face (i,j+1/2,k+1/2)
    return 0.5 * (a + jnp.roll(a, -1, 2))


def _divx_f2c(Fy, hy):  # y-face -> center divergence
    return (Fy - jnp.roll(Fy, 1, 0)) / hy


def _divy_f2c(Fx, hx):  # x-face -> center divergence
    return (Fx - jnp.roll(Fx, 1, 1)) / hx


def _divz_f2c(Fy, hz):  # z-face -> center divergence
    return (Fy - jnp.roll(Fy, 1, 2)) / hz


def _gradx_c(a, hx):
    return 0.5 * (jnp.roll(a, -1, 0) - jnp.roll(a, 1, 0)) / hx


def _grady_c(a, hy):
    return 0.5 * (jnp.roll(a, -1, 1) - jnp.roll(a, 1, 1)) / hy


def _gradz_c(a, hz):
    return 0.5 * (jnp.roll(a, -1, 2) - jnp.roll(a, 1, 2)) / hz


def _grad2x_c(a, hx):
    return (jnp.roll(a, -1, 0) - 2 * a + jnp.roll(a, 1, 0)) / hx**2


def _grad2y_c(a, hy):
    return (jnp.roll(a, -1, 1) - 2 * a + jnp.roll(a, 1, 1)) / hy**2


def _grad2z_c(a, hz):
    return (jnp.roll(a, -1, 2) - 2 * a + jnp.roll(a, 1, 2)) / hz**2


def _grad2xy_c(a, hx, hy):
    return (
        jnp.roll(jnp.roll(a, -1, 0), -1, 1)
        + jnp.roll(jnp.roll(a, 1, 0), 1, 1)
        - jnp.roll(jnp.roll(a, -1, 0), 1, 1)
        - jnp.roll(jnp.roll(a, 1, 0), -1, 1)
    ) / (4.0 * hx * hy)
