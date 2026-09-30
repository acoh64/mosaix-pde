from contextlib import contextmanager

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaix_pde import ROCK2JAX


@contextmanager
def enable_x64(value=True):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", value)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("x64", [False, True])
def test_rock2_coefficients_follow_precision(x64):
    with enable_x64(x64):
        solver = ROCK2JAX()
        expected = jnp.float64 if x64 else jnp.float32
        assert solver.recf.dtype == expected
        assert solver.fp1_tab.dtype == expected
        assert solver.fp2_tab.dtype == expected


def test_rock2_stiff_decay():
    with enable_x64():
        rates = jnp.array([-1.0, -100.0])
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: rates * y),
            ROCK2JAX(),
            t0=0.0,
            t1=0.1,
            dt0=1e-4,
            y0=jnp.ones(2),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
            max_steps=100000,
        )
        np.testing.assert_allclose(
            solution.ys[-1],
            np.exp(np.asarray(rates) * 0.1),
            rtol=1e-4,
            atol=1e-8,
        )
