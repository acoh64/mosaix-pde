import pytest
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt

from pde_opt.numerics.domains import Domain
from pde_opt.numerics.equations import CahnHilliard2DPeriodic, AllenCahn2DPeriodic
from pde_opt.numerics.solvers import SemiImplicitFourierSpectral

# Force JAX to use CPU for tests
jax.config.update('jax_platforms', 'cpu')

def test_1d_cahn_hilliard():
    Nx, Ny = 256, 1
    Lx = 0.01 * Nx
    Ly = 0.01 * Ny
    domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

    t_start = 0.0
    t_final = 10.0
    dt = 0.00005

    ts_save = jnp.linspace(t_start, t_final, 200)

    kappa = 0.002

    eq = CahnHilliard2DPeriodic(
        domain,
        kappa,
        lambda c: c**3 - c,
        lambda c: jnp.ones_like(c),
        derivs="fd"
    )

    solver = SemiImplicitFourierSpectral(0.5, eq.fourier_symbol, eq.fft, eq.ifft)

    u0 = jnp.ones((Nx, Ny))
    u0 = u0.at[:Nx//2, :].set(-1.0)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(jax.jit(lambda t, y, args: eq.rhs(y, t))),
        solver,
        t0=t_start,
        t1=t_final,
        dt0=dt,
        y0=u0,
        saveat=diffrax.SaveAt(ts=ts_save),
        max_steps=1000000,
    )

    analytic_solution = jnp.tanh(domain.axes()[0] / np.sqrt(2 * kappa))
    np.testing.assert_allclose(solution.ys[-1].squeeze()[Nx//4:3*Nx//4], analytic_solution[Nx//4:3*Nx//4], rtol=1e-3, atol=1e-3)

def test_1d_allen_cahn():
    Nx, Ny = 256, 1
    Lx = 0.01 * Nx
    Ly = 0.01 * Ny
    domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

    t_start = 0.0
    t_final = 10.0
    dt = 0.00005
    ts_save = jnp.linspace(t_start, t_final, 200)

    kappa = 0.002

    eq = AllenCahn2DPeriodic(
        domain,
        kappa,
        lambda c: c**3 - c,
        lambda c: jnp.ones_like(c),
        derivs="fd"
    )

    solver= diffrax.Tsit5()

    u0 = jnp.ones((Nx, Ny))
    u0 = u0.at[:Nx//2, :].set(-1.0)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(jax.jit(lambda t, y, args: eq.rhs(y, t))),
        solver,
        t0=t_start,
        t1=t_final,
        dt0=dt,
        y0=u0,
        saveat=diffrax.SaveAt(ts=ts_save),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=1000000,
    )

    analytic_solution = np.tanh(domain.axes()[0] / np.sqrt(2 * kappa))
    np.testing.assert_allclose(solution.ys[-1].squeeze()[Nx//4:3*Nx//4], analytic_solution[Nx//4:3*Nx//4], rtol=1e-3, atol=1e-3)