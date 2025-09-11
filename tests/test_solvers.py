import pytest
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt

from pde_opt.numerics.domains import Domain
from pde_opt.numerics.equations import CahnHilliard2DPeriodic, AllenCahn2DPeriodic, GPE2DTSControl
from pde_opt.numerics.utils.initialization_utils import initialize_Psi
from pde_opt.numerics.solvers import SemiImplicitFourierSpectral, StrangSplitting
from pde_opt import PDEModel

# Force JAX to use CPU for tests
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)

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

def test_2d_gross_pitaevskii():

    def density(psi):
        return jnp.abs(psi)**2

    atoms = 5e5
    hbar = 1.05e-34 #J*s
    omega = 2*jnp.pi*10 #1/s
    omega_z = jnp.sqrt(8)*omega
    epsilon = 0.0
    mass = 3.8175406e-26 #kg (atomic mass of sodium-23)
    a0 = 5.29177210903e-11
    a_s = 100*a0
    N = 128

    x_s = jnp.sqrt(hbar/(mass*omega))
    t_s = 1/omega

    # Length of the x and y axes in meters
    Lx = 150e-6 #meters
    Ly = 150e-6 #meters

    # Get dimensionless variables
    Lx_ = Lx/x_s
    Ly_ = Ly/x_s

    # Get k
    k = 4*jnp.pi*a_s*atoms*jnp.sqrt((mass*omega_z)/(2*jnp.pi*hbar))

    epsilon = 0.0

    t_start = 0.0
    t_final = 0.1
    dt = 1e-5

    t_start_ = t_start/t_s
    t_final_ = t_final/t_s
    dt_ = dt/t_s

    domain_ = Domain((N,N,), ((-Lx_/2, Lx_/2), (-Ly_/2, Ly_/2),), "dimensionless")

    Psi0 = initialize_Psi(N, width=100, vortexnumber=0)
    Psi0_ = Psi0*x_s
    Psi0_ /= jnp.sqrt(jnp.sum(density(Psi0_))*domain_.dx[0]**2)

    eq = GPE2DTSControl(domain_, k, epsilon, lambda a,b,c: 0.0, trap_factor=1.0)

    solver = StrangSplitting(eq.A_term, eq.domain.dx[0], eq.fft, eq.ifft, -1j)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(jax.jit(lambda t, y, args: eq.B_terms(y, t))),
        solver,
        t0=t_start_,
        t1=t_final_,
        dt0=dt_,
        y0=jnp.stack([Psi0_.real, Psi0_.imag], axis=-1),
        # stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        saveat=diffrax.SaveAt(ts=jnp.linspace(t_start_, t_final_, 100)),
        max_steps=1000000,
    )

    def tf_mu_2d_from_class(N, g, trap_factor, e):
        # squared freqs:
        wx2 = trap_factor * (1.0 + e)
        wy2 = trap_factor * (1.0 - e)
        wx, wy = jnp.sqrt(wx2), jnp.sqrt(wy2)
        mu = jnp.sqrt((N * g * wx * wy) / (2.0 * jnp.pi))
        return mu, wx2, wy2

    def tf_density_2d_from_class(X, Y, N, g, trap_factor, e):
        mu, wx2, wy2 = tf_mu_2d_from_class(N, g, 0.5 * trap_factor, e)
        V = 0.5 * (wx2 * X**2 + wy2 * Y**2)
        n = jnp.clip((mu - V) / g, min=0.0)
        # tiny renormalization (discretization error) to hit N exactly:
        dx = (X[1,0] - X[0,0])
        dy = (Y[0,1] - Y[0,0])
        Nin = jnp.sum(n) * dx * dy
        n = n * (N / (Nin + 1e-12))
        return n

    X, Y = domain_.mesh()

    tf_density = tf_density_2d_from_class(X, Y, 1.0, k, 1.0, 0.0)   

    np.testing.assert_allclose(tf_density, density(solution.ys[-1][...,0] + 1j*solution.ys[-1][...,1]), rtol=1e-3, atol=1e-3)

def test_1d_cahn_hilliard_pde_model():
    Nx, Ny = 256, 1
    Lx = 0.01 * Nx
    Ly = 0.01 * Ny
    domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

    t_start = 0.0
    t_final = 10.0
    dt = 0.00005
    ts_save = jnp.linspace(t_start, t_final, 200)

    model = PDEModel(
        equation_type=CahnHilliard2DPeriodic,
        domain=domain,
        solver_type=SemiImplicitFourierSpectral
    )

    u0 = jnp.ones((Nx, Ny))
    u0 = u0.at[:Nx//2, :].set(-1.0)

    solution = model.solve(
        parameters={"kappa": 0.002, "mu": lambda c: c**3 - c, "D": lambda c: jnp.ones_like(c), "derivs": "fd"},
        y0=u0,
        ts=ts_save,
        solver_parameters={"A": 0.5},
        dt0=dt
    )

    analytic_solution = jnp.tanh(domain.axes()[0] / jnp.sqrt(2 * 0.002))

    np.testing.assert_allclose(solution[-1].squeeze()[Nx//4:3*Nx//4], analytic_solution[Nx//4:3*Nx//4], rtol=1e-3, atol=1e-3)

def test_1d_allen_cahn_pde_model():
    Nx, Ny = 256, 1
    Lx = 0.01 * Nx
    Ly = 0.01 * Ny
    domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

    t_start = 0.0
    t_final = 10.0
    dt = 0.00005
    ts_save = jnp.linspace(t_start, t_final, 200)

    model = PDEModel(
        equation_type=AllenCahn2DPeriodic,
        domain=domain,
        solver_type=diffrax.Tsit5
    )

    u0 = jnp.ones((Nx, Ny))
    u0 = u0.at[:Nx//2, :].set(-1.0)

    solution = model.solve(
        parameters={"kappa": 0.002, "mu": lambda c: c**3 - c, "R": lambda c: jnp.ones_like(c), "derivs": "fd"},
        y0=u0,
        ts=ts_save,
        solver_parameters={},
        dt0=dt,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6)
    )

    analytic_solution = jnp.tanh(domain.axes()[0] / jnp.sqrt(2 * 0.002))

    np.testing.assert_allclose(solution[-1].squeeze()[Nx//4:3*Nx//4], analytic_solution[Nx//4:3*Nx//4], rtol=1e-3, atol=1e-3)

def test_2d_gross_pitaevskii_pde_model():
    def density(psi):
        return jnp.abs(psi)**2

    atoms = 5e5
    hbar = 1.05e-34 #J*s
    omega = 2*jnp.pi*10 #1/s
    omega_z = jnp.sqrt(8)*omega
    epsilon = 0.0
    mass = 3.8175406e-26 #kg (atomic mass of sodium-23)
    a0 = 5.29177210903e-11
    a_s = 100*a0
    N = 128

    x_s = jnp.sqrt(hbar/(mass*omega))
    t_s = 1/omega

    # Length of the x and y axes in meters
    Lx = 150e-6 #meters
    Ly = 150e-6 #meters

    # Get dimensionless variables
    Lx_ = Lx/x_s
    Ly_ = Ly/x_s

    # Get k
    k = 4*jnp.pi*a_s*atoms*jnp.sqrt((mass*omega_z)/(2*jnp.pi*hbar))

    epsilon = 0.0

    t_start = 0.0
    t_final = 0.1
    dt = 1e-5

    t_start_ = t_start/t_s
    t_final_ = t_final/t_s
    dt_ = dt/t_s

    domain_ = Domain((N,N,), ((-Lx_/2, Lx_/2), (-Ly_/2, Ly_/2),), "dimensionless")

    Psi0 = initialize_Psi(N, width=100, vortexnumber=0)
    Psi0_ = Psi0*x_s
    Psi0_ /= jnp.sqrt(jnp.sum(density(Psi0_))*domain_.dx[0]**2)

    pde_model = PDEModel(
        equation_type=GPE2DTSControl,
        domain=domain_,
        solver_type=StrangSplitting
    )

    solution = pde_model.solve(
        parameters={"k": k, "e": epsilon, "lights": lambda t, x, y: 0.0, "trap_factor": 1.0},
        y0=jnp.stack([Psi0_.real, Psi0_.imag], axis=-1),
        ts=jnp.linspace(t_start_, t_final_, 100),
        solver_parameters={"time_scale": -1j},
        dt0=dt_
    )

    def tf_mu_2d_from_class(N, g, trap_factor, e):
        # squared freqs:
        wx2 = trap_factor * (1.0 + e)
        wy2 = trap_factor * (1.0 - e)
        wx, wy = jnp.sqrt(wx2), jnp.sqrt(wy2)
        mu = jnp.sqrt((N * g * wx * wy) / (2.0 * jnp.pi))
        return mu, wx2, wy2

    def tf_density_2d_from_class(X, Y, N, g, trap_factor, e):
        mu, wx2, wy2 = tf_mu_2d_from_class(N, g, 0.5 * trap_factor, e)
        V = 0.5 * (wx2 * X**2 + wy2 * Y**2)
        n = jnp.clip((mu - V) / g, min=0.0)
        # tiny renormalization (discretization error) to hit N exactly:
        dx = (X[1,0] - X[0,0])
        dy = (Y[0,1] - Y[0,0])
        Nin = jnp.sum(n) * dx * dy
        n = n * (N / (Nin + 1e-12))
        return n

    X, Y = domain_.mesh()

    tf_density = tf_density_2d_from_class(X, Y, 1.0, k, 1.0, 0.0)   

    np.testing.assert_allclose(tf_density, density(solution[-1][...,0] + 1j*solution[-1][...,1]), rtol=1e-3, atol=1e-3)