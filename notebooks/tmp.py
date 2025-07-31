import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pde_opt
from pde_opt.model import OptimizationModel
from pde_opt.numerics.equations import CahnHilliardSIFFT
from pde_opt.numerics.solvers import SemiImplicitFourierSpectral
from pde_opt.numerics.domains import Domain
from pde_opt.numerics.functions import (
    DiffusionLegendrePolynomials,
    ChemicalPotentialLegendrePolynomials,
)

import equinox as eqx
import diffrax



Nx = Ny = 128
Lx = Ly = 0.01 * 128

domain = Domain(
    (Nx, Ny),
    (
        (-Lx / 2, Lx / 2),
        (-Ly / 2, Ly / 2),
    ),
    "dimensionless",
)

opt_model = OptimizationModel(
    equation_type=CahnHilliardSIFFT,
    domain=domain,
    solver_type=SemiImplicitFourierSpectral,
)

data = {}
data['ys'] = sol
data['ts'] = ts

inds = [[30,40,50], [50,60,70], [70,80,90]]

init_params = {
    "kappa": 0.002,
    "mu": ChemicalPotentialLegendrePolynomials(
        jnp.array([0.0, -2.0]), lambda x: jnp.log(x / (1.0 - x))
    ),
    "D": DiffusionLegendrePolynomials(jnp.array([0.0])),
}

weights = {
    "kappa": 0.0,
    "mu": ChemicalPotentialLegendrePolynomials(jnp.array([0.0, 0.0])),
    "D": None,
}


res = opt_model.train(data, inds, init_params, solver_params, weights, 1.0)




params = {
    "reset_func": reset_func,
    "diffusion_coefficient": 0.1,
    "max_control_step": 0.1,
    "end_time": 3.0,
    "step_dt": 0.05,
    "numeric_dt": 0.0001,
    "domain": domain,
    "field_dim": 1,
    "reward_function": lambda x: np.var(x),
    "discrete_action_space": True
}

env = gym.make('AdvectionDiffusion-v0', **params)

observation, info = env.reset()

observation, reward, terminated, truncated, info = env.step(action)


