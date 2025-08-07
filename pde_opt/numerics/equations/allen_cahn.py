from abc import ABC, abstractmethod
import dataclasses
from typing import Callable, TypeVar, Union
import jax
import jax.numpy as jnp
import equinox as eqx

import pde_opt.numerics.utils.fft_utils as fftutils
from ..domains import Domain
from .base_eq import BaseEquation
from ..utils.derivatives import gradient, laplacian


@dataclasses.dataclass
class AllenCahn2DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fourier"
    # TODO: add smoothing for fft (aliasing)

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        
        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.gradx = jax.jit(gradient(self.domain, 0))
            self.grady = jax.jit(gradient(self.domain, 1))
            self.lap = jax.jit(laplacian(self.domain))
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        state_hat = self.fft(state)
        mu = self.ifft(self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat)
        return self.R(state) * mu

    def rhs_fd(self, state, t):
        mu = self.mu(state) - self.kappa * self.lap(state)
        return self.R(state) * mu
