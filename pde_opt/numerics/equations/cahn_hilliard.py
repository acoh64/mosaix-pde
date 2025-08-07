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
class CahnHilliard2DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fourier"
    # TODO: add smoothing for fft (aliasing)

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2) ** 2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        self.fourier_symbol = self.kappa * self.two_pi_i_k_4
        
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
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        return self.ifft(self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy).real

    def rhs_fd(self, state, t):
        mu = self.mu(state) - self.kappa * self.lap(state)
        gradx = self.gradx(self.D(state) * self.gradx(mu))
        grady = self.grady(self.D(state) * self.grady(mu))
        return gradx + grady

# @dataclasses.dataclass
# class CahnHilliard2DSmoothedBoundary(BaseEquation):
#     domain: Domain
#     gamma: float
#     chem_pot: Callable
#     D: Callable
#     space: str = "R"

#     def __post_init__(self):
#         self.psi = self.domain.geometry.smooth
#         self.gradxf = (
#             lambda arr: (jnp.roll(arr, -1, axis=1) - arr) / (self.domain.dx[0])
#         )
#         self.gradyf = (
#             lambda arr: (jnp.roll(arr, -1, axis=0) - arr) / (self.domain.dx[1])
#         )
#         self.gradxb = lambda arr: (arr - jnp.roll(arr, 1, axis=1)) / (self.domain.dx[0])
#         self.gradyb = lambda arr: (arr - jnp.roll(arr, 1, axis=0)) / (self.domain.dx[0])

#     def rhs(self, state, t):
#         tmp1 = self.psi * self.gradxf(state)
#         tmp2 = self.psi * self.gradyf(state)
#         tmp3 = (self.gamma / self.psi) * (self.gradxb(tmp1) + self.gradyb(tmp2))
#         tmp4 = self.chem_pot(state) - tmp3
#         tmp5 = self.gradxf(tmp4)
#         tmp6 = self.gradyf(tmp4)
#         tmp7 = self.psi * self.D(state) * state
#         return (self.gradxb(tmp7 * tmp5) + self.gradyb(tmp7 * tmp6)) / self.psi

@dataclasses.dataclass
class CahnHilliard3DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fourier"
    # TODO: add smoothing for fft (aliasing)

    def __post_init__(self):
        self.kx, self.ky, self.kz = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kz = 2j * jnp.pi * self.kz
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_kz_2 = (self.two_pi_i_kz) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2 + self.two_pi_i_kz_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2) ** 2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        self.fourier_symbol = self.kappa * self.two_pi_i_k_4
        
        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.gradx = jax.jit(gradient(self.domain, 0))
            self.grady = jax.jit(gradient(self.domain, 1))
            self.gradz = jax.jit(gradient(self.domain, 2))
            self.lap = jax.jit(laplacian(self.domain))
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        state_hat = self.fft(state)
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        tmpz = self.fft(self.D(state) * self.ifft(self.two_pi_i_kz * tmp))
        return self.ifft(
            self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy + self.two_pi_i_kz * tmpz
        ).real

    def rhs_fd(self, state, t):
        mu = self.mu(state) - self.kappa * self.lap(state)
        gradx = self.gradx(self.D(state) * self.gradx(mu))
        grady = self.grady(self.D(state) * self.grady(mu))
        gradz = self.gradz(self.D(state) * self.gradz(mu))
        return gradx + grady + gradz
