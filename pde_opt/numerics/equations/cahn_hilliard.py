from abc import ABC, abstractmethod
import dataclasses
from typing import Callable, TypeVar, Union
import jax
import jax.numpy as jnp
import equinox as eqx
import sympy as sp

import pde_opt.numerics.utils.fft_utils as fftutils
from ..domains import Domain
from .base_eq import BaseEquation

def _lap_2nd_2D(u, hx, hy):
    return (
        (jnp.roll(u, -1, 0) - 2*u + jnp.roll(u,  1, 0)) / hx**2 +
        (jnp.roll(u, -1, 1) - 2*u + jnp.roll(u,  1, 1)) / hy**2
    )

def _lap_2nd_3D(u, hx, hy, hz):
    return (
        (jnp.roll(u, -1, 0) - 2*u + jnp.roll(u,  1, 0)) / hx**2 +
        (jnp.roll(u, -1, 1) - 2*u + jnp.roll(u,  1, 1)) / hy**2 +
        (jnp.roll(u, -1, 2) - 2*u + jnp.roll(u,  1, 2)) / hz**2
    )

def _gradx_c2f(a, hy):  # center -> y-face (i,j+1/2)
    return (jnp.roll(a, -1, 0) - a) / hy

def _grady_c2f(a, hx):  # center -> x-face (i+1/2,j), 2nd-order at the face
    return (jnp.roll(a, -1, 1) - a) / hx

def _gradz_c2f(a, hx, hy):  # center -> z-face (i,j+1/2,k+1/2)
    return (jnp.roll(a, -1, 2) - a) / (hx * hy)

def _avgx_c2f(a):       # average to y-face
    return 0.5 * (a + jnp.roll(a, -1, 0))

def _avgy_c2f(a):       # average to x-face
    return 0.5 * (a + jnp.roll(a, -1, 1))

def _avgz_c2f(a, hx, hy):  # average to z-face (i,j+1/2,k+1/2)
    return 0.5 * (a + jnp.roll(a, -1, 2))

def _divx_f2c(Fy, hy):  # y-face -> center divergence
    return (Fy - jnp.roll(Fy, 1, 0)) / hy

def _divy_f2c(Fx, hx):  # x-face -> center divergence
    return (Fx - jnp.roll(Fx, 1, 1)) / hx

def _divz_f2c(Fy, hx, hy):  # z-face -> center divergence
    return (Fy - jnp.roll(Fy, 1, 2)) / (hx * hy)

@dataclasses.dataclass
class CahnHilliard2DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fourier"
    # TODO: add smoothing for fft (aliasing)

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

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
        """
        2nd-order conservative RHS for Cahn–Hilliard:
            ∂_t u = ∇·( D(u) ∇μ ),   μ = μ_nl(u) - κ Δu,
        with periodic BCs. Works for variable mobility D(u).
        """
        hx, hy = self.domain.dx

        # chemical potential: μ = μ_nl(u) - κ Δu
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)

        # gradients of μ at faces (2nd-order at faces)
        mux_f = _gradx_c2f(mu, hx)
        muy_f = _grady_c2f(mu, hy)

        # mobility at faces (2nd-order average)
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)

        # fluxes at faces
        Fx = Dx_f * mux_f
        Fy = Dy_f * muy_f

        # divergence back to centers (2nd-order)
        return _divx_f2c(Fx, hx) + _divy_f2c(Fy, hy)

@dataclasses.dataclass
class CahnHilliard3DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fourier"
    # TODO: add smoothing for fft (aliasing)

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

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
        """
        2nd-order conservative RHS for Cahn–Hilliard:
            ∂_t u = ∇·( D(u) ∇μ ),   μ = μ_nl(u) - κ Δu,
        with periodic BCs. Works for variable mobility D(u).
        """
        hx, hy, hz = self.domain.dx
        
        # chemical potential: μ = μ_nl(u) - κ Δu
        mu = self.mu(state) - self.kappa * _lap_2nd_3D(state, hx, hy, hz)

        # gradients of μ at faces (2nd-order at faces)
        mux_f = _gradx_c2f(mu, hx)
        muy_f = _grady_c2f(mu, hy)
        muz_f = _gradz_c2f(mu, hx, hy)

        # mobility at faces (2nd-order average)
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)
        Dz_f = _avgz_c2f(Du, hx, hy)

        # fluxes at faces
        Fx = Dx_f * mux_f
        Fy = Dy_f * muy_f
        Fz = Dz_f * muz_f

        # divergence back to centers (2nd-order)
        return _divx_f2c(Fx, hx) + _divy_f2c(Fy, hy) + _divz_f2c(Fz, hx, hy)






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
