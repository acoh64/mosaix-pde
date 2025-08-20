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
from ..utils.derivatives import (
    _lap_2nd_2D,
    _lap_2nd_3D,
    _gradx_c2f,
    _grady_c2f,
    _gradz_c2f,
    _avgx_c2f,
    _avgy_c2f,
    _avgz_c2f,
    _divx_f2c,
    _divy_f2c,
    _divz_f2c,
)


@dataclasses.dataclass
class CahnHilliard2DPeriodic(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"

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
    derivs: str = "fd"
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


@dataclasses.dataclass
class CahnHilliard2DSmoothedBoundary(BaseEquation):
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        mask_avgx = _avgx_c2f(self.psi)
        mask_avgy = _avgy_c2f(self.psi)
        inner_term = mu - (self.kappa / self.psi) * (_divx_f2c(mask_avgx * _gradx_c2f(state, hx), hx) + _divy_f2c(mask_avgy * _grady_c2f(state, hy), hy))
        gradx_inner = _gradx_c2f(inner_term, hx)
        grady_inner = _grady_c2f(inner_term, hy)
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)
        Fx = mask_avgx * Dx_f * gradx_inner
        Fy = mask_avgy * Dy_f * grady_inner
        return (_divx_f2c(Fx, hx) + _divy_f2c(Fy, hy)) / self.psi