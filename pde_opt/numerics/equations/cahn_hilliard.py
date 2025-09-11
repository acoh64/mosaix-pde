"""
This module contains various Cahn-Hilliard equation classes.
"""

import dataclasses
from typing import Callable, Union
import jax
import jax.numpy as jnp
import equinox as eqx

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
    _gradx_c,
    _grady_c,
)


@dataclasses.dataclass
class CahnHilliard2DPeriodic(BaseEquation):
    """Cahn–Hilliard equation in 2D with periodic boundary conditions.

    The Cahn-Hilliard equation describes phase separation and coarsening dynamics.
    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = \\nabla \\cdot (D(u) \\nabla \\mu)

    where u is the concentration, D(u) is the mobility, and μ is the chemical potential.
    The chemical potential is given by:

    .. math::
        \\mu = \\mu_h(u) - \\kappa \\nabla^2 u
    """

    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the mobility"""
    derivs: str = "fd"
    """Type of derivative computation"""
    fft = None
    ifft = None
    fourier_symbol = None

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
        hx, hy = self.domain.dx

        # chemical potential: μ = μ_nl(u) - κ Δu
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)

        # gradients of μ at faces
        mux_f = _gradx_c2f(mu, hx)
        muy_f = _grady_c2f(mu, hy)

        # mobility at faces
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)

        # fluxes at faces
        Fx = Dx_f * mux_f
        Fy = Dy_f * muy_f

        # divergence back to centers
        return _divx_f2c(Fx, hx) + _divy_f2c(Fy, hy)


@dataclasses.dataclass
class CahnHilliard3DPeriodic(BaseEquation):
    """Cahn–Hilliard equation in 3D with periodic boundary conditions.

    The Cahn-Hilliard equation describes phase separation and coarsening dynamics.
    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = \\nabla \\cdot (D(u) \\nabla \\mu)

    where u is the concentration, D(u) is the mobility, and μ is the chemical potential.
    The chemical potential is given by:

    .. math::
        \\mu = \\mu_h(u) - \\kappa \\nabla^2 u
    """

    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the mobility"""
    derivs: str = "fd"
    """Type of derivative computation"""
    fft = None
    ifft = None
    fourier_symbol = None

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
        hx, hy, hz = self.domain.dx

        # chemical potential: μ = μ_nl(u) - κ Δu
        mu = self.mu(state) - self.kappa * _lap_2nd_3D(state, hx, hy, hz)

        # gradients of μ at faces
        mux_f = _gradx_c2f(mu, hx)
        muy_f = _grady_c2f(mu, hy)
        muz_f = _gradz_c2f(mu, hz)

        # mobility at faces
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)
        Dz_f = _avgz_c2f(Du)

        # fluxes at faces
        Fx = Dx_f * mux_f
        Fy = Dy_f * muy_f
        Fz = Dz_f * muz_f

        # divergence back to centers
        return _divx_f2c(Fx, hx) + _divy_f2c(Fy, hy) + _divz_f2c(Fz, hz)


@dataclasses.dataclass
class CahnHilliard2DSmoothedBoundary(BaseEquation):
    """Cahn–Hilliard equation with smoothed boundary method for arbitrary geometries.

    This class implements the Cahn-Hilliard equation using the smoothed boundary
    method, which allows for complex domain geometries through a smooth
    level-set function ψ.

    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = \\frac{1}{\\psi} \\nabla \\cdot (\\psi D(u) \\nabla \\mu) + \\frac{|\\nabla \\psi|}{\\psi} J_n

    where the chemical potential includes boundary effects:

    .. math::
        \\mu = \\mu_h(u) - \\frac{\\kappa}{\\psi} \\nabla \\cdot (\\psi \\nabla u)
        - \\sqrt{\\kappa} \\frac{|\\nabla \\psi|}{\\psi} \\sqrt{2f} \\cos(\\theta)
    """

    domain: Domain
    """Domain of the equation"""
    kappa: float
    """Gradient energy coefficient"""
    f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the free energy density"""
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the chemical potential"""
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the mobility"""
    theta: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the contact angle"""
    flux: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    """Function for the normal flux"""
    derivs: str = "fd"
    """Type of derivative computation"""

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        self.psi = self.domain.geometry.smooth
        self.sqrt_kappa = jnp.sqrt(self.kappa)
        self.hx, self.hy = self.domain.dx
        self.norm_grad_psi = (
            jnp.sqrt(
                _gradx_c(self.psi, self.hx) ** 2 + _grady_c(self.psi, self.hy) ** 2
            )
            / self.psi
        )
        self.left_half = jnp.zeros_like(self.psi)
        self.left_half = self.left_half.at[:50, :].set(1.0)
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        f = self.f(state)
        mu = self.mu(state)
        mask_avgx = _avgx_c2f(self.psi)
        mask_avgy = _avgy_c2f(self.psi)
        inner_term = (
            mu
            - (self.kappa / self.psi)
            * (
                _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
                + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
            )
            - self.sqrt_kappa
            * self.norm_grad_psi
            * jnp.sqrt(2.0 * f)
            * (
                jnp.cos(self.theta(t)) * self.left_half
                + jnp.cos(jnp.pi - self.theta(t)) * (1.0 - self.left_half)
            )
        )
        gradx_inner = _gradx_c2f(inner_term, self.hx)
        grady_inner = _grady_c2f(inner_term, self.hy)
        Du = self.D(state)
        Dx_f = _avgx_c2f(Du)
        Dy_f = _avgy_c2f(Du)
        Fx = mask_avgx * Dx_f * gradx_inner
        Fy = mask_avgy * Dy_f * grady_inner
        return (
            _divx_f2c(Fx, self.hx) + _divy_f2c(Fy, self.hy)
        ) / self.psi + self.norm_grad_psi * self.flux(t)
