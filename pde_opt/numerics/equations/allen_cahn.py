import dataclasses
from typing import Callable, Union
import jax
import jax.numpy as jnp
import equinox as eqx

from ..domains import Domain
from .base_eq import BaseEquation
from ..utils.derivatives import _lap_2nd_2D
from ..utils.derivatives import _gradx_c, _grady_c, _avgx_c2f, _avgy_c2f, _divx_f2c, _divy_f2c, _gradx_c2f, _grady_c2f

@dataclasses.dataclass
class AllenCahn2DPeriodic(BaseEquation):
    """Allen–Cahn equation in 2D with periodic boundary conditions.

    The equation is of the form

        d/dt u = -R(u)μ,   μ = μ_h(u) - κ Δu,

    where u is the concentration, R(u) is the reaction term, μ is the chemical potential, and κ is a parameter (the gradient energy coefficient).

    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        mu: Function for the chemical potential
        R: Function for the reaction term
        derivs: Type of derivative to use, "fourier" or "fd"
    """

    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
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
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        
        if self.derivs == "fourier":
            self.rhs = jax.jit(self.rhs_fourier)
        elif self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fourier(self, state, t):
        """RHS of the Allen–Cahn equation, with derivatives in Fourier space"""

        state_hat = self.fft(state)
        mu = self.ifft(self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t):
        """2nd-order finite difference RHS for Allen–Cahn with periodic BCs"""

        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        return -self.R(state) * mu

@dataclasses.dataclass
class AllenCahn2DSmoothedBoundary(BaseEquation):
    """Allen–Cahn equation in 2D solved with the smoothed boundary method for arbitrary boundaries.

    The equation is of the form

        d/dt u = -R(u)μ,   μ = μ_h(u) - κ/ψ ∇·(ψ ∇u) - sqrt(κ) |∇ψ|/ψ sqrt(2f) cos(θ),

    where u is the concentration, R(u) is the reaction term, μ is the chemical potential, κ is a parameter (the gradient energy coefficient), ψ is a smooth function that is 1 inside the domain and 0 outside, and f is the free energy density, and θ is the contact angle between the phase boundary and the interface.

    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        f: Function for the free energy density
        mu: Function for the chemical potential
        R: Function for the reaction term
        theta: Function for the contact angle
        flux: Function for the normal flux
        derivs: Type of derivative to use, "fd"
    """

    domain: Domain
    kappa: float
    f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    theta: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    flux: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"

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
        self.left_half = self.left_half.at[:, :100].set(1.0)
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        """2nd-order finite difference RHS for Allen–Cahn with smoothed boundary"""

        f = self.f(state)
        mu = self.mu(state)
        mask_avgx = _avgx_c2f(self.psi)
        mask_avgy = _avgy_c2f(self.psi)
        mu += (
            - (self.kappa / self.psi)
            * (
                _divx_f2c(mask_avgx * _gradx_c2f(state, self.hx), self.hx)
                + _divy_f2c(mask_avgy * _grady_c2f(state, self.hy), self.hy)
            )
            - self.sqrt_kappa
            * self.norm_grad_psi
            * jnp.sqrt(2.0 * f)
            * jnp.cos(self.theta(t)) * self.left_half
        )
        return -self.R(state) * mu