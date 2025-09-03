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
    _gradz_c,
)


@dataclasses.dataclass
class CahnHilliard2DPeriodic(BaseEquation):
    """Cahn–Hilliard equation in 2D with periodic boundary conditions.

    The equation is of the form

        d/dt u = ∇·( D(u) ∇μ ),   μ = μ_h(u) - κ Δu,

    where u is the concentration, D(u) is the mobility, μ is the chemical potential, and κ is a parameter (the gradient energy coefficient).

    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        mu: Function for the chemical potential
        D: Function for the mobility
        derivs: Type of derivative to use, "fourier" or "fd"
    """

    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"
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
        """RHS of the Cahn–Hilliard equation, with derivatives in Fourier space"""

        state_hat = self.fft(state)
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        return self.ifft(self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy).real

    def rhs_fd(self, state, t):
        """2nd-order finite difference RHS for Cahn–Hilliard with periodic BCs"""

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

    The equation is of the form

        d/dt u = ∇·( D(u) ∇μ ),   μ = μ_h(u) - κ Δu,

    where u is the concentration, D(u) is the mobility, μ is the chemical potential, and κ is a parameter (the gradient energy coefficient).

    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        mu: Function for the chemical potential
        D: Function for the mobility
        derivs: Type of derivative to use, "fourier" or "fd"
    """

    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"
    fft = None
    ifft = None
    fourier_symbol = None
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
        """RHS of the Cahn–Hilliard equation, with derivatives in Fourier space"""

        state_hat = self.fft(state)
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        tmpz = self.fft(self.D(state) * self.ifft(self.two_pi_i_kz * tmp))
        return self.ifft(
            self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy + self.two_pi_i_kz * tmpz
        ).real

    def rhs_fd(self, state, t):
        """2nd-order finite difference RHS for Cahn–Hilliard with periodic BCs"""

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
    """Cahn–Hilliard equation in 2D solved with the smoothed boundary method for arbitrary boundaries.

    The equation is of the form

        d/dt u = 1/ψ ∇·( ψ D(u) ∇μ ) + |∇ψ|/ψ J_n,   μ = μ_h(u) - κ/ψ ∇·(ψ ∇u) - sqrt(κ) |∇ψ|/ψ sqrt(2f) cos(θ),

    where u is the concentration, D(u) is the mobility, μ is the chemical potential, κ is a parameter (the gradient energy coefficient), ψ is a smooth function that is 1 inside the domain and 0 outside, and J_n is the normal flux, f is the free energy density, and θ is the contact angle between the phase boundary and the interface.

    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        f: Function for the free energy density
        mu: Function for the chemical potential
        D: Function for the mobility
        theta: Function for the contact angle
        flux: Function for the normal flux
        derivs: Type of derivative to use, "fd"
    """

    domain: Domain
    kappa: float
    f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
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
        self.left_half = self.left_half.at[:50, :].set(1.0)
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        """2nd-order finite difference RHS for Cahn–Hilliard with smoothed boundary"""

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
