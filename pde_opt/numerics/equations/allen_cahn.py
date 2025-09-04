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
    """Allen-Cahn equation in 2D with periodic boundary conditions.

    The Allen-Cahn equation describes phase transitions and interface dynamics.
    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = -R(u) \\mu

    where u is the concentration, R(u) is the reaction term, μ is the chemical potential, and κ is a parameter (the gradient energy coefficient).
    The chemical potential is given by:

    .. math::
        \\mu = \\mu_h(u) - \\kappa \\nabla^2 u

    Parameters
    ----------
    domain : Domain
        The computational domain for the equation
    kappa : float
        Gradient energy coefficient (positive parameter)
    mu : Callable or eqx.Module
        Function for the chemical potential μ_h(u)
    R : Callable or eqx.Module  
        Function for the reaction term R(u)
    derivs : str, default "fourier"
        Type of derivative computation: "fourier" or "fd"
    """

    domain: Domain  # The computational domain for the equation
    """Test docstring"""
    
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
        """Compute RHS using Fourier spectral method.
        
        Parameters
        ----------
        state : jax.Array
            Current state u(x,y)
        t : float
            Current time
            
        Returns
        -------
        jax.Array
            Right-hand side of the Allen-Cahn equation
        """
        state_hat = self.fft(state)
        mu = self.ifft(self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t):
        """Compute RHS using finite difference method.
        
        Parameters
        ----------
        state : jax.Array
            Current state u(x,y)
        t : float
            Current time
            
        Returns
        -------
        jax.Array
            Right-hand side of the Allen-Cahn equation
        """
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        return -self.R(state) * mu

@dataclasses.dataclass
class AllenCahn2DSmoothedBoundary(BaseEquation):
    """Allen-Cahn equation with smoothed boundary method for arbitrary geometries.

    This class implements the Allen-Cahn equation using the smoothed boundary
    method, which allows for complex domain geometries through a smooth
    level-set function ψ.

    The equation is:

    .. math::
        \\frac{\\partial u}{\\partial t} = -R(u) \\mu

    where the chemical potential includes boundary effects:

    .. math::
        \\mu = \\mu_h(u) - \\frac{\\kappa}{\\psi} \\nabla \\cdot (\\psi \\nabla u) 
        - \\sqrt{\\kappa} \\frac{|\\nabla \\psi|}{\\psi} \\sqrt{2f} \\cos(\\theta)

    Parameters
    ----------
    domain : Domain
        The computational domain with geometry information
    kappa : float
        Gradient energy coefficient
    f : Callable or eqx.Module
        Free energy density function
    mu : Callable or eqx.Module
        Chemical potential function μ_h(u)
    R : Callable or eqx.Module
        Reaction term function R(u)
    theta : Callable or eqx.Module
        Contact angle function θ(t)
    derivs : str, default "fd"
        Derivative method (only "fd" supported)
        
    Notes
    -----
    - Uses smoothed boundary method for complex geometries
    - ψ is the smooth level-set function (1 inside domain, 0 outside)
    - Includes contact angle effects at boundaries
    """

    domain: Domain
    kappa: float
    f: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    theta: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
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