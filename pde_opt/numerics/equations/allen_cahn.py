from abc import ABC, abstractmethod
import dataclasses
from typing import Callable, TypeVar, Union
import jax
import jax.numpy as jnp
import equinox as eqx

import pde_opt.numerics.utils.fft_utils as fftutils
from ..domains import Domain
from .base_eq import BaseEquation

def _lap_2nd_2D(u, hx, hy):
    return (
        (jnp.roll(u, -1, 0) - 2*u + jnp.roll(u,  1, 0)) / hx**2 +
        (jnp.roll(u, -1, 1) - 2*u + jnp.roll(u,  1, 1)) / hy**2
    )

def _lap_2nd_2D_zero_gradient(u, hx, hy):
    """
    2D Laplacian with zero normal gradient boundary conditions.
    Uses ghost cells with edge padding to implement Neumann BCs.
    """
    # Pad the field with edge values (equivalent to zero normal gradient)
    padded = jnp.pad(u, ((1, 1), (1, 1)), mode='edge')
    
    # Apply 2nd order central difference Laplacian
    laplacian = (
        (padded[2:, 1:-1] - 2*padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / hx**2 +
        (padded[1:-1, 2:] - 2*padded[1:-1, 1:-1] + padded[1:-1, :-2]) / hy**2
    )
    
    return laplacian

@dataclasses.dataclass
class AllenCahn2DPeriodic(BaseEquation):
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
        state_hat = self.fft(state)
        mu = self.ifft(self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat).real
        return -self.R(state) * mu

    def rhs_fd(self, state, t):
        hx, hy = self.domain.dx
        mu = self.mu(state) - self.kappa * _lap_2nd_2D(state, hx, hy)
        return -self.R(state) * mu


@dataclasses.dataclass
class AllenCahn2DZeroGradient(BaseEquation):
    """
    Allen-Cahn equation with zero normal gradient boundary conditions.
    
    The equation is: ∂u/∂t = R(u) * (μ(u) - κ∇²u)
    where μ(u) is the chemical potential and R(u) is the mobility.
    
    Boundary conditions: ∂u/∂n = 0 (zero normal gradient)
    Implemented using finite differences with ghost cells.
    """
    domain: Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    R: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    derivs: str = "fd"

    def rhs(self, state, t):
        raise NotImplementedError("rhs method not implemented")

    def __post_init__(self):
        # For zero gradient BCs, we only support finite differences
        # since FFT requires periodic boundary conditions
        if self.derivs == "fd":
            self.rhs = jax.jit(self.rhs_fd)
        else:
            raise ValueError(f"Invalid derivative type: {self.derivs}")

    def rhs_fd(self, state, t):
        """
        Right-hand side of the Allen-Cahn equation with zero normal gradient BCs.
        
        Args:
            state: Current state field u(x, y)
            t: Current time
            
        Returns:
            ∂u/∂t at the current state and time
        """
        hx, hy = self.domain.dx
        
        # Compute chemical potential: μ = μ(u) - κ∇²u
        # Use the zero gradient Laplacian implementation
        mu = self.mu(state) - self.kappa * _lap_2nd_2D_zero_gradient(state, hx, hy)
        
        # Return the RHS: ∂u/∂t = R(u) * μ
        return -self.R(state) * mu
