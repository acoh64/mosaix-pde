"""
This module contains various Gross-Pitaevskii equation classes.
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp

from ..domains import Domain
from .base_eq import TimeSplittingEquation

# Useful physics constants for the GPE
hbar = 1.05e-34  # J*s
mass_Na23 = 3.8175406e-26  # kg (atomic mass of sodium-23)
a0 = 5.29177210903e-11  # Bohr radius


@dataclasses.dataclass
class GPE2DTSControl(TimeSplittingEquation):
    """Gross-Pitaevskii equation in 2D with time-splitting and control.

    The Gross-Pitaevskii equation describes the dynamics of Bose-Einstein condensates.
    The equation is:

    .. math::
        i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r}, t) + g|\\psi|^2\\right]\\psi

    where ψ is the wave function, V is the external potential, and g is the interaction strength.
    The external potential includes a harmonic trap and control field:

    .. math::
        V(\\mathbf{r}, t) = \\frac{1}{2}m\\omega^2\\left[(1+\\epsilon)x^2 + (1-\\epsilon)y^2\\right] + V_{control}(\\mathbf{r}, t)
    """

    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    k: float
    """Interaction strength parameter"""
    e: float
    """Trap ellipticity parameter"""
    lights: Callable
    """Function for the control field"""
    trap_factor: float = 1.0
    """Scaling factor for the harmonic trap"""
    fft = None
    ifft = None
    A_term = None
    dx = None

    def __post_init__(self):
        self.dx = self.domain.dx[0]
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()
        self.control = lambda t: self.lights(t, self.xmesh, self.ymesh)
        self.A_term = 0.5j * self.two_pi_i_k_2 * 0.0

    def A_terms(self, state, t):
        return self.A_term * 0.0

    def B_terms(self, state, t):
        tmp = (
            -0.5j
            * self.trap_factor
            * ((1 + self.e) * self.xmesh**2 + (1 - self.e) * self.ymesh**2)
            - 1j * self.control(t)
            - self.k * 1j * (jnp.abs(state[..., 0] + 1j * state[..., 1]) ** 2)
        )
        return jnp.stack([tmp.real, tmp.imag], axis=-1)

    def rhs(self, state, t):
        # # TODO: implement this using fourier space (just A + B) and using finite difference
        # raise NotImplementedError("rhs method not implemented")
        # this needs to be fixed
        return self.B_terms(state, t)


@dataclasses.dataclass
class GPE2DTSRot(TimeSplittingEquation):
    """Gross-Pitaevskii equation in 2D with time-splitting and rotation.

    The Gross-Pitaevskii equation describes the dynamics of Bose-Einstein condensates.
    The equation is:

    .. math::
        i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r}) + g|\\psi|^2 - \\Omega L_z\\right]\\psi

    where ψ is the wave function, V is the external potential, g is the interaction strength,
    and Ω is the rotation frequency with L_z being the angular momentum operator.
    The external potential includes a harmonic trap:

    .. math::
        V(\\mathbf{r}) = \\frac{1}{2}m\\omega^2\\left[(1+\\epsilon)x^2 + (1-\\epsilon)y^2\\right]
    """

    domain: Domain  # The computational domain for the equation
    """Domain of the equation"""
    k: float
    """Interaction strength parameter"""
    e: float
    """Trap ellipticity parameter"""
    omega: float
    """Rotation frequency"""

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()

    def A_terms(self, state_hat, t):
        return (
            0.5j * self.two_pi_i_kx_2 - self.omega * self.ymesh * self.two_pi_i_kx,
            0.5j * self.two_pi_i_ky_2 + self.omega * self.xmesh * self.two_pi_i_ky,
        )

    def B_terms(self, state, t):
        return -0.5j * (
            (1 + self.e) * self.xmesh**2 + (1 - self.e) * self.ymesh**2
        ) - self.k * 1j * (jnp.abs(state) ** 2)
