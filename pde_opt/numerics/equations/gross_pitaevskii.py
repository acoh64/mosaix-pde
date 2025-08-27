import dataclasses
from typing import Callable

import jax.numpy as jnp

from ..domains import Domain
from .base_eq import TimeSplittingEquation


@dataclasses.dataclass
class GPE2DTSControl(TimeSplittingEquation):
    domain: Domain
    k: float
    e: float
    lights: Callable
    trap_factor: float = 1.0
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
        self.A_term = 0.5j * self.two_pi_i_k_2

    def A_terms(self, state, t):
        return self.A_term

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
    # TODO: need to fix this
    domain: Domain
    k: float
    e: float
    omega: float
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = jnp.fft.fftn
        self.ifft = jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()
        
    def A_terms(self, state_hat, t):
        return 0.5j*self.two_pi_i_kx_2 - self.omega*self.ymesh*self.two_pi_i_kx, 0.5j*self.two_pi_i_ky_2 + self.omega*self.xmesh*self.two_pi_i_ky

    def B_terms(self, state, t):
        return -0.5j*((1+self.e)*self.xmesh**2 + (1-self.e)*self.ymesh**2) - self.k*1j*(jnp.abs(state)**2)
