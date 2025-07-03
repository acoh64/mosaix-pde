import dataclasses
from typing import Callable, TypeVar

import jax.numpy as jnp

import pde_opt.numerics.utils.fft_utils as fftutils
from pde_opt.numerics import domains

State = TypeVar("State")

class ODE:
    def rhs(self, state, t):
        raise NotImplementedError

class TimeSplittingODE(ODE):
    def B_terms(self, state: State, t:float) -> State:
        raise NotImplementedError

@dataclasses.dataclass
class CahnHilliardSIFFT(ODE):

    domain: domains.Domain
    kappa: float
    mu: Callable
    D: Callable
    smooth: bool = False

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.two_pi_i_k_4 = (self.two_pi_i_k_2) ** 2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        self.fourier_symbol = self.kappa * self.two_pi_i_k_4

    def rhs(self, state, t):
        state_hat = self.fft(state)
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        return self.ifft(self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy).real
    

@dataclasses.dataclass
class GPE2DTSControl(TimeSplittingODE):
    
    domain: domains.Domain
    k: float
    e: float
    lights: Callable
    smooth: bool = True
    
    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx)**2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky)**2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()
        self.control = lambda t: self.lights(t, self.xmesh, self.ymesh)
        self.A_term = 0.5j*self.two_pi_i_k_2
    
    def B_terms(self, state, t):
        tmp = -0.5j*((1+self.e)*self.xmesh**2 + (1-self.e)*self.ymesh**2) - 1j*self.control(t) - self.k*1j*(jnp.abs(state[...,0] + 1j*state[...,1])**2)
        return jnp.stack([tmp.real, tmp.imag], axis=-1)