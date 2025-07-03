import dataclasses
from typing import Callable

import jax.numpy as jnp

import src.fftutils as fftutils
from src.domains import Domain

class ODE:
    def rhs(self, state, t):
        raise NotImplementedError

@dataclasses.dataclass
class CahnHilliardSIFFTR(ODE):

    domain: Domain
    kappa: float
    mu: Callable
    D: Callable
    smooth: bool = False
    space: str = "R"

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

    def rhs(self, state, t):
        state_hat = self.fft(state)
        tmp = self.fft(self.mu(state)) - self.kappa * (self.two_pi_i_k_2) * state_hat
        tmpx = self.fft(self.D(state) * self.ifft(self.two_pi_i_kx * tmp))
        tmpy = self.fft(self.D(state) * self.ifft(self.two_pi_i_ky * tmp))
        return self.ifft(self.two_pi_i_kx * tmpx + self.two_pi_i_ky * tmpy).real