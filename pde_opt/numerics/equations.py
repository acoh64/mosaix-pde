from abc import ABC, abstractmethod
import dataclasses
from typing import Callable, TypeVar, Union

import jax.numpy as jnp
import equinox as eqx

import pde_opt.numerics.utils.fft_utils as fftutils
from pde_opt.numerics import domains

State = TypeVar("State")


class BaseEquation(ABC):
    @abstractmethod
    def rhs(self, state: State, t: float) -> State:
        raise NotImplementedError


class TimeSplittingEquation(BaseEquation):
    @abstractmethod
    def A_terms(self, state: State, t: float) -> State:
        raise NotImplementedError

    @abstractmethod
    def B_terms(self, state: State, t: float) -> State:
        raise NotImplementedError


@dataclasses.dataclass
class CahnHilliardSIFFT(BaseEquation):
    domain: domains.Domain
    kappa: float
    mu: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
    D: Union[Callable, eqx.Module]  # Can be a callable or Equinox module
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
class GPE2DTSControl(TimeSplittingEquation):
    domain: domains.Domain
    k: float
    e: float
    lights: Callable
    trap_factor: float = 1.0
    smooth: bool = False

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.two_pi_i_kx_2 = (self.two_pi_i_kx) ** 2
        self.two_pi_i_ky_2 = (self.two_pi_i_ky) ** 2
        self.two_pi_i_k_2 = self.two_pi_i_kx_2 + self.two_pi_i_ky_2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
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


@dataclasses.dataclass
class AdvectionDiffusion2D(BaseEquation):
    """
    2D Advection-Diffusion equation with periodic boundary conditions.
    du/dt + div(b(x, y, t) * u) = D * Laplacian(u)
    where b is a user-supplied velocity field (callable returning (bx, by)),
    and D is the diffusion coefficient (float or callable).
    """

    domain: domains.Domain
    b: Callable  # b(t, x, y) -> (bx, by)
    D: float | Callable  # Diffusion coefficient (can be callable or float)
    smooth: bool = False

    def __post_init__(self):
        self.kx, self.ky = self.domain.fft_mesh()
        self.two_pi_i_kx = 2j * jnp.pi * self.kx
        self.two_pi_i_ky = 2j * jnp.pi * self.ky
        self.k2 = (self.kx * 2 * jnp.pi) ** 2 + (self.ky * 2 * jnp.pi) ** 2
        self.fft = fftutils.truncated_fft_2x_2D if self.smooth else jnp.fft.fftn
        self.ifft = fftutils.padded_ifft_2x_2D if self.smooth else jnp.fft.ifftn
        self.xmesh, self.ymesh = self.domain.mesh()

    def rhs(self, state, t):
        # Advection velocity field at this time/position
        bx, by = self.b(t, self.xmesh, self.ymesh)
        # Advection term: div(b * u)
        adv_x = self.fft(bx * state)
        adv_y = self.fft(by * state)
        div_bu_hat = self.two_pi_i_kx * adv_x + self.two_pi_i_ky * adv_y
        advection_term = self.ifft(div_bu_hat).real
        # Diffusion term
        Dval = self.D(state) if callable(self.D) else self.D
        state_hat = self.fft(state)
        diffusion_term = self.ifft(-Dval * self.k2 * state_hat).real
        return -advection_term + diffusion_term
