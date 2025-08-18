from abc import ABC, abstractmethod
import dataclasses
from typing import Callable, TypeVar, Union

import jax.numpy as jnp
import equinox as eqx

import pde_opt.numerics.utils.fft_utils as fftutils
from pde_opt.numerics import domains

State = TypeVar("State")


class BaseSymbolicEquation(ABC):

    @abstractmethod
    def u_exact(self, t: float) -> State:
        raise NotImplementedError
    
    @abstractmethod
    def rhs_exact(self, t: float) -> State:
        raise NotImplementedError
    
