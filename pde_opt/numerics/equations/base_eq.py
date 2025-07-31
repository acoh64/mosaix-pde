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