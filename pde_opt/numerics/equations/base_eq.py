"""
This module contains the base equation classes for the PDEs.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

State = TypeVar("State")


class BaseEquation(ABC):
    """Base class for time-dependent PDE equations.

    Abstract base class for time-dependent PDE equations of the form

    .. math::
        \\frac{d}{dt} \\text{state} = F(\\text{state}, t)

    where state is the state of the system and t is the time.

    Subclasses should implement the rhs method, which returns the right hand side of the equation.
    """

    @abstractmethod
    def rhs(self, state: State, t: float) -> State:
        """Right hand side of the equation."""
        raise NotImplementedError


class TimeSplittingEquation(BaseEquation):
    """Time splitting equation.

    Time splitting equation of the form

    .. math::
        \\frac{d}{dt} \\text{state} = A(\\text{state}, t) + B(\\text{state}, t)

    where A(state, t) and B(state, t) are the A and B terms of the equation.

    Subclasses should implement the A_terms and B_terms methods, which return the A and B terms of the equation.
    """

    @abstractmethod
    def A_terms(self, state: State, t: float) -> State:
        """A terms of the equation."""
        raise NotImplementedError

    @abstractmethod
    def B_terms(self, state: State, t: float) -> State:
        """B terms of the equation."""
        raise NotImplementedError
