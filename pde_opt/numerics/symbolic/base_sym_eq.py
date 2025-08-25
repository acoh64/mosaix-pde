from abc import ABC, abstractmethod
from typing import TypeVar

State = TypeVar("State")


class BaseSymbolicEquation(ABC):
    """Base class for symbolic equations."""

    @abstractmethod
    def u_exact(self, t: float) -> State:
        """Exact solution for the equation"""
        raise NotImplementedError
    
    @abstractmethod
    def rhs_exact(self, t: float) -> State:
        """Exact RHS for the equation"""
        raise NotImplementedError
    
