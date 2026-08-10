"""PDE equation classes."""

from .allen_cahn import AllenCahn2DPeriodic, AllenCahn2DSmoothedBoundary
from .base_eq import BaseEquation
from .cahn_hilliard import (
    CahnHilliard2DPeriodic,
    CahnHilliard2DSmoothedBoundary,
    CahnHilliard3DPeriodic,
)
from .gross_pitaevskii import GPE2DTSControl, GPE2DTSRot

__all__ = [
    "AllenCahn2DPeriodic",
    "AllenCahn2DSmoothedBoundary",
    "BaseEquation",
    "CahnHilliard2DPeriodic",
    "CahnHilliard2DSmoothedBoundary",
    "CahnHilliard3DPeriodic",
    "GPE2DTSControl",
    "GPE2DTSRot",
]
