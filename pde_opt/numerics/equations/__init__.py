"""PDE equation classes."""

from .base_eq import BaseEquation
from .allen_cahn import AllenCahn2DPeriodic, AllenCahn2DSmoothedBoundary
from .cahn_hilliard import (
    CahnHilliard2DPeriodic,
    CahnHilliard3DPeriodic,
    CahnHilliard2DSmoothedBoundary,
)
from .gross_pitaevskii import GPE2DTSControl, GPE2DTSRot

__all__ = [
    'BaseEquation',
    'AllenCahn2DPeriodic',
    'AllenCahn2DSmoothedBoundary',
    'CahnHilliard2DPeriodic',
    'CahnHilliard3DPeriodic',
    'CahnHilliard2DSmoothedBoundary',
    'GPE2DTSControl',
    'GPE2DTSRot',
]