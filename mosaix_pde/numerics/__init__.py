"""Numerical methods for PDEs."""

# Domains and shapes
from .domains import Domain

# Equations
from .equations import (
    AllenCahn2DPeriodic,
    AllenCahn2DSmoothedBoundary,
    CahnHilliard2DPeriodic,
    CahnHilliard2DSmoothedBoundary,
    CahnHilliard3DPeriodic,
    GPE2DTSControl,
    GPE2DTSRot,
)

# Functions
from .functions import (
    ChemicalPotentialLegendrePolynomials,
    DiffusionLegendrePolynomials,
    LegendrePolynomialExpansion,
    Mixer2d,
    PeriodicCNN,
)
from .shapes import Shape

# Solvers
from .solvers import SemiImplicitFourierSpectral, StrangSplitting

__all__ = [
    "AllenCahn2DPeriodic",
    "AllenCahn2DSmoothedBoundary",
    "CahnHilliard2DPeriodic",
    "CahnHilliard2DSmoothedBoundary",
    "CahnHilliard3DPeriodic",
    "ChemicalPotentialLegendrePolynomials",
    "DiffusionLegendrePolynomials",
    "Domain",
    "GPE2DTSControl",
    "GPE2DTSRot",
    "LegendrePolynomialExpansion",
    "Mixer2d",
    "PeriodicCNN",
    "SemiImplicitFourierSpectral",
    "Shape",
    "StrangSplitting",
]
