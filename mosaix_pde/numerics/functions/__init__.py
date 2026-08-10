"""Function representations for PDEs."""

from .cnn import PeriodicCNN
from .legendre import (
    ChemicalPotentialLegendrePolynomials,
    DiffusionLegendrePolynomials,
    LegendrePolynomialExpansion,
)
from .mixer_mlp import Mixer2d

__all__ = [
    "ChemicalPotentialLegendrePolynomials",
    "DiffusionLegendrePolynomials",
    "LegendrePolynomialExpansion",
    "Mixer2d",
    "PeriodicCNN",
]
