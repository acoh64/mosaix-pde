"""Function representations for PDEs."""

from .cnn import PeriodicCNN
from .legendre import (
    LegendrePolynomialExpansion,
    DiffusionLegendrePolynomials,
    ChemicalPotentialLegendrePolynomials,
)
from .mixer_mlp import Mixer2d

__all__ = [
    'PeriodicCNN',
    'LegendrePolynomialExpansion',
    'DiffusionLegendrePolynomials',
    'ChemicalPotentialLegendrePolynomials',
    'Mixer2d',
]
