"""Numerical methods for PDEs."""

# Domains and shapes
from .domains import Domain
from .shapes import Shape

# Equations
from .equations import *

# Functions
from .functions import *

# Solvers
from .solvers import SemiImplicitFourierSpectral, StrangSplitting

__all__ = [
    # Domains and shapes
    'Domain',
    'Shape',
    
    # Equations (imported from .equations)
    'AllenCahn2DPeriodic',
    'AllenCahn2DSmoothedBoundary',
    'CahnHilliard2DPeriodic',
    'CahnHilliard3DPeriodic',
    'CahnHilliard2DSmoothedBoundary',
    'GPE2DTSControl',
    'GPE2DTSRot',
    
    # Functions (imported from .functions)
    'PeriodicCNN',
    'LegendrePolynomialExpansion',
    'DiffusionLegendrePolynomials',
    'ChemicalPotentialLegendrePolynomials',
    'Mixer2d',
    
    # Solvers
    'SemiImplicitFourierSpectral',
    'StrangSplitting',
    
]
