"""Public API for pde_opt."""

from .numerics.equations.allen_cahn import AllenCahn2DPeriodic
from .numerics.equations.cahn_hilliard import CahnHilliard2DPeriodic
from .numerics.equations.gross_pitaevskii import GPE2DTSControl, GPE2DTSRot
from .numerics.symbolic.allen_cahn_sym import SymbolicAllenCahn2DPeriodic
from .numerics.utils.testing import test_convergence, plot_convergence

__all__ = [
    'AllenCahn2DPeriodic',
    'CahnHilliard2DPeriodic',
    'GPE2DTSControl',
    'GPE2DTSRot',
    'SymbolicAllenCahn2DPeriodic',
    'test_convergence',
    'plot_convergence',
]