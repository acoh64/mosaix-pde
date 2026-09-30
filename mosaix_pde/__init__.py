"""Public API for mosaix_pde."""

# Core PDE model and environment
# Numerics - Domains and Shapes
from .numerics.domains import Domain

# Numerics - Equations
from .numerics.equations.allen_cahn import (
    AllenCahn2DPeriodic,
    AllenCahn2DSmoothedBoundary,
)
from .numerics.equations.base_eq import BaseEquation
from .numerics.equations.cahn_hilliard import (
    CahnHilliard2DPeriodic,
    CahnHilliard2DSmoothedBoundary,
    CahnHilliard3DPeriodic,
)
from .numerics.equations.gross_pitaevskii import (
    GPE2DTSControl,
    GPE2DTSRot,
)

# Numerics - Functions
from .numerics.functions.cnn import (
    PeriodicCNN,
)
from .numerics.functions.legendre import (
    ChemicalPotentialLegendrePolynomials,
    DiffusionLegendrePolynomials,
    LegendrePolynomialExpansion,
)
from .numerics.functions.mixer_mlp import (
    Mixer2d,
)
from .numerics.shapes import Shape

# Numerics - Solvers
from .numerics.solvers import (
    SemiImplicitFourierSpectral,
    StrangSplitting,
)
from .numerics.solvers_rock2 import ROCK2JAX
from .pde_env import PDEEnv
from .pde_model import PDEModel

__all__ = [
    "ROCK2JAX",
    "AllenCahn2DPeriodic",
    "AllenCahn2DSmoothedBoundary",
    "BaseEquation",
    "CahnHilliard2DPeriodic",
    "CahnHilliard2DSmoothedBoundary",
    "CahnHilliard3DPeriodic",
    "ChemicalPotentialLegendrePolynomials",
    "DiffusionLegendrePolynomials",
    "Domain",
    "GPE2DTSControl",
    "GPE2DTSRot",
    "LegendrePolynomialExpansion",
    "LegendrePolynomialExpansion2D",
    "Mixer2d",
    "PDEEnv",
    "PDEModel",
    "PeriodicCNN",
    "SemiImplicitFourierSpectral",
    "Shape",
    "StrangSplitting",
]
