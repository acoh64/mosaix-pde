"""Public API for pde_opt."""

# Core PDE model and environment
from .pde_model import PDEModel
from .pde_env import PDEEnv

# Numerics - Equations
from .numerics.equations.allen_cahn import (
    AllenCahn2DPeriodic,
    AllenCahn2DSmoothedBoundary,
)
from .numerics.equations.cahn_hilliard import (
    CahnHilliard2DPeriodic,
    CahnHilliard3DPeriodic,
    CahnHilliard2DSmoothedBoundary,
)
from .numerics.equations.gross_pitaevskii import (
    GPE2DTSControl,
    GPE2DTSRot,
)
from .numerics.equations.base_eq import BaseEquation

# Numerics - Domains and Shapes
from .numerics.domains import Domain
from .numerics.shapes import Shape

# Numerics - Functions
from .numerics.functions.cnn import (
    PeriodicCNN,
)
from .numerics.functions.legendre import (
    LegendrePolynomialExpansion,
    DiffusionLegendrePolynomials,
    ChemicalPotentialLegendrePolynomials,
)
from .numerics.functions.mixer_mlp import (
    Mixer2d,
)

# Numerics - Solvers
from .numerics.solvers import (
    SemiImplicitFourierSpectral,
    StrangSplitting,
)

__all__ = [
    # Core classes
    "PDEModel",
    "PDEEnv",
    # Equations
    "BaseEquation",
    "AllenCahn2DPeriodic",
    "AllenCahn2DSmoothedBoundary",
    "CahnHilliard2DPeriodic",
    "CahnHilliard3DPeriodic",
    "CahnHilliard2DSmoothedBoundary",
    "GPE2DTSControl",
    "GPE2DTSRot",
    # Domains and Shapes
    "Domain",
    "Shape",
    # Functions
    "PeriodicCNN",
    "LegendrePolynomialExpansion",
    "LegendrePolynomialExpansion2D",
    "DiffusionLegendrePolynomials",
    "ChemicalPotentialLegendrePolynomials",
    "MixerBlock",
    "MixerMLP",
    # Solvers
    "SemiImplicitFourierSpectral",
    "StrangSplitting",
]
