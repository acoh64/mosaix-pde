from gymnasium.envs.registration import register

register(
    id='AdvectionDiffusion-v0',
    entry_point='pde_opt.environments.advection_diffusion_env:AdvectionDiffusionEnv',
)

from .numerics.equations.allen_cahn import AllenCahn2DPeriodic
from .numerics.equations.cahn_hilliard import CahnHilliard2DPeriodic
from .numerics.equations.gross_pitaevskii import GPE2DTSControl, GPE2DTSRot
from .numerics.symbolic.allen_cahn_sym import SymbolicAllenCahn2DPeriodic
from .numerics.utils.testing import test_convergence, plot_convergence