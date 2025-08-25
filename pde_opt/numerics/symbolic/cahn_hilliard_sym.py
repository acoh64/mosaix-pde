# symbolic/cahn_hilliard_sym.py
from dataclasses import dataclass
from typing import Callable
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import jax.numpy as jnp  # only to return jnp arrays if you like; optional

from .base_sym_eq import BaseSymbolicEquation

@dataclass
class SymbolicCahnHilliard2DPeriodic(BaseSymbolicEquation):
    """Build exact RHS for Cahn–Hilliard equation, used only in tests.
    
    Args:
        domain: Domain of the equation
        kappa: Parameter of the equation
        mu_sym: Symbolic chemical potential
        D_sym: Symbolic mobility
        u_star: Test solution for the equation
    """

    domain: object
    kappa: float
    mu_sym: Callable[[sp.Expr], sp.Expr]  # e.g., lambda u: u**3 - u
    D_sym:  Callable[[sp.Expr], sp.Expr]  # e.g., lambda u: 1
    u_star: sp.Expr                         # test solution u*(x,y,t)

    def __post_init__(self):
        x, y, t = sp.symbols('x y t', real=True)
        u = self.u_star

        u_x  = sp.diff(u, x)
        u_y  = sp.diff(u, y)
        u_xx = sp.diff(u, x, 2)
        u_yy = sp.diff(u, y, 2)

        mu_expr = self.mu_sym(u) - self.kappa * (u_xx + u_yy)
        mu_x, mu_y = sp.diff(mu_expr, x), sp.diff(mu_expr, y)
        rhs_expr = sp.diff(self.D_sym(u) * mu_x, x) + sp.diff(self.D_sym(u) * mu_y, y)

        # Cache fast array-callables
        self._u_fn   = lambdify((x, y, t), sp.simplify(u),        'numpy')
        self._rhs_fn = lambdify((x, y, t), sp.simplify(rhs_expr), 'numpy')

    # ---- Public evaluators for tests ----
    def u_exact(self, t: float):
        """Exact solution for the equation"""
        X, Y = self.domain.mesh()
        return jnp.asarray(self._u_fn(X, Y, float(t)))

    def rhs_exact(self, t: float):
        """Exact RHS for the equation"""
        X, Y = self.domain.mesh()
        return jnp.asarray(self._rhs_fn(X, Y, float(t)))