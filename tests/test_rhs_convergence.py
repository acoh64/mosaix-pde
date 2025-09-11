import numpy as np
import sympy as sp
import jax

from pde_opt.numerics.equations import AllenCahn2DPeriodic, CahnHilliard2DPeriodic
from pde_opt.numerics.symbolic.allen_cahn_sym import SymbolicAllenCahn2DPeriodic
from pde_opt.numerics.symbolic.cahn_hilliard_sym import SymbolicCahnHilliard2DPeriodic
from pde_opt.numerics.utils.testing import check_convergence

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


def test_rhs_convergence_allen_cahn():
    x, y, t = sp.symbols("x y t", real=True)
    u_star = sp.sin(2 * x) * sp.cos(3 * y) * sp.exp(-0.7 * t)

    def mu_sym(u):
        return u**3 - u

    def R_sym(u):
        return 1 + u**2

    Ns = [32, 64, 128, 256, 512]

    numeric_args = {"kappa": 1e-2, "mu": mu_sym, "R": R_sym, "derivs": "fd"}
    symbolic_args = {"kappa": 1e-2, "mu_sym": mu_sym, "R_sym": R_sym, "u_star": u_star}

    dx, errors = check_convergence(
        AllenCahn2DPeriodic,
        SymbolicAllenCahn2DPeriodic,
        numeric_args,
        symbolic_args,
        Ns,
        2 * np.pi,
    )

    # Fit slope of log(errors) vs log(dx) to check convergence order
    log_dx = np.log(dx)
    log_errors = np.log(errors)
    slope, _ = np.polyfit(log_dx, log_errors, 1)

    # Check slope is close to expected order (2)
    np.testing.assert_allclose(slope, 2.0, rtol=0.1)


def test_rhs_convergence_cahn_hilliard():
    x, y, t = sp.symbols("x y t", real=True)
    u_star = sp.sin(2 * x) * sp.cos(3 * y) * sp.exp(-0.7 * t)

    def mu_sym(u):
        return u**3 - u

    def D_sym(u):
        return 1 + u**2

    Ns = [32, 64, 128, 256, 512]

    numeric_args = {"kappa": 1e-2, "mu": mu_sym, "D": D_sym, "derivs": "fd"}
    symbolic_args = {"kappa": 1e-2, "mu_sym": mu_sym, "D_sym": D_sym, "u_star": u_star}

    dx, errors = check_convergence(
        CahnHilliard2DPeriodic,
        SymbolicCahnHilliard2DPeriodic,
        numeric_args,
        symbolic_args,
        Ns,
        2 * np.pi,
    )

    # Fit slope of log(errors) vs log(dx) to check convergence order
    log_dx = np.log(dx)
    log_errors = np.log(errors)
    slope, _ = np.polyfit(log_dx, log_errors, 1)

    # Check slope is close to expected order (2)
    np.testing.assert_allclose(slope, 2.0, rtol=0.1)
