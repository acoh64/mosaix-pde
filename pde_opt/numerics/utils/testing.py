"""
This module contains helper functions for testing the PDEs.
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Type
from ..domains import Domain
from ..equations.base_eq import BaseEquation
from ..symbolic.base_sym_eq import BaseSymbolicEquation


def l2_rel_err(numeric, symbolic):
    """L2 relative error between numeric and symbolic"""
    numeric = np.asarray(numeric)
    symbolic = np.asarray(symbolic)
    return np.sqrt(np.sum((numeric - symbolic) ** 2)) / np.sqrt(np.sum(symbolic**2))


def test_convergence(
    numeric: Type[BaseEquation],
    symbolic: Type[BaseSymbolicEquation],
    numeric_args,
    symbolic_args,
    Ns,
    L,
):
    """Test convergence of a numeric equation against a symbolic equation.

    Args:
        numeric: Numeric equation class
        symbolic: Symbolic equation class
        numeric_args: Arguments for the numeric equation
        symbolic_args: Arguments for the symbolic equation
        Ns: List of grid sizes to test
        L: Length of the domain

    Returns:
        List of grid sizes and errors
    """

    errors = []
    dxs = []

    for N in Ns:
        domain = Domain((N, N), ((-L / 2, L / 2), (-L / 2, L / 2)), "dimensionless")
        numeric_args["domain"] = domain
        symbolic_args["domain"] = domain

        numeric_eq = numeric(**numeric_args)
        symbolic_eq = symbolic(**symbolic_args)

        u_exact = symbolic_eq.u_exact(0)
        F_numeric = numeric_eq.rhs(u_exact, 0)
        F_symbolic = symbolic_eq.rhs_exact(0)

        errors.append(l2_rel_err(F_numeric, F_symbolic))
        dxs.append(domain.dx[0])

    return dxs, errors


def plot_convergence(dx, err, orders=(0.5, 1.0, 1.5, 2.0), anchor="min"):
    """
    Log–log plot of error vs dx with dotted reference slopes.

    anchor: 'min' -> anchor reference lines at smallest dx point,
            'max' -> anchor at largest dx point.
    """
    dx = np.asarray(dx, float)
    err = np.asarray(err, float)
    i = np.argsort(dx)  # ensure monotone in dx
    dx, err = dx[i], err[i]

    # main curve
    plt.figure()
    plt.loglog(dx, err, "o-", label="measured")

    # choose anchor point
    x0, y0 = (dx[0], err[0]) if anchor == "min" else (dx[-1], err[-1])
    xref = np.array([dx[0], dx[-1]])

    # add dotted reference lines of given orders, calibrated to pass through (x0, y0)
    for q in orders:
        yref = y0 * (xref / x0) ** q
        plt.loglog(xref, yref, linestyle=":", label=f"order {q:g}")

    # optional: global slope fit (for quick eyeball)
    slope = np.polyfit(np.log(dx), np.log(err), 1)[0]

    plt.xlabel(r"$\Delta x$")
    plt.ylabel("error")
    plt.title(f"Convergence (global slope ≈ {abs(slope):.3f})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
