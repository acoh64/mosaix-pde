import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pde_opt.numerics.functions import LegendrePolynomialExpansion, DiffusionLegendrePolynomials, ChemicalPotentialLegendrePolynomials

# Force JAX to use CPU for tests
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)

# Standard Python implementation of Legendre polynomials using numpy.polynomial
from numpy.polynomial.legendre import legval

def reference_legendre(params, x):
    # params: array-like, coefficients for P_0, P_1, ..., P_n
    return legval(x, params)

def test_legendre_polynomial_expansion_matches_numpy():
    params = jnp.array([1.0, 0.5, 0.2, 0.1, -0.05, -0.02, 0.01])
    x = jnp.linspace(-1, 1, 20)
    eqx_legendre = LegendrePolynomialExpansion(params)
    y_eqx = eqx_legendre(x)
    y_np = reference_legendre(params, x)
    np.testing.assert_allclose(y_eqx, y_np, rtol=1e-5, atol=1e-7)

def test_diffusion_legendre_polynomials_positive_and_matches_exp():
    params = jnp.array([0.2, -0.1, 0.05, -0.02, 0.01, -0.005, 0.002])
    x = jnp.linspace(0, 1, 20)
    eqx_diff = DiffusionLegendrePolynomials(params)
    # Reference: exp(legendre)
    scaled_x = 2.0 * x - 1.0
    y_np = jnp.exp(reference_legendre(params, scaled_x))
    y_eqx = eqx_diff(x)
    # Check positivity
    assert jnp.all(y_eqx > 0)
    np.testing.assert_allclose(y_eqx, y_np, rtol=1e-5, atol=1e-7)

def test_chemical_potential_legendre_polynomials_matches_legendre():
    params = jnp.array([0.3, 0.1, -0.2, -0.1, 0.45, -2.02, 0.01])
    x = jnp.linspace(0, 1, 20)
    eqx_chem = ChemicalPotentialLegendrePolynomials(params)
    scaled_x = 2.0 * x - 1.0
    y_np = reference_legendre(params, scaled_x)
    y_eqx = eqx_chem(x)
    np.testing.assert_allclose(y_eqx, y_np, rtol=1e-5, atol=1e-7)

def test_chemical_potential_with_prior():
    params = jnp.array([0.3, 0.1, -0.2])
    x = jnp.linspace(0, 1, 20)
    prior_fn = lambda x: 2.0 * x
    eqx_chem = ChemicalPotentialLegendrePolynomials(params, prior_fn=prior_fn)
    scaled_x = 2.0 * x - 1.0
    y_np = reference_legendre(params, scaled_x) + 2.0 * x
    y_eqx = eqx_chem(x)
    np.testing.assert_allclose(y_eqx, y_np, rtol=1e-5, atol=1e-7) 