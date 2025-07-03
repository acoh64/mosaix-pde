import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class DiffusionLegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        self.leg_poly = ExpLegendrePolynomials(self.max_degree)

    def __call__(self, params, inputs):
        return self.leg_poly(params, 2.0 * inputs - 1.0)


@dataclasses.dataclass
class ChemicalPotentialLegendrePolynomials:

    max_degree: int

    def __post_init__(self):
        self.leg_poly = LegendrePolynomialsRecurrence(self.max_degree)

    def __call__(self, params, inputs):
        return self.leg_poly(params, 2.0 * inputs - 1.0)


@dataclasses.dataclass
class ExpLegendrePolynomials:

    max_degree: int

    def __post_init__(self):
        leg_poly = LegendrePolynomialsRecurrence(self.max_degree)
        self.func = jax.jit(lambda p, x: jnp.exp(leg_poly(p, x)))

    def __call__(self, params, inputs):
        return self.func(params, inputs)


@dataclasses.dataclass
class LegendrePolynomialsRecurrence:

    max_degree: int

    def __post_init__(self):
        self.func = jax.jit(self._evaluate_legendre)

    def _evaluate_legendre(self, params, x):
        # Compute all Legendre polynomials up to max_degree
        polynomials = self._compute_all_legendre(x)

        # Take only the coefficients we need based on params length
        n_params = min(self.max_degree + 1, len(params))
        used_polys = polynomials[:n_params]
        used_params = params[:n_params]

        # Reshape params to allow broadcasting: (21,) -> (21,1,1,1)
        reshaped_params = used_params[:, None, None, None]

        # Now multiplication will broadcast correctly
        weighted_polys = reshaped_params * used_polys

        # Sum along the first dimension
        return jnp.sum(weighted_polys, axis=0)

    def _compute_all_legendre(self, x):
        """Compute all Legendre polynomials up to max_degree using jax.scan."""
        # Initialize with P_0(x) and P_1(x)
        p0 = jnp.ones_like(x)
        p1 = x

        # Define scan function to compute next polynomial using recurrence relation
        def scan_fn(carry, n):
            p_prev, p_curr = carry
            # (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
            p_next = ((2 * n + 1) * x * p_curr - n * p_prev) / (n + 1)
            return (p_curr, p_next), p_next

        # Run scan to compute polynomials P_2 through P_max_degree
        init_carry = (p0, p1)
        _, higher_polys = jax.lax.scan(
            scan_fn, init_carry, jnp.arange(1, self.max_degree)
        )

        # Combine P_0, P_1, and higher polynomials
        return jnp.vstack([p0[None, :], p1[None, :], higher_polys])

    def __call__(self, params, inputs):
        return self.func(params, inputs)