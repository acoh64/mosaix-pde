import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
import dataclasses
class LegendrePolynomialExpansion(eqx.Module):
    params: jax.Array  # shape (max_degree+1,)
    max_degree: int

    def __init__(self, params: jax.Array):
        super().__init__()
        self.params = params
        self.max_degree = len(params) - 1

    def __call__(self, inputs):
        # Inputs are assumed to be in [-1, 1]
        result = self.params[0] * jnp.ones_like(inputs)
        if self.max_degree >= 1:
            result += self.params[1] * inputs
        p_prev = jnp.ones_like(inputs)
        p_curr = inputs
        for n in range(2, self.max_degree + 1):
            p_next = ((2 * n - 1) * inputs * p_curr - (n - 1) * p_prev) / n
            result += self.params[n] * p_next
            p_prev, p_curr = p_curr, p_next
        return result

class DiffusionLegendrePolynomials(eqx.Module):
    expansion: LegendrePolynomialExpansion

    def __init__(self, params: jax.Array):
        super().__init__()
        self.expansion = LegendrePolynomialExpansion(params)

    @eqx.filter_jit
    def __call__(self, inputs):
        # Scale inputs to [-1, 1] and apply exp to ensure positivity
        scaled_inputs = 2.0 * inputs - 1.0
        return jnp.exp(self.expansion(scaled_inputs))

class ChemicalPotentialLegendrePolynomials(eqx.Module):
    expansion: LegendrePolynomialExpansion
    prior_fn: Callable 

    def __init__(self, params: jax.Array, prior_fn: Callable = None):
        super().__init__()
        self.expansion = LegendrePolynomialExpansion(params)
        self.prior_fn = prior_fn

    @eqx.filter_jit
    def __call__(self, inputs):
        # Scale inputs to [-1, 1]
        scaled_inputs = 2.0 * inputs - 1.0
        result = self.expansion(scaled_inputs)
        if self.prior_fn is not None:
            result += self.prior_fn(inputs)
        return result
    




@dataclasses.dataclass
class LegendrePolynomials:
    max_degree: int

    def __post_init__(self):
        if self.max_degree == 0:
            self.func = jax.jit(lambda p, x: p[0] * self.T0(x))
        elif self.max_degree == 1:
            self.func = jax.jit(lambda p, x: p[0] * self.T0(x) + p[1] * self.T1(x))
        elif self.max_degree == 2:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x) + p[1] * self.T1(x) + p[2] * self.T2(x)
            )
        elif self.max_degree == 3:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
            )
        elif self.max_degree == 4:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
            )
        elif self.max_degree == 5:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
            )
        elif self.max_degree == 6:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
            )
        elif self.max_degree == 7:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
            )
        elif self.max_degree == 8:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
            )
        elif self.max_degree == 9:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
                + p[9] * self.T9(x)
            )
        elif self.max_degree == 10:
            self.func = jax.jit(
                lambda p, x: p[0] * self.T0(x)
                + p[1] * self.T1(x)
                + p[2] * self.T2(x)
                + p[3] * self.T3(x)
                + p[4] * self.T4(x)
                + p[5] * self.T5(x)
                + p[6] * self.T6(x)
                + p[7] * self.T7(x)
                + p[8] * self.T8(x)
                + p[9] * self.T9(x)
                + p[10] * self.T10(x)
            )

    def __call__(self, params, inputs):
        return self.func(params, inputs)

    def T0(self, x):
        return 1.0 * jnp.ones_like(x)

    def T1(self, x):
        return x

    def T2(self, x):
        return 0.5 * (3 * x**2 - 1.0)

    def T3(self, x):
        return 0.5 * (5 * x**3 - 3 * x)

    def T4(self, x):
        return 0.125 * (35 * x**4 - 30 * x**2 + 3)

    def T5(self, x):
        return 0.125 * (63 * x**5 - 70 * x**3 + 15 * x)

    def T6(self, x):
        return 0.0625 * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5)

    def T7(self, x):
        return 0.0625 * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x)

    def T8(self, x):
        return 0.0078125 * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35)

    def T9(self, x):
        return 0.0078125 * (
            12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x
        )

    def T10(self, x):
        return 0.00390625 * (
            46189 * x**10
            - 109395 * x**8
            + 90090 * x**6
            - 30030 * x**4
            + 3465 * x**2
            - 63
        )