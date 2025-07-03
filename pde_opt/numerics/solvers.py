import diffrax as dfx
import jax
from typing import Callable


class SemiImplicitFourierSpectral(dfx.AbstractSolver):

    A: float
    two_pi_i_k_4: jax.Array
    kappa: float
    ifft: Callable
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)

        euler_y1 = y0 + δt * f0
        tmp = 1.0 + self.A * δt * self.kappa * self.two_pi_i_k_4
        y1 = y0 + δt * f0 / tmp

        y_error = self.ifft(y1).real - self.ifft(euler_y1).real
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class SemiImplicitFourierSpectralR(dfx.AbstractSolver):

    A: float
    two_pi_i_k_4: jax.Array
    kappa: float
    fft: Callable
    ifft: Callable
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)

        euler_y1 = y0 + δt * f0
        tmp = 1.0 + self.A * δt * self.kappa * self.two_pi_i_k_4
        y1 = y0 + δt * self.ifft(self.fft(f0) / tmp).real

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)