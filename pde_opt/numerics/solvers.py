import diffrax as dfx
import jax
import jax.numpy as jnp
from typing import Callable

class SemiImplicitFourierSpectral(dfx.AbstractSolver):

    A: float
    fourier_symbol: jax.Array
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
        tmp = 1.0 + self.A * δt * self.fourier_symbol
        y1 = y0 + δt * self.ifft(self.fft(f0) / tmp).real

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
    
class StrangSplitting(dfx.AbstractSolver):

    A_term: jax.Array
    dx: float
    fft: Callable
    ifft: Callable
    time_scale: float
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = (t1 - t0) * self.time_scale

        y0_ = y0[...,0] + 1j*y0[...,1]

        exp_A_term = jnp.exp(self.A_term*0.5*δt)

        tmp = self.fft(y0_)*exp_A_term # 1
        tmp = self.ifft(tmp) # 2
        b_term = terms.vf(t0, y0, args)
        tmp = tmp * jnp.exp((b_term[...,0] + 1j*b_term[...,1])*δt) # 3
        tmp /= jnp.sqrt(jnp.sum(jnp.abs(tmp)**2) * self.dx**2) # 4
        tmp = self.fft(tmp) # 5
        tmp = tmp*exp_A_term # 6
        y1_ = self.ifft(tmp) # 7
        y1 = jnp.stack([y1_.real, y1_.imag], axis=-1)
        # TODO: I should be able to change order and reduce number of ffts

        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dfx.RESULTS.successful
        return y1, None, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return NotImplementedError