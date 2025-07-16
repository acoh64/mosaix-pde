from typing import Type, Dict, Any, List

from pde_opt.numerics.equations import BaseEquation
from pde_opt.numerics import domains
import diffrax as dfx
import jax
import jax.numpy as jnp
import optimistix as optx
from functools import partial
import equinox as eqx


class OptimizationModel:
    def __init__(
        self,
        equation_type: Type[BaseEquation],
        domain: domains.Domain,
        solver_type: Type[dfx.AbstractSolver],
    ):
        """
        equation_type: The class of the equation to optimize (from numerics/equations.py)
        domain: The domain to use for the equation
        parameters: Dictionary of all parameters (except domain) to instantiate the equation
        optimize_params: List of parameter names to optimize over
        solver_type: The class of the solver to use (e.g., diffrax.Tsit5)
        """
        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type

    def solve(
        self,
        parameters: Dict[str, Any],
        y0,
        ts,
        solver_parameters: Dict[str, Any],
        adjoint=dfx.ForwardMode(),
        dt0=0.000001,
        max_steps=1000000,
    ):
        """
        Solve the equation with given parameters.
        
        Args:
            parameters: Dictionary of parameters to use for the equation
            y0: Initial condition
            saveat: SaveAt object specifying when to save solution
            solver_parameters: Dictionary of parameters for the solver
            solver_equation_attrs: List of equation attribute names to pass to solver
                                 (if None, will try to get from solver.required_equation_attrs)
            adjoint: Adjoint mode for differentiation
            dt0: Initial time step
            max_steps: Maximum number of steps
        """
        # Initialize the equation with the given parameters
        equation = self.equation_type(domain=self.domain, **parameters)
        
        
        # Try to get required attributes from solver class
        if hasattr(self.solver_type, 'required_equation_attrs'):
            solver_equation_attrs = self.solver_type.required_equation_attrs
        else:
            solver_equation_attrs = []
        
        # Prepare solver parameters including equation attributes if specified
        full_solver_params = solver_parameters.copy()
        if solver_equation_attrs:
            for attr_name in solver_equation_attrs:
                if hasattr(equation, attr_name):
                    full_solver_params[attr_name] = getattr(equation, attr_name)
        
        # Initialize the solver with solver_parameters and equation attributes
        solver = self.solver_type(**full_solver_params)
        
        # Solve with diffrax
        solution = dfx.diffeqsolve(
            dfx.ODETerm(jax.jit(lambda t, y, args: equation.rhs(y, t))),
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
        )
        
        return solution.ys

    def residual_single(
        self,
        parameters,
        solver_parameters,
        y0,
        values,
        ts,
        adjoint=dfx.ForwardMode(),
    ):
        """
        Compute residuals and regularization for a single initial condition.
        Args:
            parameters: parameters for the equation
            y0: initial condition, shape (Nx, Ny)
            values: observed values, shape (timepoints, Nx, Ny)
            ts: timepoints, shape (timepoints,)
            weights: regularization weights, pytree matching parameters
            lambda_reg: regularization coefficient
            adjoint: adjoint mode
        Returns:
            data_residual: (timepoints, Nx, Ny)
            reg: scalar
        """
        pred = self.solve(parameters, y0, ts, solver_parameters, adjoint=adjoint)
        data_residual = values - pred[1:]  # pred[0] is initial, values aligns with pred[1:]

        return data_residual

    def residuals(
        self,
        parameters,
        y0s__values,
        solver_parameters,
        ts,
        weights,
        lambda_reg,
        adjoint=dfx.ForwardMode(),
    ):
        """
        Compute batched residuals and total regularization.
        Args:
            parameters: parameters for the equation
            y0s: (batch_dim, Nx, Ny)
            values: (batch_dim, timepoints, Nx, Ny)
            ts: (timepoints,)
            weights: pytree matching parameters
            lambda_reg: regularization coefficient
            adjoint: adjoint mode
        Returns:
            batch_residuals: (batch_dim, timepoints, Nx, Ny)
            total_reg: scalar
        """
        y0s, values = y0s__values
        single = partial(self.residual_single, parameters, solver_parameters, ts=ts, adjoint=adjoint)
        batch_residuals = eqx.filter_vmap(
            lambda y0, val: single(y0, val),
            in_axes=(0, 0)
        )(y0s, values)

        reg = lambda_reg * jax.tree_util.tree_reduce(
            jax.numpy.add,
            jax.tree_util.tree_map(
                lambda w, v: None if w is None else (jax.numpy.sum(w * v**2) if isinstance(v, jax.Array) else 0.0),
                weights,
                parameters,
                is_leaf=lambda x: x is None
            ),
        )
        
        return batch_residuals, reg 

    def train(self, data, inds, init_parameters, solver_parameters, weights, lambda_reg):

        y0s = jnp.array([data["ys"][ind[0]] for ind in inds])
        values = jnp.array([jnp.array([data["ys"][ind[i]] for i in range(1, len(ind))]) for ind in inds])
        ts = jnp.array([data["ts"][inds[0][i]] - data["ts"][inds[0][0]] for i in range(len(inds[0]))])

        params_array, params_aux = eqx.partition(init_parameters, eqx.is_array)

        residuals_ = partial(self.residuals, solver_parameters=solver_parameters, weights=weights, lambda_reg=lambda_reg, ts=ts)

        def residuals_wrapper(params_array, args):
            # Recombine array and aux parts
            full_params = eqx.combine(params_array, params_aux)
            return residuals_(full_params, args)

        solver = optx.LevenbergMarquardt(
            rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
        )

        sol = optx.least_squares(residuals_wrapper, solver, params_array, args=(y0s, values))

        full_params = eqx.combine(sol.value, params_aux)
        
        return full_params
        