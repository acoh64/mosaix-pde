from typing import Type, Dict, Any, List

from .numerics.equations import BaseEquation
from .numerics import domains
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
        """Class for optimizing parameters or functions of a PDE.

        Args:
            equation_type: The class of the equation to optimize (from numerics/equations)
            domain: The domain to use for the equation
            solver_type: The class of the solver to use for timestepping
        """
        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type
        self.check_equation_solver_compatibility()

    def check_equation_solver_compatibility(self):
        """Check that equation type has all required attributes specified by solver.

        Raises:
            ValueError: If equation is missing any required attributes from solver.
        """
        # Get required attributes from solver if they exist
        if not hasattr(self.solver_type, "required_equation_attrs"):
            return

        solver_required_attrs = self.solver_type.required_equation_attrs

        # Check each required attribute exists in equation class
        missing_attrs = []
        for attr in solver_required_attrs:
            if not hasattr(self.equation_type, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            raise ValueError(
                f"Equation type {self.equation_type.__name__} is missing required "
                f"attributes for solver {self.solver_type.__name__}: {missing_attrs}"
            )

    def _prepare_solver_params(self, solver_parameters, equation):
        """Prepare solver parameters by extracting required equation attributes."""
        # Create the equation with the given parameters

        full_solver_params = solver_parameters.copy()
        if hasattr(self.solver_type, "required_equation_attrs"):
            for attr_name in self.solver_type.required_equation_attrs:
                full_solver_params[attr_name] = getattr(equation, attr_name)

        return full_solver_params

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

        Returns:
            The solution to the equation at the given times in saveat
        """

        # Initialize the equation with the given parameters
        equation = self.equation_type(domain=self.domain, **parameters)

        full_solver_params = self._prepare_solver_params(solver_parameters, equation)

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
        data_residual = (
            values - pred[1:]
        )  # pred[0] is initial, values aligns with pred[1:]

        return data_residual

    def regularization(
        self,
        parameters,
        weights,
        lambda_reg,
    ):
        """
        Compute regularization for a single initial condition.
        Args:
            parameters: parameters for the equation
            weights: regularization weights
            lambda_reg: regularization coefficient
        Returns:
            reg: scalar
        """
        # Loop through weights keys and apply regularization to corresponding parameters
        reg = 0.0

        # Filter out None values and only process valid arrays
        def safe_weighted_square(w, v):
            if eqx.is_inexact_array_like(w) and eqx.is_inexact_array_like(v):
                return jax.numpy.sum(w * v**2)
            return 0.0

        for key in weights.keys():
            # Use tree_map to handle nested structures within this key
            reg += lambda_reg * jax.tree_util.tree_reduce(
                jax.numpy.add,
                jax.tree_util.tree_map(
                    safe_weighted_square,
                    weights[key],
                    parameters[key],
                    is_leaf=lambda x: x is None,
                ),
            )

        return reg

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
        single = partial(
            self.residual_single, parameters, solver_parameters, ts=ts, adjoint=adjoint
        )
        batch_residuals = eqx.filter_vmap(
            lambda y0, val: single(y0, val), in_axes=(0, 0)
        )(y0s, values)

        reg = self.regularization(parameters, weights, lambda_reg)

        return batch_residuals, reg

    def mse(
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
        Compute the mean squared error for a single initial condition.
        """
        batch_residuals, reg = self.residuals(
            parameters,
            y0s__values,
            solver_parameters,
            ts,
            weights,
            lambda_reg,
            adjoint=adjoint,
        )
        return jnp.mean(batch_residuals**2) + reg

    def train(
        self,
        data,
        inds,
        opt_parameters,
        other_parameters,
        solver_parameters,
        weights,
        lambda_reg,
        method="least_squares",
    ):
        """
        Train the model on the given data.
        Args:
            data: The data to train on
            inds: The indices of the data to train on
            init_parameters: The initial parameters for the model
        """

        # TODO: might need to make it so all parameters you want to optimize are jax arrays

        y0s = jnp.array([data["ys"][ind[0]] for ind in inds])
        values = jnp.array(
            [
                jnp.array([data["ys"][ind[i]] for i in range(1, len(ind))])
                for ind in inds
            ]
        )
        ts = jnp.array(
            [
                data["ts"][inds[0][i]] - data["ts"][inds[0][0]]
                for i in range(len(inds[0]))
            ]
        )

        residuals_ = partial(
            self.residuals,
            solver_parameters=solver_parameters,
            weights=weights,
            lambda_reg=lambda_reg,
            ts=ts,
        )

        opt_params, opt_params_static = eqx.partition(
            opt_parameters, eqx.is_inexact_array_like
        )

        if method == "least_squares":

            def residuals_wrapper(_opt_params, args):
                full_params = eqx.combine(_opt_params, opt_params_static)
                return residuals_({**full_params, **other_parameters}, args)

            solver = optx.LevenbergMarquardt(
                rtol=1e-8,
                atol=1e-8,
                verbose=frozenset({"step", "accepted", "loss", "step_size"}),
            )

            sol = optx.least_squares(
                residuals_wrapper, solver, opt_params, args=(y0s, values)
            )

            res = eqx.combine(sol.value, opt_params_static)

            return {**res, **other_parameters}

        elif method == "mse":

            def mse_wrapper(_opt_params, args):
                full_params = eqx.combine(_opt_params, opt_params_static)
                return self.mse(
                    {**full_params, **other_parameters},
                    args,
                    solver_parameters,
                    ts,
                    weights,
                    lambda_reg,
                    adjoint=dfx.RecursiveCheckpointAdjoint(),
                )

            solver = optx.BFGS(
                rtol=1e-8,
                atol=1e-8,
                verbose=frozenset({"step", "accepted", "loss", "step_size"}),
            )

            sol = optx.minimise(mse_wrapper, solver, opt_params, args=(y0s, values))

            res = eqx.combine(sol.value, opt_params_static)

            return {**res, **other_parameters}
