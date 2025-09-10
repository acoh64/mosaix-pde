from typing import Type, Dict, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import optimistix as optx
from functools import partial
import equinox as eqx

from .numerics.equations import BaseEquation
from .numerics import domains
from .utils import check_equation_solver_compatibility, prepare_solver_params


class PDEModel:
    """Manage the solving and optimization of partial differential equations (PDEs).

    The PDEModel class provides a unified interface for solving PDEs and optimizing their
    parameters using gradient-based methods. It supports both forward simulation and
    parameter estimation.

    The class is designed to work with JAX-based PDE implementations and leverages
    automatic differentiation for efficient gradient computation during optimization.

    Attributes:
        equation_type (Type[BaseEquation]): The equation class to optimize. Must be a
            subclass of BaseEquation from numerics.equations.
        domain (domains.Domain): The spatial domain for solving the equation. Contains
            grid information, boundary conditions, and coordinate systems.
        solver_type (Type[dfx.AbstractSolver]): The numerical solver class for time
            integration. Must be a subclass of dfx.AbstractSolver and can be existing 
            diffrax solvers like Tsit5 or custom solvers like defined in numerics.solvers.

    For examples, see the documentation notebooks.
    """

    def __init__(
        self,
        equation_type: Type[BaseEquation],
        domain: domains.Domain,
        solver_type: Type[dfx.AbstractSolver],
    ):
        """Initialize the PDE optimization model.

        Args:
            equation_type (Type[BaseEquation]): The equation class to optimize. Must be a
                subclass of BaseEquation.
            domain (domains.Domain): The spatial domain for the PDE. Defines the grid
                resolution, spatial bounds, and coordinate system.
            solver_type (Type[dfx.AbstractSolver]): The numerical solver for time
                integration. Must be a subclass of dfx.AbstractSolver and can be existing diffrax solvers like Tsit5 or custom solvers like
                defined in numerics.solvers.    

        Raises:
            ValueError: If the equation and solver are incompatible (e.g., solver
                requires attributes that the equation doesn't provide).

        Note:
            The solver and equation compatibility is automatically checked during
            initialization. Some solvers require specific attributes from equations
            (e.g., fourier_symbol, fft, ifft for semi-implicit spectral methods).
        """
        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type
        check_equation_solver_compatibility(self.solver_type, self.equation_type)

    def solve(
        self,
        parameters: Dict[str, Any],
        y0,
        ts,
        solver_parameters: Dict[str, Any] = {},
        adjoint=dfx.ForwardMode(),
        dt0=0.000001,
        max_steps=1000000,
        stepsize_controller=dfx.ConstantStepSize(),
    ):
        """Solve the PDE with given parameters and initial conditions.

        This method performs forward simulation of the PDE using the specified solver
        and parameters. The solution is computed at the requested time points and
        returned.

        Args:
            parameters (Dict[str, Any]): Dictionary of equation parameters. Keys should
                match the parameter names expected by the equation class.
            y0: Initial condition array. Shape should match the spatial dimensions of
                the domain.
            ts: Time points at which to save the solution. Should be a 1D array of
                increasing time values. The solver will integrate from ts[0] to ts[-1]
                and return solutions at all specified time points.
            solver_parameters (Dict[str, Any], optional): Additional parameters for the
                solver. These are passed directly to the solver constructor.
            adjoint (dfx.AbstractAdjoint, optional): Adjoint mode for automatic
                differentiation. Defaults to ForwardMode() for forward-mode AD.
                Use RecursiveCheckpointAdjoint() for reverse-mode AD when the number of parameters is large.
            dt0 (float, optional): Initial time step for the solver. Defaults to 1e-6.
                The solver may adapt this step size during integration.
            max_steps (int, optional): Maximum number of integration steps. Defaults to
                1,000,000.
            stepsize_controller (dfx.AbstractStepSizeController, optional): Controller
                for adaptive step sizing. Defaults to ConstantStepSize().

        Returns:
            Solution array with shape (len(ts), *y0.shape). 
        """

        # Initialize the equation with the given parameters
        equation = self.equation_type(domain=self.domain, **parameters)

        full_solver_params = prepare_solver_params(self.solver_type, solver_parameters, equation)

        # Initialize the solver with solver_parameters and equation attributes
        solver = self.solver_type(**full_solver_params)

        # Solve with diffrax
        solution = dfx.diffeqsolve(
            dfx.ODETerm(jax.jit(lambda t, y, args: equation.rhs(y, t))), # TODO: might need to remove jit or change to filter_jit
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
            stepsize_controller=stepsize_controller,
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
        """Compute residuals for a single trajectory.

        This method computes the difference between model predictions and observed data
        for a single initial condition and trajectory. It's used internally by the
        batched residuals computation.

        Args:
            parameters (Dict[str, Any]): Current parameter values for the equation.
            solver_parameters (Dict[str, Any]): Parameters for the numerical solver.
            y0 (jax.Array): Initial condition.
            values (jax.Array): Observed values for computing residuals.
            ts (jax.Array): Time points with shape (timepoints,).
            adjoint (dfx.AbstractAdjoint, optional): Adjoint mode for automatic
                differentiation. Defaults to ForwardMode().

        Returns:
            Residuals array with shape (timepoints, *y0.shape).
                The residuals are computed as: values - predicted[1:] (values should not include the initial condition).
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
        """Compute parameter regularization term.

        This method computes a weighted L2 regularization term for the parameters to
        prevent overfitting and improve parameter stability during optimization. The
        regularization is computed as:

        .. math::
            \\text{reg} = \\lambda \\sum_i w_i p_i^2

        where λ is the regularization coefficient, w_i are the weights, and p_i are
        the parameter values.

        Args:
            parameters (Dict[str, Any]): Current parameter values for the equation.
                Can contain nested structures (pytrees) of parameters.
            weights (Dict[str, Any]): Regularization weights with the same structure
                as parameters. Higher weights penalize large parameter values more
                strongly. None values in weights are ignored.
            lambda_reg (float): Regularization coefficient controlling the overall
                strength of the regularization term.

        Returns:
            Scalar regularization term to be added to the loss function.
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
        """Compute batched residuals and regularization for parameter optimization.

        This method computes the difference between model predictions and observed data
        for multiple trajectories simultaneously, along with parameter regularization.
        It's used internally by the optimization algorithms in the train() method.

        Args:
            parameters (Dict[str, Any]): Current parameter values for the equation.
            y0s__values (Tuple[jax.Array, jax.Array]): Tuple containing:
                - y0s: Batch of initial conditions
                - values: Batch of observed values
            solver_parameters (Dict[str, Any]): Parameters for the numerical solver.
            ts (jax.Array): Time points for the simulation.
            weights (Dict[str, jax.Array]): Regularization weights for each parameter.
                Should match the structure of parameters.
            lambda_reg (float): Regularization coefficient controlling the strength
                of parameter penalties.
            adjoint (dfx.AbstractAdjoint, optional): Adjoint mode for automatic
                differentiation. Defaults to ForwardMode().

        Returns:
            Tuple[jax.Array, float]: Tuple containing:
                - batch_residuals: Residuals array with shape (batch_size, timepoints, *y0.shape)
                - reg: Scalar regularization term
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
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    ):
        """Compute the mean squared error loss for parameter optimization.

        This method computes the mean squared error between model predictions and
        observed data, plus a regularization term. It's used as the objective function
        for the "mse" optimization method in the train() method.

        The loss function is:

        .. math::
            \\text{MSE} = \\text{mean}((\\text{predicted} - \\text{observed})^2) + \\lambda \\cdot \\text{regularization}

        Args:
            parameters (Dict[str, Any]): Current parameter values for the equation.
            y0s__values (Tuple[jax.Array, jax.Array]): Tuple containing:
                - y0s: Batch of initial conditions
                - values: Batch of observed values
            solver_parameters (Dict[str, Any]): Parameters for the numerical solver.
            ts (jax.Array): Time points for the simulation.
            weights (Dict[str, jax.Array]): Regularization weights for each parameter.
                Should match the structure of parameters.
            lambda_reg (float): Regularization coefficient controlling the strength
                of parameter penalties.
            adjoint (dfx.AbstractAdjoint, optional): Adjoint mode for automatic
                differentiation. Defaults to RecursiveCheckpointAdjoint().

        Returns:
            float: Mean squared error loss including regularization term.
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
        # return jnp.sum(batch_residuals**2) + reg

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
        max_steps=100,
    ):
        """Train the model by optimizing parameters to fit observed data.

        This method performs parameter estimation by minimizing the difference between
        model predictions and observed data. It supports two optimization approaches:
        least-squares (which uses the Levenberg-Marquardt algorithm) and mean squared error minimization
        (which uses the BFGS algorithm).

        Args:
            data (Dict[str, List]): Training data dictionary with keys:
                - "ys": List of solution snapshots at different times
                - "ts": List of corresponding time points
                Example: {"ys": [y0, y1, y2, ...], "ts": [0, 0.1, 0.2, ...]}
            inds (List[List[int]]): Indices specifying which data points to use for
                each training trajectory. Each inner list represents a trajectory:
                [initial_time_index, ...intermediate_indices...].
                Example: [[0, 1, 2], [0, 1, 2]] for two trajectories using
                time points 0, 1, 2.
            opt_parameters (Dict[str, jax.Array]): Parameters to optimize.
            other_parameters (Dict[str, Any]): Fixed parameters that won't be optimized.
            solver_parameters (Dict[str, Any]): Parameters for the numerical solver
                during optimization. Passed to the solver constructor.
            weights (Dict[str, jax.Array]): Regularization weights for each parameter.
                Should have the same structure as opt_parameters.
            lambda_reg (float): Regularization coefficient. Controls the strength of
                parameter regularization.
            method (str, optional): Optimization method. Options:
                - "least_squares": Uses Levenberg-Marquardt algorithm with ForwardMode
                  adjoint. Best when parameter number is small (not using neural networks).
                - "mse": Uses BFGS algorithm with RecursiveCheckpointAdjoint. Better
                  when parameter number is large (using neural networks).
            max_steps (int, optional): Maximum number of optimization iterations.
                Defaults to 100.

        Returns:
            Dict[str, Any]: Optimized parameters combined with fixed parameters.
            The returned dictionary contains both the optimized parameters and the
            other_parameters, ready for use in the solve() method.
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
                residuals_wrapper, solver, opt_params, args=(y0s, values), max_steps=max_steps, throw=False
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

            sol = optx.minimise(mse_wrapper, solver, opt_params, args=(y0s, values), max_steps=max_steps, throw=False)

            res = eqx.combine(sol.value, opt_params_static)

            return {**res, **other_parameters}
