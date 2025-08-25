import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import diffrax
import jax
from typing import Callable, Optional, Type, Dict, Any, Tuple

from .numerics.equations import BaseEquation
from .numerics import domains
from .utils import check_equation_solver_compatibility, prepare_solver_params

# TODO: just make a file in pde_opt instead of in the environments folder. Can get rid of the environments folder.
# TODO: create RL environments that can control multiple parameters at once.


class PDEEnv(gym.Env):
    """
    Manage a reinforcement learning environment for a PDE.

    Args:
        equation_type: Class of the equation to optimize (from numerics/equations)
        domain: Domain to use for the equation
        solver_type: Class of the solver to use for timestepping
        end_time: End time of the simulation
        step_dt: Timestep for each step of the environment
        numeric_dt: Timestep for the numerical solver
        field_dim: Dimension of the field
        state_to_observation_func: Function to convert the state to the observation
        reward_function: Function to compute the reward from the state
        reset_func: Function to reset the environment state
        reset_control_value: Function to reset the control value
        update_control_value: Function to update the control value from the action and the old control value
        update_control_parameter: Function to update the control parameter from the old and new control values
        action_space_config: Configuration of the action space
        static_equation_parameters: Parameters of the equation that are not controlled by the agent
        control_equation_parameter_name: Name of the parameter of the equation that are controlled by the agent
        solver_parameters: Parameters of the solver
    """

    def __init__(
        self,
        equation_type: Type[BaseEquation],
        domain: domains.Domain,
        solver_type: Type[diffrax.AbstractSolver],
        end_time: float,
        step_dt: float,
        numeric_dt: float,
        field_dim: int,
        state_to_observation_func: Callable,
        reward_function: Callable,
        reset_func: Callable,
        reset_control_value,
        update_control_value: Callable,
        update_control_parameter: Callable,
        action_space_config: Dict[str, Any],
        static_equation_parameters: Dict[
            str, Any
        ],  # keys are parameters of the equation that are not controlled by the agent, values are the values of the parameters
        control_equation_parameter_name: str,  # name of the parameter of the equation that are controlled by the agent
        solver_parameters: Dict[str, Any],
    ):
        super().__init__()

        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type
        check_equation_solver_compatibility(self.solver_type, self.equation_type)

        self.end_time = end_time
        self.step_dt = step_dt
        self.numeric_dt = numeric_dt
        self.field_dim = field_dim
        self.reward_function = reward_function
        self.reset_func = reset_func
        self.state_to_observation_func = state_to_observation_func

        # Set up observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(
                1,
                *self.domain.points,
            ),
            dtype=np.uint8,
        )

        # Set up action space based on configuration
        self._setup_action_space(action_space_config)

        self.reset_control_value = reset_control_value
        self.update_control_value = update_control_value
        self.update_control_parameter = update_control_parameter

        self.static_equation_parameters = static_equation_parameters
        self.control_equation_parameter_name = control_equation_parameter_name

        self.solver_parameters = solver_parameters

    def _setup_action_space(self, config: Dict[str, Any]):
        """Set up the action space based on configuration."""
        action_type = config.get("type", "continuous")

        if action_type == "discrete":
            num_actions = config.get("num_actions", 5)
            self.action_space = spaces.Discrete(num_actions)
            self._action_to_direction = config.get("action_mapping", {})
        else:
            # Continuous action space
            action_shape = config.get("shape", (2,))
            low = config.get("low", -1.0)
            high = config.get("high", 1.0)
            self.action_space = spaces.Box(low=low, high=high, shape=action_shape)
            self._action_to_direction = None

    def _initialize_equation_and_solver(self):
        """Initialize the equation and solver with current parameters."""
        # Initialize equation with domain and parameters
        self._equation = self.equation_type(
            domain=self.domain, **self.equation_parameters
        )

        full_solver_params = prepare_solver_params(
            self.solver_type, self.solver_parameters, self._equation
        )

        # Initialize solver
        self._solver = self.solver_type(**full_solver_params)

    def _get_obs(self):
        """Convert state to observation."""
        # return (
        #     np.clip(np.array(self._state), self.state_min_max[0], self.state_min_max[1])
        #     * 255
        # ).astype(np.uint8)[None]
        return self.state_to_observation_func(self._state)

    def _get_info(self):
        return {}

    def _terminate(self):
        """Check if the episode should terminate."""
        return self._time >= self.end_time

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._state = self.reset_func(self.domain, seed=seed)
        else:
            self._state = self.reset_func(self.domain)

        self._time = 0.0

        self._control_value = self.reset_control_value

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.

        Args:
            action: The action to take

        Returns:
            observation, reward, terminated, truncated, info
        """

        offset = (
            action
            if not self.discrete_action_space
            else self._action_to_direction[action]
        )

        old_control_value = self._control_value
        self._control_value = self.update_control_value(
            offset, old_control_value
        )  # TODO: make this update_control_value function

        control_parameter = self.update_control_parameter(
            old_control_value, self._control_value
        )

        full_equation_parameters = {
            **self.static_equation_parameters,
            self.control_equation_parameter_name: control_parameter,
        }

        eq = self.equation_type(domain=self.domain, **full_equation_parameters)

        full_solver_params = prepare_solver_params(
            self.solver_type, self.solver_parameters, eq
        )
        solver = self.solver_type(**full_solver_params)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(jax.jit(lambda t, y, args: eq.rhs(y, t))),
            solver,
            t0=0.0,
            t1=self.step_dt,
            dt0=self.numeric_dt,
            y0=self._state,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1000000,
        )

        self._state = solution.ys[-1]
        self._time += self.step_dt

        obs = self._get_obs()
        reward = self.reward_function(self._state)

        return (
            obs,
            reward,
            self._terminate(),
            False,  # truncated
            self._get_info(),
        )
