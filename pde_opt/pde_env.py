import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import jax.numpy as jnp
import diffrax
import jax
from typing import Callable, Optional, Type, Dict, Any, Tuple

from .numerics.equations import BaseEquation
from .numerics import domains
from .utils import check_equation_solver_compatibility, prepare_solver_params

# TODO: create RL environments that can control multiple parameters at once.

# Register the environment with gymnasium
register(
    id="PDEEnv-v0",
    entry_point="pde_opt.pde_env:PDEEnv",
)


class PDEEnv(gym.Env):
    """Reinforcement learning environment for controlling partial differential equations.

    The PDEEnv class provides a Gymnasium-compatible environment for reinforcement
    learning control of PDEs. It allows agents to learn control policies by taking
    actions that modify equation parameters and receiving rewards based on the resulting
    system behavior.

    Attributes:
        equation_type (Type[BaseEquation]): The PDE equation class to control.
        domain (domains.Domain): Spatial domain for the PDE simulation.
        solver_type (Type[diffrax.AbstractSolver]): Numerical solver for time integration.
        end_time (float): Maximum simulation time for each episode.
        step_dt (float): Time duration for each RL step.
        numeric_dt (float): Internal time step for numerical integration.
        observation_space (gym.Space): Gymnasium observation space.
        action_space (gym.Space): Gymnasium action space.

    For an example, see the documentation notebooks.
    """

    def __init__(
        self,
        equation_type: Type[BaseEquation],
        domain: domains.Domain,
        solver_type: Type[diffrax.AbstractSolver],
        end_time: float,
        step_dt: float,
        numeric_dt: float,
        state_to_observation_func: Callable,
        reward_function: Callable,
        reset_func: Callable,
        reset_control_value,
        update_control_value: Callable,
        update_control_parameter: Callable,
        action_space_config: Dict[str, Any],
        static_equation_parameters: Dict[str, Any],
        control_equation_parameter_name: str,
        solver_parameters: Dict[str, Any],
    ):
        """Initialize the PDE reinforcement learning environment.

        Args:
            equation_type (Type[BaseEquation]): The PDE equation class to control.
                Must be a subclass of BaseEquation from numerics.equations.
            domain (domains.Domain): Spatial domain for the PDE simulation. Defines
                the grid resolution, spatial bounds, and coordinate system.
            solver_type (Type[diffrax.AbstractSolver]): Numerical solver for time
                integration. Must be compatible with the equation type.
            end_time (float): Maximum simulation time for each episode. Episodes
                terminate when this time is reached.
            step_dt (float): Time duration for each RL step. This is the time
                interval between consecutive actions taken by the agent.
            numeric_dt (float): Internal time step for numerical integration.
                Likely to be smaller than step_dt for numerical stability.
            state_to_observation_func (Callable): Function that converts the PDE
                state to an observation for the RL agent.
            reward_function (Callable): Function that computes the reward from
                the current PDE state.
            reset_func (Callable): Function that generates initial conditions for
                the PDE. Should accept (domain, seed=None) and return initial state.
            reset_control_value: Initial value for the control parameter.
            update_control_value (Callable): Function that updates the control
                value based on the agent's action. Should accept (action, old_value)
                and return the new control value.
            update_control_parameter (Callable): Function that converts control
                values to equation parameters. Should accept (old_value, new_value)
                and return the parameter value for the equation.
            action_space_config (Dict[str, Any]): Configuration for the action space.
                Should contain:
                - "type": "continuous" or "discrete"
                - For continuous: "shape", "low", "high"
                - For discrete: "num_actions", "action_mapping"
            static_equation_parameters (Dict[str, Any]): Fixed parameters for the
                equation that are not controlled by the agent. Keys should match
                equation parameter names.
            control_equation_parameter_name (str): Name of the equation parameter
                that will be controlled by the agent's actions.
            solver_parameters (Dict[str, Any]): Additional parameters for the
                numerical solver. Passed directly to the solver constructor.
        """
        super().__init__()

        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type
        check_equation_solver_compatibility(self.solver_type, self.equation_type)

        self.end_time = end_time
        self.step_dt = step_dt
        self.numeric_dt = numeric_dt
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
        """Set up the action space based on configuration.

        This method configures the action space for the RL environment based on the
        provided configuration dictionary. It supports both continuous and discrete
        action spaces.

        Args:
            config (Dict[str, Any]): Action space configuration dictionary. Should contain:
                - "type": "continuous" or "discrete" (default: "continuous")
                - For continuous spaces:
                  - "shape": Tuple defining action dimensions (default: (2,))
                  - "low": Lower bound for actions (default: -1.0)
                  - "high": Upper bound for actions (default: 1.0)
                - For discrete spaces:
                  - "num_actions": Number of discrete actions (default: 5)
                  - "action_mapping": Dict mapping action indices to control values
        """
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
        """Initialize the equation and solver with current parameters.

        This method creates new instances of the equation and solver with the current
        parameter values. It's used internally to set up the PDE simulation components.
        """
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
        """Convert state to observation.

        This method converts the current PDE state to an observation suitable for
        the RL agent using the provided state_to_observation_func.
        """
        return self.state_to_observation_func(self._state)

    def _get_info(self):
        """Get additional information about the environment state.

        Returns:
            dict: Empty dictionary (placeholder for future extensions).
        """
        return {}

    def _terminate(self):
        """Check if the episode should terminate.

        This method determines whether the current episode should end based on the
        simulation time reaching the maximum end_time.

        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        return self._time >= self.end_time

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state.

        This method resets the PDE environment to its initial state, generating new
        initial conditions and resetting the control parameters. It follows the
        standard Gymnasium reset interface.

        Args:
            seed (Optional[int]): Random seed for reproducible initial conditions.
            options (Optional[dict]): Additional options for reset.

        Returns:
            Tuple[np.ndarray, dict]: A tuple containing:
                - observation: Initial observation from the reset state
                - info: Additional information dictionary (currently empty)
        """
        if seed is not None:
            self._state = self.reset_func(self.domain, seed=seed)
        else:
            self._state = self.reset_func(self.domain)

        self._time = 0.0

        self._control_value = self.reset_control_value

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        This method executes one step of the RL environment by:
        1. Converting the agent's action to a control parameter update
        2. Updating the PDE equation with the new control parameter
        3. Simulating the PDE forward in time for one step_dt
        4. Computing the reward and next observation
        5. Checking for episode termination

        Args:
            action (np.ndarray): The action taken by the RL agent. Should be compatible
                with the environment's action space. For continuous actions, this is
                typically a numpy array. For discrete actions, this is an integer.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: A tuple containing:
                - observation: Next observation from the updated state
                - reward: Reward value computed from the current state
                - terminated: Whether the episode has ended (reached end_time)
                - truncated: Whether the episode was truncated (always False for now)
                - info: Additional information dictionary (currently empty)
        """

        offset = (
            action
            if not self._action_to_direction
            else self._action_to_direction[action]
        )

        old_control_value = self._control_value
        self._control_value = self.update_control_value(offset, old_control_value)

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
            # stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6), TODO: add option for adaptive step size controller
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
