import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import diffrax
import jax
from typing import Callable, Optional, Type, Dict, Any

from pde_opt.numerics.equations import BaseEquation
from pde_opt.numerics import domains


class FlexiblePDEEnv(gym.Env):
    """
    A flexible PDE environment that automatically configures itself based on the equation type.
    Just pass the equation type, domain, and solver, and it creates the environment for you.
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
        reward_function: Callable,
        reset_func: Callable,
        action_space_config: Dict[str, Any],
        equation_parameters: Dict[str, Any],
        solver_parameters: Dict[str, Any],
    ):
        super().__init__()
        
        # Store configuration
        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type
        self.end_time = end_time
        self.step_dt = step_dt
        self.numeric_dt = numeric_dt
        self.field_dim = field_dim
        self.reward_function = reward_function
        self.reset_func = reset_func
        self.equation_parameters = equation_parameters
        self.solver_parameters = solver_parameters
        
        # Initialize state
        self._state = np.zeros((*self.domain.points, self.field_dim))
        self._time = 0.0
        
        # Set up observation space
        self.observation_space = spaces.Box(
            low=0., high=255., 
            shape=(1, *self.domain.points,), 
            dtype=np.uint8
        )
        
        # Set up action space based on configuration
        self._setup_action_space(action_space_config)
        
        # Initialize equation and solver
        self._equation = None
        self._solver = None
        self._initialize_equation_and_solver()
    
    def _setup_action_space(self, config: Dict[str, Any]):
        """Set up the action space based on configuration."""
        action_type = config.get('type', 'continuous')
        
        if action_type == 'discrete':
            num_actions = config.get('num_actions', 5)
            self.action_space = spaces.Discrete(num_actions)
            self._action_to_direction = config.get('action_mapping', {})
        else:
            # Continuous action space
            action_shape = config.get('shape', (2,))
            low = config.get('low', -1.0)
            high = config.get('high', 1.0)
            self.action_space = spaces.Box(
                low=low, high=high, shape=action_shape
            )
            self._action_to_direction = None
    
    def _initialize_equation_and_solver(self):
        """Initialize the equation and solver with current parameters."""
        # Initialize equation with domain and parameters
        self._equation = self.equation_type(
            domain=self.domain, 
            **self.equation_parameters
        )
        
        # Initialize solver
        self._solver = self.solver_type(**self.solver_parameters)
    
    def _get_obs(self):
        """Convert state to observation."""
        return (np.clip(np.array(self._state), 0., 1.) * 255).astype(np.uint8)[None]
    
    def _get_info(self):
        """Get additional information about the environment."""
        return {
            'time': self._time,
            'equation_type': self.equation_type.__name__,
            'solver_type': self.solver_type.__name__,
        }
    
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
        
        # Reinitialize equation and solver in case parameters changed
        self._initialize_equation_and_solver()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action if using discrete action space
        if self._action_to_direction is not None:
            action = self._action_to_direction[action]
        
        # Update equation parameters based on action (subclasses can override)
        self._update_equation_from_action(action)
        
        # Solve the PDE for one step
        self._solve_step()
        
        # Update time
        self._time += self.step_dt
        
        # Get observation and reward
        obs = self._get_obs()
        reward = self.reward_function(self._state)
        
        return (
            obs,
            reward,
            self._terminate(),
            False,  # truncated
            self._get_info(),
        )
    
    def _update_equation_from_action(self, action: np.ndarray):
        """
        Update equation parameters based on the action.
        This method can be overridden by subclasses for specific control schemes.
        """
        # Default implementation: no action effect on equation
        # Subclasses should override this for specific control mechanisms
        pass
    
    def _solve_step(self):
        """Solve the PDE for one step using the current equation and solver."""
        # Solve with diffrax
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(jax.jit(lambda t, y, args: self._equation.rhs(y, t))),
            self._solver,
            t0=0.0,
            t1=self.step_dt,
            dt0=self.numeric_dt,
            y0=self._state,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1000000,
        )
        
        self._state = solution.ys[-1]
    
    def update_equation_parameters(self, new_parameters: Dict[str, Any]):
        """Update equation parameters and reinitialize the equation."""
        self.equation_parameters.update(new_parameters)
        self._initialize_equation_and_solver()
    
    def update_solver_parameters(self, new_parameters: Dict[str, Any]):
        """Update solver parameters and reinitialize the solver."""
        self.solver_parameters.update(new_parameters)
        self._initialize_equation_and_solver() 