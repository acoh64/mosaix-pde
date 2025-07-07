import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import abstractmethod
from typing import Callable

from pde_opt.numerics.domains import Domain

class PDEEnv(gym.Env):
    """
    This is the base env for all PDE problems. All 1D custom environments should inherit this environment and implement the eccording methods

    :param T: The end time of the simulation.
    :param dt: The temporal timestep of the simulation.
    :param X: The spatial length of the simulation.
    :param dx: The spatial timestep of the simulation.
    :param reward_class: An instance of the reward class to specify user reward for each simulation step. Must inherit BaseReward class. See `reward documentation <../../utils/rewards.html>`_ for detials.
    :param normalize: Chooses whether to take action inputs between -1 and 1 and normalize them to betwen (``-max_control_value``, ``max_control_value``) or to leave inputs unaltered. ``max_control_value`` is environment specific so please see the environment for details. 
    """
    def __init__(self, end_time: float, step_dt: float, numeric_dt: float, domain: Domain, field_dim: int, reward_function: Callable):
        super().__init__()
        self.end_time = end_time
        self.step_dt = step_dt
        self.numeric_dt = numeric_dt
        self.domain = domain
        self.field_dim = field_dim
        self.observation_space = spaces.Box(low=0., high=255., shape=(1, *self.domain.points,), dtype=np.uint8)
        self._state = np.zeros((*self.domain.points, self.field_dim))
        self._time = 0.0
        self.reward_function = reward_function
    def _get_obs(self):
        return (np.clip(np.array(self._state), 0., 1.) * 255).astype(np.uint8)[None]
    
    def _get_info(self):
        return {}
    
    def _terminate(self):
        return self._time >= self.end_time
        
    @abstractmethod
    def step(self, action: np.ndarray):
        """
        step

        Implements the environment behavior for a single timestep depending on a given action

        :param action: The action to take in the environment. 
        """
        pass

    @abstractmethod
    def reset(self, init_cond: np.ndarray, recirculation_func):
        """
        reset 

        Resets the environment at the start of each epsiode

        :param init_cond: The intial condition to reset the PDE :math:`u(x, 0)` to. 

        :param recirculation_func: Specifies the plant parameter function. See each individual environment for details on implementation.
        """
        pass