import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import abstractmethod
from typing import Type

class PDEEnv1D(gym.Env):
    """
    This is the base env for all 1D PDE problems. All 1D custom environments should inherit this environment and implement the eccording methods

    :param T: The end time of the simulation.
    :param dt: The temporal timestep of the simulation.
    :param X: The spatial length of the simulation.
    :param dx: The spatial timestep of the simulation.
    :param reward_class: An instance of the reward class to specify user reward for each simulation step. Must inherit BaseReward class. See `reward documentation <../../utils/rewards.html>`_ for detials.
    :param normalize: Chooses whether to take action inputs between -1 and 1 and normalize them to betwen (``-max_control_value``, ``max_control_value``) or to leave inputs unaltered. ``max_control_value`` is environment specific so please see the environment for details. 
    """
    def __init__(self):
        super().__init__()

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