import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax.numpy as jnp
import diffrax
import jax
from typing import Callable, Optional

from pde_opt.environments.base_env import PDEEnv
from pde_opt.numerics.equations import AdvectionDiffusion2D


class AdvectionDiffusionEnv(PDEEnv):
    def __init__(
        self,
        reset_func: Callable,
        diffusion_coefficient: float,
        max_control_step: float,
        discrete_action_space: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reset_func = reset_func
        self._source_location = (0.0, 0.0)
        self.D = diffusion_coefficient
        self.discrete_action_space = discrete_action_space
        if discrete_action_space:
            self.action_space = spaces.Discrete(5)
            self._action_to_direction = {
                0: np.array([max_control_step, 0]),  # Move right (positive x)
                1: np.array([0, max_control_step]),  # Move up (positive y)
                2: np.array([-max_control_step, 0]),  # Move left (negative x)
                3: np.array([0, -max_control_step]),  # Move down (negative y)
                4: np.array([0, 0]),  # No movement
            }
        else:
            self.action_space = spaces.Box(
                low=-max_control_step, high=max_control_step, shape=(2,)
            )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # super().reset(seed=seed)
        if seed is not None:
            self._state = self.reset_func(self.domain, seed=seed)
        else:
            self._state = self.reset_func(self.domain)
        self._time = 0.0
        self._source_location = (0.0, 0.0)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        offset = (
            action
            if not self.discrete_action_space
            else self._action_to_direction[action]
        )
        old_source_location = self._source_location
        self._source_location = (
            np.clip(
                self._source_location[0] + offset[0],
                self.domain.box[0][0],
                self.domain.box[0][1],
            ),
            np.clip(
                self._source_location[1] + offset[1],
                self.domain.box[1][0],
                self.domain.box[1][1],
            ),
        )

        path_fn_ = lambda t: (
            old_source_location[0]
            + (self._source_location[0] - old_source_location[0]) * t / self.step_dt,
            old_source_location[1]
            + (self._source_location[1] - old_source_location[1]) * t / self.step_dt,
        )

        def advection(t, p, xs, ys):
            xi, yi = path_fn_(t)
            r2 = ((xs - xi) ** 2 + (ys - yi) ** 2) / (2.0 * p[1])
            grad_x = -(xs - xi) / p[1] * jnp.exp(-r2)
            grad_y = -(ys - yi) / p[1] * jnp.exp(-r2)
            return p[0] * grad_x, p[0] * grad_y

        eq = AdvectionDiffusion2D(
            self.domain,
            lambda t, x, y: advection(t, [0.1, 0.01], x, y),
            self.D,
            smooth=False,
        )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(jax.jit(lambda t, y, args: eq.rhs(y, t))),
            diffrax.Tsit5(),
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
        reward = self.reward_function(self._state) #TODO: should this be obs or state?
        return (
            obs,
            reward,
            self._terminate(),
            False,
            self._get_info(),
        )  # TODO: implement a terminate state and the truncated will be if the time limit is reached (right now, terminated is the time limit and truncated is always false)
