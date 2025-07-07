import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, CheckpointCallback
# from sbx import DQN
import pde_opt
from pde_opt.numerics.domains import Domain

ref = np.load("notebooks/reference.npy")

def reset_func(domain, seed=0):
    return 0.5 * jnp.ones(domain.points) + 0.01 * random.normal(random.PRNGKey(seed), domain.points)

Nx, Ny = 64, 64
Lx = 0.02 * Nx
Ly = 0.02 * Ny
domain = Domain((Nx, Ny), ((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2)), "dimensionless")

params = {
    "reset_func": reset_func,
    "diffusion_coefficient": 0.1,
    "max_control_step": 0.1,
    "end_time": 1.0,
    "step_dt": 0.05,
    "numeric_dt": 0.0001,
    "domain": domain,
    "field_dim": 1,
    "reward_function": lambda x: np.linalg.norm(x[45,45]),
    "discrete_action_space": False
}

env = gym.make('AdvectionDiffusion-v0', **params)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/", n_steps=20, batch_size=40, n_epochs=5)

checkpoint_callbackPPO = CheckpointCallback(
    save_freq=100,
    save_path="./logsPPO",
    name_prefix="rl_model",
    save_replay_buffer=False,
    save_vecnormalize=False,
 )

model.learn(total_timesteps=10000, callback=checkpoint_callbackPPO)

model.save("model_10000")