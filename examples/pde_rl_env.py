"""Create and sample a reinforcement-learning environment for a PDE.

This is the standalone script counterpart to ``docs/notebooks/pde_rl_env.ipynb``.
It computes a Gross--Pitaevskii ground state, constructs ``PDEEnv-v0``, samples
random actions for one episode, and displays an animation. The ground-state solve
and episode simulation may take several minutes on CPU.
"""

import diffrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import zoom

from mosaix_pde.numerics.domains import Domain
from mosaix_pde.numerics.equations.gross_pitaevskii import (
    GPE2DTSControl,
    a0,
    hbar,
    mass_Na23,
)
from mosaix_pde.numerics.solvers import StrangSplitting
from mosaix_pde.numerics.utils.initialization_utils import initialize_Psi
from mosaix_pde.rl_utils import density, detect_vortices


def main():
    atoms = 5e5
    omega = 2 * jnp.pi * 10
    omega_z = jnp.sqrt(8) * omega
    eccentricity = 0.0
    scattering_length = 100 * a0
    lx = ly = 150e-6

    length_scale = jnp.sqrt(hbar / (mass_Na23 * omega))
    time_scale = 1 / omega
    lx_dimensionless = lx / length_scale
    ly_dimensionless = ly / length_scale
    coupling = (
        4
        * jnp.pi
        * scattering_length
        * atoms
        * jnp.sqrt((mass_Na23 * omega_z) / (2 * jnp.pi * hbar))
    )

    grid_points = 256
    domain = Domain(
        (grid_points, grid_points),
        (
            (-lx_dimensionless / 2, lx_dimensionless / 2),
            (-ly_dimensionless / 2, ly_dimensionless / 2),
        ),
        "dimensionless",
    )

    psi0 = initialize_Psi(grid_points, width=100, vortexnumber=0) * length_scale
    psi0 /= jnp.sqrt(jnp.sum(density(psi0)) * domain.dx[0] ** 2)

    equation = GPE2DTSControl(
        domain, coupling, eccentricity, lambda a, b, c: 0.0, trap_factor=1.0
    )
    solver = StrangSplitting(
        equation.A_term,
        equation.domain.dx[0],
        equation.fft,
        equation.ifft,
        -1j,
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(jax.jit(lambda t, y, args: equation.B_terms(y, t))),
        solver,
        t0=0.0,
        t1=0.05 / time_scale,
        dt0=1e-5 / time_scale,
        y0=jnp.stack([psi0.real, psi0.imag], axis=-1),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=1000000,
    )
    ground_state = solution.ys[-1]

    end_time = 1.0 / time_scale
    step_dt = 0.01 / time_scale
    numeric_dt = 1e-5 / time_scale

    def reset_function(reset_domain):
        if ground_state.shape[:-1] != reset_domain.points:
            zoom_factors = (
                reset_domain.points[0] / ground_state.shape[0],
                reset_domain.points[1] / ground_state.shape[1],
                1.0,
            )
            return zoom(np.array(ground_state), zoom_factors, order=1)
        return ground_state

    def state_to_observation(state):
        state_density = density(state[..., 0] + 1j * state[..., 1])
        return (np.clip(np.array(state_density), 0, 0.01) * 100 * 255).astype(np.uint8)[
            None
        ]

    def reward_function(state):
        psi = state[..., 0] + 1j * state[..., 1]
        vortices = detect_vortices(psi, amp_thresh=0.00005, tol=0.5)
        return vortices["num_vortices"]

    action_space_config = {
        "type": "continuous",
        "shape": (2,),
        "low": float(-1e-5 / length_scale),
        "high": float(1e-5 / length_scale),
    }

    def update_control_value(action, old_control_value):
        return (
            old_control_value[0] + action[0],
            old_control_value[1] + action[1],
        )

    def update_control_parameter(old_control_value, new_control_value):
        def path(t):
            return (
                old_control_value[0]
                + (new_control_value[0] - old_control_value[0]) * t / step_dt,
                old_control_value[1]
                + (new_control_value[1] - old_control_value[1]) * t / step_dt,
            )

        def light(t, xs, ys):
            amplitude = 30.0
            sigma = 2e-6 / length_scale
            xi, yi = path(t)
            return amplitude * jnp.exp(
                -((xs - xi) ** 2 + (ys - yi) ** 2) / (2.0 * sigma**2)
            )

        return light

    environment_parameters = {
        "equation_type": GPE2DTSControl,
        "domain": domain,
        "solver_type": StrangSplitting,
        "end_time": end_time,
        "step_dt": step_dt,
        "numeric_dt": numeric_dt,
        "state_to_observation_func": state_to_observation,
        "reward_function": reward_function,
        "reset_func": reset_function,
        "reset_control_value": (0.0, 0.0),
        "update_control_value": update_control_value,
        "update_control_parameter": update_control_parameter,
        "action_space_config": action_space_config,
        "static_equation_parameters": {
            "k": coupling,
            "e": eccentricity,
            "trap_factor": 1.0,
        },
        "control_equation_parameter_name": "lights",
        "solver_parameters": {"time_scale": 1.0 - 1j * 0.01},
    }
    environment = gym.make("PDEEnv-v0", **environment_parameters)
    print(f"Action space: {environment.action_space}")
    print(f"Sample action: {environment.action_space.sample()}")
    print(f"Observation space: {environment.observation_space}")

    observation, _ = environment.reset()
    plt.imshow(observation[0], vmin=0, vmax=255)
    plt.show()

    action = environment.action_space.sample()
    observation, reward, _, _, _ = environment.step(action)
    print(f"Reward after one step: {reward}")
    plt.imshow(observation[0], vmin=0, vmax=255)
    plt.show()

    observation, _ = environment.reset()
    episode_over = False
    rewards = []
    observations = []
    total_reward = 0
    while not episode_over:
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, _ = environment.step(action)
        total_reward += reward
        episode_over = terminated or truncated
        rewards.append(reward)
        observations.append(observation[0])

    print(f"Episode finished. Total reward: {total_reward}")
    environment.close()

    fig, axis = plt.subplots(figsize=(4, 4))
    frames = []
    for i in range(0, len(observations), 2):
        image = axis.imshow(
            observations[i],
            animated=True,
            vmin=0.0,
            vmax=255,
            extent=[
                domain.box[0][0],
                domain.box[0][1],
                domain.box[1][0],
                domain.box[1][1],
            ],
        )
        title = axis.text(
            0.5,
            1.05,
            f"Step {i}, Reward: {rewards[i]:.2f}",
            ha="center",
            transform=axis.transAxes,
        )
        frames.append([image, title])

    # Keep blitting disabled for compatibility with native GUI backends on macOS.
    _animation = animation.ArtistAnimation(fig, frames, interval=100, blit=False)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
