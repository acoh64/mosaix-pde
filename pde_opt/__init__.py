from gymnasium.envs.registration import register

register(
    id='AdvectionDiffusion-v0',
    entry_point='pde_opt.environments.advection_diffusion_env:AdvectionDiffusionEnv',
)