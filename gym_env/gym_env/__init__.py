from gym.envs.registration import register, registry

if "airsim-env-v0" not in registry:
    register(
        id="airsim-env-v0",
        entry_point="gym_env.gym_env.envs.airsim_env:AirsimGymEnv",
    )
