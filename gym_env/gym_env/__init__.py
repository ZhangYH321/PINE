from gym.envs.registration import register

try:
    register(
        id="airsim-env-v0",
        entry_point="gym_env.gym_env.envs.airsim_env:AirsimGymEnv",
    )
except Exception:
    pass
