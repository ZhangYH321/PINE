from __future__ import annotations

from gym.envs.registration import register


ENV_ID = "airsim-env-v0"
ENTRY_POINT = "gym_env.gym_env.envs.airsim_env:AirsimGymEnv"


def ensure_env_registered() -> None:
    try:
        register(id=ENV_ID, entry_point=ENTRY_POINT)
    except Exception:
        pass

