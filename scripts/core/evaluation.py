from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gym
import numpy as np
from PyQt5 import QtCore

from scripts.core.config import load_config
from scripts.core.registry import ensure_env_registered
from scripts.utils.pine_td3 import PINE_TD3


@dataclass
class EvaluationResult:
    average_reward: float
    success_rate: float
    crash_rate: float
    average_success_steps: float
    output_dir: Path


class EvaluationRunner:
    def __init__(
        self,
        eval_path: str | Path,
        config_file: str | Path,
        model_file: str | Path,
        eval_episodes: int,
        eval_env: Optional[str] = None,
        eval_dynamics: Optional[str] = None,
        seeds: Optional[Iterable[int]] = None,
    ) -> None:
        ensure_env_registered()
        self.eval_path = Path(eval_path)
        self.config_file = Path(config_file)
        self.model_file = Path(model_file)
        self.eval_episodes = int(eval_episodes)
        self.seeds = list(seeds) if seeds is not None else None
        self.seed_idx = 0

        self.cfg = load_config(self.config_file)
        if eval_env is not None:
            self.cfg.set("options", "env_name", eval_env)
        if eval_env == "NH_center":
            self.cfg.set("environment", "accept_radius", "2")
        if eval_dynamics is not None:
            self.cfg.set("options", "dynamic_name", eval_dynamics)

        self.env = gym.make("airsim-env-v0")
        self.env.set_config(self.cfg)
        self.eval_env = self.cfg.get("options", "env_name")
        self.eval_dynamics = self.cfg.get("options", "dynamic_name")

    def _set_next_seed(self) -> None:
        if self.seeds is None or self.seed_idx >= len(self.seeds):
            return
        seed = int(self.seeds[self.seed_idx])
        np.random.seed(seed)
        if hasattr(self.env, "set_goal_seed"):
            self.env.set_goal_seed(seed)
        self.seed_idx += 1

    def _output_dir(self) -> Path:
        return self.eval_path / f"eval_{self.eval_episodes}_{self.eval_env}_{self.eval_dynamics}"

    def run(self) -> EvaluationResult:
        model = PINE_TD3.load(str(self.model_file), env=self.env)
        self.env.model = model
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectories = []
        actions = []
        states = []
        observations = []
        winds = []
        goals = []
        energies = []
        rewards = []
        successes = []
        crashes = []
        success_steps = []

        self._set_next_seed()
        obs = self.env.reset()
        episode_reward = 0.0
        traj, action_log, state_log, obs_log, wind_log = [], [], [], [], []

        while len(rewards) < self.eval_episodes:
            action, _ = model.predict(obs.astype(np.float64), deterministic=True)
            next_obs, reward, done, info = self.env.step(action)

            traj.append(self.env.dynamic_model.get_position())
            action_log.append(action)
            state_log.append(np.copy(self.env.dynamic_model.state_raw))
            obs_log.append(obs)
            wind_log.append(getattr(self.env.dynamic_model, "wind_vec", (0.0, 0.0, 0.0)))
            episode_reward += float(reward)
            obs = next_obs

            if not done:
                continue

            goals.append(np.copy(self.env.dynamic_model.goal_position))
            energies.append(info.get("Cumulated Power", 0.0))
            successes.append(float(info.get("is_success", False)))
            crashes.append(float(info.get("is_crash", False)))
            rewards.append(episode_reward)

            if info.get("is_success"):
                traj.append(1)
                action_log.append(1)
                success_steps.append(info.get("step_num", 0))
            elif info.get("is_crash"):
                traj.append(2)
                action_log.append(2)
            else:
                traj.append(3)
                action_log.append(3)
            wind_log.append(1)

            trajectories.append(traj)
            actions.append(action_log)
            states.append(state_log)
            observations.append(obs_log)
            winds.append(wind_log)

            np.save(output_dir / "traj_eval.npy", np.array(trajectories, dtype=object))
            np.save(output_dir / "action_eval.npy", np.array(actions, dtype=object))
            np.save(output_dir / "state_eval.npy", np.array(states, dtype=object))
            np.save(output_dir / "obs_eval.npy", np.array(observations, dtype=object))
            np.save(output_dir / "wind_eval.npy", np.array(winds, dtype=object))
            np.save(output_dir / "goal_positions_eval.npy", np.array(goals, dtype=object))
            np.save(output_dir / "energy_eval.npy", np.array(energies, dtype=object))
            np.save(output_dir / "rewards_eval.npy", np.array(rewards, dtype=float))
            np.save(output_dir / "successes_eval.npy", np.array(successes, dtype=float))
            np.save(output_dir / "crashes_eval.npy", np.array(crashes, dtype=float))
            np.save(output_dir / "step_nums_eval.npy", np.array(success_steps, dtype=object))

            print(f"episode {len(rewards)} reward={episode_reward:.3f} success={bool(info.get('is_success'))}")

            self._set_next_seed()
            obs = self.env.reset()
            episode_reward = 0.0
            traj, action_log, state_log, obs_log, wind_log = [], [], [], [], []

        result = EvaluationResult(
            average_reward=float(np.mean(rewards)) if rewards else 0.0,
            success_rate=float(np.mean(successes)) if successes else 0.0,
            crash_rate=float(np.mean(crashes)) if crashes else 0.0,
            average_success_steps=float(np.mean(success_steps)) if success_steps else 0.0,
            output_dir=output_dir,
        )
        np.save(
            output_dir / "results.npy",
            np.array(
                [
                    result.average_reward,
                    result.success_rate,
                    result.crash_rate,
                    result.average_success_steps,
                ]
            ),
        )
        return result


class EvaluateThread(QtCore.QThread):
    def __init__(
        self,
        eval_path,
        config,
        model_file,
        eval_ep_num,
        eval_env=None,
        eval_dynamics=None,
        seeds=None,
    ) -> None:
        super().__init__()
        self.runner = EvaluationRunner(eval_path, config, model_file, eval_ep_num, eval_env, eval_dynamics, seeds)
        self.env = self.runner.env

    def run(self):
        return self.runner.run()

