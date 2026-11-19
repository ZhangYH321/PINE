from __future__ import annotations

import csv
import copy
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch as th
try:
    import wandb
except ImportError:
    wandb = None
from PyQt5 import QtCore
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.noise import NormalActionNoise

from scripts.core.config import PROJECT_ROOT, load_config, parse_net_arch
from scripts.core.registry import ensure_env_registered
from scripts.utils.pine_td3 import PINE_TD3
from scripts.utils.pine_sac import PineSACPolicy


class LinearActionNoiseCallback(BaseCallback):
    """Linearly update Gaussian exploration noise over environment steps."""

    def __init__(self, initial_sigma: float, final_sigma: float, total_timesteps: int) -> None:
        super().__init__()
        self.initial_sigma = float(initial_sigma)
        self.final_sigma = float(final_sigma)
        self.total_timesteps = int(total_timesteps)
        if self.initial_sigma < 0.0 or self.final_sigma < 0.0:
            raise ValueError("Action-noise sigmas must be non-negative.")
        if self.final_sigma > self.initial_sigma:
            raise ValueError("action_noise_sigma_final must not exceed action_noise_sigma_initial.")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive for action-noise scheduling.")

    def sigma_at_step(self, step: int) -> float:
        fraction = float(np.clip(float(step) / self.total_timesteps, 0.0, 1.0))
        return self.initial_sigma + fraction * (self.final_sigma - self.initial_sigma)

    def _apply_sigma(self) -> None:
        action_noise = getattr(self.model, "action_noise", None)
        if not isinstance(action_noise, NormalActionNoise):
            return
        sigma = self.sigma_at_step(self.num_timesteps)
        action_noise._sigma = np.full_like(action_noise._sigma, sigma, dtype=action_noise._dtype)
        self.logger.record("train/action_noise_sigma", sigma)

    def _on_training_start(self) -> None:
        self._apply_sigma()

    def _on_step(self) -> bool:
        self._apply_sigma()
        return True


class CrashRateCallback(BaseCallback):
    def __init__(self, log_freq: int, env) -> None:
        super().__init__()
        self.log_freq = int(log_freq)
        self.env = env
        self.crash_count = 0
        self.episode_count = 0
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1
        # VecEnv has already reset finished environments before callbacks run.
        # Read the terminal transition rather than the reset env's flags.
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict) or "episode" not in info:
                continue
            self.episode_count += 1
            if info.get("is_crash", False):
                self.crash_count += 1
        if self.step_counter % self.log_freq == 0 and self.episode_count > 0:
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "crash_rate": self.crash_count / self.episode_count,
                        "total_episodes": getattr(self.env, "episode_total_count", 0),
                        "crash_episodes": getattr(self.env, "episode_crash_count", 0),
                    },
                    step=self.num_timesteps,
                )
            self.crash_count = 0
            self.episode_count = 0
        return True


class RolloutMetricsCallback(BaseCallback):
    """Persist SB3-style rollout metrics with their corresponding global step."""

    FIELDNAMES = (
        "global_step",
        "rollout/success_rate",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "rollout/ep_energy_mean_j",
        "rollout/success_ep_energy_mean_j",
        "rollout/net_wind_assist_energy_mean_j",
        "completed_episodes",
    )

    def __init__(self, output_path: str | Path, stats_window_size: int = 100) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.stats_window_size = int(stats_window_size)
        self.success_buffer = deque(maxlen=self.stats_window_size)
        self.reward_buffer = deque(maxlen=self.stats_window_size)
        self.length_buffer = deque(maxlen=self.stats_window_size)
        self.episode_window = deque(maxlen=self.stats_window_size)
        self.total_episodes = 0
        self._handle = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        has_header = self.output_path.exists() and self.output_path.stat().st_size > 0
        self._handle = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        if not has_header:
            self._writer.writeheader()
        self._handle.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict) or "episode" not in info:
                continue
            episode = info["episode"]
            self.total_episodes += 1
            self.reward_buffer.append(float(episode.get("r", 0.0)))
            self.length_buffer.append(float(episode.get("l", 0.0)))
            success = bool(info.get("is_success", False))
            energy_j = float(
                info.get("episode_energy_j", info.get("Energy Consumed (J)", 0.0))
            )
            wind_assist_j = float(info.get("net_wind_assist_energy_j", 0.0))
            self.success_buffer.append(float(success))
            self.episode_window.append(
                {
                    "success": success,
                    "energy_j": energy_j,
                    "wind_assist_j": wind_assist_j,
                }
            )
            success_energies = [
                item["energy_j"] for item in self.episode_window if item["success"]
            ]
            row = {
                "global_step": int(self.num_timesteps),
                "rollout/success_rate": float(np.mean(self.success_buffer)),
                "rollout/ep_rew_mean": float(np.mean(self.reward_buffer)),
                "rollout/ep_len_mean": float(np.mean(self.length_buffer)),
                "rollout/ep_energy_mean_j": float(
                    np.mean([item["energy_j"] for item in self.episode_window])
                ),
                "rollout/success_ep_energy_mean_j": (
                    float(np.mean(success_energies)) if success_energies else ""
                ),
                "rollout/net_wind_assist_energy_mean_j": float(
                    np.mean([item["wind_assist_j"] for item in self.episode_window])
                ),
                "completed_episodes": self.total_episodes,
            }
            self._writer.writerow(row)
            self._handle.flush()
            self.logger.record("rollout/success_rate", row["rollout/success_rate"])
            self.logger.record(
                "rollout/ep_energy_mean_j", row["rollout/ep_energy_mean_j"]
            )
            self.logger.record(
                "rollout/net_wind_assist_energy_mean_j",
                row["rollout/net_wind_assist_energy_mean_j"],
            )
            if success_energies:
                self.logger.record(
                    "rollout/success_ep_energy_mean_j",
                    row["rollout/success_ep_energy_mean_j"],
                )
        return True

    def _on_training_end(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None




class DeterministicEvalBestModelCallback(BaseCallback):
    """Select ``best_model`` using an independent deterministic evaluation.

    Training rollouts contain exploration noise and are therefore unsuitable
    as the sole checkpoint-selection signal.  This callback evaluates the
    current policy in a separate environment every ``eval_freq_steps`` using
    fixed, independently seeded episodes and ``deterministic=True`` actions.
    Selection is lexicographic: success rate (higher), successful-episode
    energy (lower), then successful-episode steps (lower).
    """

    def __init__(
        self,
        save_path: str | Path,
        config,
        eval_freq_steps: int = 40_000,
        eval_episodes: int = 50,
        eval_seed: int = 1_000_003,
        eval_env=None,
    ) -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self.config = config
        self.eval_freq_steps = int(eval_freq_steps)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        if self.eval_freq_steps <= 0:
            raise ValueError("eval_freq_steps must be positive.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")
        self.eval_env = None
        self.last_eval_step = 0
        self.best_key = None
        self.best_metrics = None
        self._owns_eval_env = eval_env is None
        self._provided_eval_env = eval_env
        self._saved_env_seed = None
        self._saved_episode_num = None
        self._saved_data_path = None

    def _on_training_start(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if self._provided_eval_env is not None:
            self.eval_env = self._provided_eval_env
            return

        eval_cfg = copy.deepcopy(self.config)
        if not eval_cfg.has_section("options"):
            eval_cfg.add_section("options")
        eval_cfg.set("options", "seed", str(self.eval_seed))
        # Evaluation must not create trajectory artifacts in the training run.
        if eval_cfg.has_section("wind"):
            eval_cfg.set("wind", "log_trajectory", "False")
        ensure_env_registered()
        self.eval_env = gym.make("airsim-env-v0", disable_env_checker=True).unwrapped
        self.eval_env.set_config(eval_cfg)
        self.eval_env.data_path = None

    @staticmethod
    def _reset_env(env):
        result = env.reset()
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _step_env(env, action):
        result = env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return observation, float(reward), bool(terminated or truncated), info
        observation, reward, done, info = result
        return observation, float(reward), bool(done), info

    def _evaluate(self):
        if self.eval_env is None:
            raise RuntimeError("Deterministic evaluation environment is not initialized.")
        env = self.eval_env
        self._saved_env_seed = getattr(env, "base_seed", None)
        self._saved_episode_num = getattr(env, "episode_num", None)
        self._saved_data_path = getattr(env, "data_path", None)
        training_counters = {
            name: getattr(env, name)
            for name in ("total_step", "episode_total_count", "episode_crash_count")
            if hasattr(env, name)
        }
        env.data_path = None
        if hasattr(env, "seed"):
            env.seed(self.eval_seed)
        if self._saved_episode_num is not None:
            env.episode_num = 0

        try:
            successes = []
            energies = []
            steps = []
            rewards = []
            for _ in range(self.eval_episodes):
                observation = self._reset_env(self.eval_env)
                episode_reward = 0.0
                while True:
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, reward, done, info = self._step_env(
                        self.eval_env, action
                    )
                    episode_reward += reward
                    if done:
                        break
                success = bool(info.get("is_success", False))
                successes.append(float(success))
                rewards.append(episode_reward)
                if success:
                    energies.append(
                        float(
                            info.get(
                                "episode_energy_j",
                                info.get("Energy Consumed (J)", 0.0),
                            )
                        )
                    )
                    steps.append(float(info.get("step_num", 0.0)))

            success_rate = float(np.mean(successes))
            success_energy = float(np.mean(energies)) if energies else np.inf
            success_steps = float(np.mean(steps)) if steps else np.inf
            return {
                "success_rate": success_rate,
                "success_energy_j": success_energy,
                "success_steps": success_steps,
                "mean_reward": float(np.mean(rewards)),
                "success_count": int(np.sum(successes)),
            }
        finally:
            if self._saved_env_seed is not None and hasattr(env, "seed"):
                env.seed(self._saved_env_seed)
            if self._saved_episode_num is not None:
                env.episode_num = self._saved_episode_num
            env.data_path = self._saved_data_path
            for name, value in training_counters.items():
                setattr(env, name, value)

    def _run_eval_if_due(self, force: bool = False) -> None:
        current_step = int(self.num_timesteps)
        if not force and current_step - self.last_eval_step < self.eval_freq_steps:
            return
        self.last_eval_step = current_step
        metrics = self._evaluate()
        key = (
            metrics["success_rate"],
            -metrics["success_energy_j"],
            -metrics["success_steps"],
        )
        improved = self.best_key is None or key > self.best_key
        if improved:
            self.best_key = key
            self.best_metrics = metrics
            self.model.save(str(self.save_path))
            print(
                "Saved deterministic evaluation best model: "
                f"step={current_step}, success={metrics['success_rate']:.3f}, "
                f"energy={metrics['success_energy_j']:.2f} J, "
                f"steps={metrics['success_steps']:.2f}"
            )
        self.logger.record("eval/success_rate", metrics["success_rate"])
        self.logger.record("eval/success_count", metrics["success_count"])
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/success_energy_j", metrics["success_energy_j"])
        self.logger.record("eval/success_steps", metrics["success_steps"])
        self.logger.record("eval/best_updated", float(improved))

    def _on_step(self) -> bool:
        self._run_eval_if_due()
        return True

    def _on_training_end(self) -> None:
        if self.eval_env is not None and self._owns_eval_env:
            close = getattr(self.eval_env, "close", None)
            if callable(close):
                close()
        self.eval_env = None


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str) -> None:
        super().__init__()
        self.save_freq = int(save_freq)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_name = f"model_step_{self.num_timesteps}"
            self.model.save(os.path.join(self.save_path, model_name))
            print(f"Saved model checkpoint: {os.path.join(self.save_path, model_name)}")
        return True


@dataclass
class TrainingArtifacts:
    root: Path
    log_dir: Path
    model_dir: Path
    config_dir: Path
    data_dir: Path
    rollout_dir: Path


def build_training_artifacts(cfg) -> TrainingArtifacts:
    env_name = cfg.get("options", "env_name")
    now_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_tag = cfg.get("options", "run_tag", fallback="")
    variant = cfg.get("options", "model_variant", fallback="pine")
    suffix = f"_{run_tag}" if run_tag else ""
    root = PROJECT_ROOT / "logs" / env_name / f"{now_string}_{cfg.get('options', 'dynamic_name')}_{cfg.get('options', 'policy_name')}_{cfg.get('options', 'algo')}_{variant}{suffix}"
    return TrainingArtifacts(
        root=root,
        log_dir=root / "tb_logs",
        model_dir=root / "models",
        config_dir=root / "config",
        data_dir=root / "data",
        rollout_dir=root / "rollout_metrics",
    )


class TrainingRunner:
    """Owns environment creation, model construction, and training."""

    def __init__(self, config_file: str | Path, use_wandb=None):
        ensure_env_registered()
        self.config_file = Path(config_file)
        self.cfg = load_config(self.config_file)
        if use_wandb is not None:
            self.cfg.set("options", "use_wandb", str(bool(use_wandb)))
        # Configuration establishes the spaces after construction. Keep the
        # legacy Gym interface for Shimmy and set attributes on the real env.
        self.env = gym.make("airsim-env-v0", disable_env_checker=True).unwrapped
        self.env.set_config(self.cfg)
        self.artifacts = build_training_artifacts(self.cfg)

        for path in (
            self.artifacts.log_dir,
            self.artifacts.model_dir,
            self.artifacts.config_dir,
            self.artifacts.data_dir,
            self.artifacts.rollout_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        with open(self.artifacts.config_dir / "config.ini", "w", encoding="utf-8") as handle:
            self.cfg.write(handle)

        self.project_name = self.cfg.get("wandb", "project", fallback=self.cfg.get("options", "env_name"))

        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        if not self.cfg.getboolean("options", "use_wandb", fallback=False):
            return
        if wandb is None:
            raise ImportError("Tracking requested: install wandb or use --no-wandb.")
        wandb.init(
            project=self.project_name,
            notes=self.cfg.get("wandb", "notes", fallback=""),
            name=self.cfg.get("wandb", "name", fallback=None),
            sync_tensorboard=True,
            save_code=False,
        )

    def _action_noise_sigmas(self) -> tuple[float, float]:
        legacy_sigma = self.cfg.getfloat("DRL", "action_noise_sigma", fallback=0.1)
        initial_sigma = self.cfg.getfloat(
            "DRL", "action_noise_sigma_initial", fallback=legacy_sigma
        )
        final_sigma = self.cfg.getfloat(
            "DRL", "action_noise_sigma_final", fallback=initial_sigma
        )
        if initial_sigma < 0.0 or final_sigma < 0.0:
            raise ValueError("Action-noise sigmas must be non-negative.")
        if final_sigma > initial_sigma:
            raise ValueError("action_noise_sigma_final must not exceed action_noise_sigma_initial.")
        return initial_sigma, final_sigma

    def build_model(self):
        activation_function = th.nn.Tanh if self.cfg.get("options", "activation_function") == "tanh" else th.nn.ReLU
        net_arch = parse_net_arch(self.cfg)
        algo = self.cfg.get("options", "algo").upper()
        seed = self.cfg.getint("options", "seed", fallback=0)
        n_actions = self.env.action_space.shape[-1]
        initial_noise_sigma, _ = self._action_noise_sigmas()
        noise_sigma = initial_noise_sigma * np.ones(n_actions)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)

        policy_name = self.cfg.get("options", "policy_name").lower()
        if policy_name != "mlp":
            raise ValueError(
                "This refactor keeps the supported TD3 path focused on vector observations. "
                f"Got policy_name={policy_name!r}, expected 'mlp'."
            )
        policy_base = "MlpPolicy"
        policy_kwargs = {"activation_fn": activation_function, "net_arch": net_arch}

        if algo == "TD3":
            model_variant = self.cfg.get("options", "model_variant", fallback="pine").lower()
            if model_variant in {"td3", "baseline", "standard"}:
                return TD3(
                    policy_base,
                    self.env,
                    action_noise=action_noise,
                    learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                    gamma=self.cfg.getfloat("DRL", "gamma"),
                    policy_kwargs=policy_kwargs,
                    learning_starts=self.cfg.getint("DRL", "learning_starts"),
                    batch_size=self.cfg.getint("DRL", "batch_size"),
                    train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                    gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                    buffer_size=self.cfg.getint("DRL", "buffer_size"),
                    tensorboard_log=str(self.artifacts.log_dir),
                    seed=seed,
                    verbose=2,
                )
            physical_feature_dim = self.cfg.getint("PGVL", "physical_feature_dim", fallback=getattr(self.env.dynamic_model, "expert_feature_length", 5))
            observed_physical_dim = int(
                getattr(self.env.dynamic_model, "expert_feature_length", 0)
            )
            if physical_feature_dim != observed_physical_dim:
                raise ValueError(
                    "PGVL physical_feature_dim must match the environment: "
                    f"configured={physical_feature_dim}, observed={observed_physical_dim}."
                )
            policy_kwargs.update(
                {
                    "physical_feature_dim": physical_feature_dim,
                    "spatial_feature_dim": self.cfg.getint(
                        "PGVL", "spatial_feature_dim", fallback=31
                    ),
                    "use_adaptive_physics_gate": self.cfg.getboolean(
                        "PGVL", "use_adaptive_physics_gate", fallback=True
                    ),
                    "use_wind_residual_actor": self.cfg.getboolean(
                        "PGVL", "use_wind_residual_actor", fallback=True
                    ),
                    "residual_hidden_dim": self.cfg.getint(
                        "PGVL", "residual_hidden_dim", fallback=16
                    ),
                    "residual_max_action": self.cfg.getfloat(
                        "PGVL", "residual_max_action", fallback=0.20
                    ),
                    "residual_yaw_max_action": self.cfg.getfloat(
                        "PGVL", "residual_yaw_max_action", fallback=0.20
                    ),
                    "obstacle_feature_dim": self.cfg.getint(
                        "PGVL", "obstacle_feature_dim", fallback=25
                    ),
                    "obstacle_proximity_threshold": self.cfg.getfloat(
                        "PGVL", "obstacle_proximity_threshold", fallback=0.60
                    ),
                    "obstacle_transition": self.cfg.getfloat(
                        "PGVL", "obstacle_transition", fallback=0.15
                    ),
                    "minimum_wind_strength": self.cfg.getfloat(
                        "PGVL", "minimum_wind_strength", fallback=0.08
                    ),
                    "wind_strength_transition": self.cfg.getfloat(
                        "PGVL", "wind_strength_transition", fallback=0.10
                    ),
                    "minimum_wind_confidence": self.cfg.getfloat(
                        "PGVL", "minimum_wind_confidence", fallback=0.25
                    ),
                    "confidence_transition": self.cfg.getfloat(
                        "PGVL", "confidence_transition", fallback=0.25
                    ),
                }
            )
            return PINE_TD3(
                policy_base,
                self.env,
                action_noise=action_noise,
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                policy_kwargs=policy_kwargs,
                learning_starts=self.cfg.getint("DRL", "learning_starts"),
                batch_size=self.cfg.getint("DRL", "batch_size"),
                train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                buffer_size=self.cfg.getint("DRL", "buffer_size"),
                tensorboard_log=str(self.artifacts.log_dir),
                seed=seed,
                verbose=2,
            )

        if algo == "SAC":
            model_variant = self.cfg.get("options", "model_variant", fallback="standard").lower()
            sac_policy = "MlpPolicy"
            sac_policy_kwargs = dict(policy_kwargs)
            if model_variant not in {"td3", "baseline", "standard"}:
                physical_feature_dim = self.cfg.getint(
                    "PGVL", "physical_feature_dim",
                    fallback=getattr(self.env.dynamic_model, "expert_feature_length", 5),
                )
                observed_physical_dim = int(
                    getattr(self.env.dynamic_model, "expert_feature_length", 0)
                )
                if physical_feature_dim != observed_physical_dim:
                    raise ValueError(
                        "PGVL physical_feature_dim must match the environment: "
                        f"configured={physical_feature_dim}, observed={observed_physical_dim}."
                    )
                sac_policy = PineSACPolicy
                sac_policy_kwargs.update(
                    {
                        "physical_feature_dim": physical_feature_dim,
                        "spatial_feature_dim": self.cfg.getint(
                            "PGVL", "spatial_feature_dim", fallback=31
                        ),
                        "use_adaptive_physics_gate": self.cfg.getboolean(
                            "PGVL", "use_adaptive_physics_gate", fallback=True
                        ),
                        "use_wind_residual_actor": self.cfg.getboolean(
                            "PGVL", "use_wind_residual_actor", fallback=True
                        ),
                        "residual_hidden_dim": self.cfg.getint(
                            "PGVL", "residual_hidden_dim", fallback=16
                        ),
                        "residual_max_action": self.cfg.getfloat(
                            "PGVL", "residual_max_action", fallback=0.20
                        ),
                        "residual_yaw_max_action": self.cfg.getfloat(
                            "PGVL", "residual_yaw_max_action", fallback=0.20
                        ),
                        "obstacle_feature_dim": self.cfg.getint(
                            "PGVL", "obstacle_feature_dim", fallback=25
                        ),
                        "obstacle_proximity_threshold": self.cfg.getfloat(
                            "PGVL", "obstacle_proximity_threshold", fallback=0.60
                        ),
                        "obstacle_transition": self.cfg.getfloat(
                            "PGVL", "obstacle_transition", fallback=0.15
                        ),
                        "minimum_wind_strength": self.cfg.getfloat(
                            "PGVL", "minimum_wind_strength", fallback=0.08
                        ),
                        "wind_strength_transition": self.cfg.getfloat(
                            "PGVL", "wind_strength_transition", fallback=0.10
                        ),
                        "minimum_wind_confidence": self.cfg.getfloat(
                            "PGVL", "minimum_wind_confidence", fallback=0.25
                        ),
                        "confidence_transition": self.cfg.getfloat(
                            "PGVL", "confidence_transition", fallback=0.25
                        ),
                    }
                )
            return SAC(
                sac_policy,
                self.env,
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                policy_kwargs=sac_policy_kwargs,
                learning_starts=self.cfg.getint("DRL", "learning_starts"),
                batch_size=self.cfg.getint("DRL", "batch_size"),
                train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                buffer_size=self.cfg.getint("DRL", "buffer_size"),
                ent_coef=self.cfg.get("DRL", "ent_coef", fallback="auto"),
                tensorboard_log=str(self.artifacts.log_dir),
                seed=seed,
                verbose=2,
            )

        if algo == "DDPG":
            return DDPG(
                policy_base,
                self.env,
                action_noise=action_noise,
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                policy_kwargs=policy_kwargs,
                learning_starts=self.cfg.getint("DRL", "learning_starts"),
                batch_size=self.cfg.getint("DRL", "batch_size"),
                train_freq=(self.cfg.getint("DRL", "train_freq"), "step"),
                gradient_steps=self.cfg.getint("DRL", "gradient_steps"),
                buffer_size=self.cfg.getint("DRL", "buffer_size"),
                tensorboard_log=str(self.artifacts.log_dir),
                seed=seed,
                verbose=2,
            )

        if algo == "PPO":
            return PPO(
                policy_base,
                self.env,
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                policy_kwargs=policy_kwargs,
                batch_size=self.cfg.getint("DRL", "batch_size"),
                tensorboard_log=str(self.artifacts.log_dir),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                seed=seed,
                verbose=2,
            )

        raise ValueError(f"Unsupported algorithm: {algo}")

    def build_callbacks(self, model) -> list[BaseCallback]:
        callbacks: list[BaseCallback] = []
        if isinstance(getattr(model, "action_noise", None), NormalActionNoise):
            initial_sigma, final_sigma = self._action_noise_sigmas()
            callbacks.append(
                LinearActionNoiseCallback(
                    initial_sigma=initial_sigma,
                    final_sigma=final_sigma,
                    total_timesteps=self.cfg.getint("options", "total_timesteps"),
                )
            )
        callbacks.extend([
            CrashRateCallback(log_freq=1000, env=self.env),
            RolloutMetricsCallback(self.artifacts.rollout_dir / "rollout_metrics.csv"),
        ])
        if self.cfg.getboolean("options", "use_wandb", fallback=False):
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(WandbCallback(model_save_freq=0, gradient_save_freq=0, verbose=2))
            callbacks.append(CheckpointCallback(save_freq=100000, save_path=str(self.artifacts.model_dir / "checkpoints")))
        return callbacks

    def run(self) -> Path:
        model = self.build_model()
        # ``run()`` trains in evaluation-sized segments so the single AirSim
        # vehicle can be reused for deterministic checkpoint evaluation.  SB3
        # otherwise creates a new TensorBoard writer on every ``learn()`` call,
        # which W&B imports as separate runs and renders as periodic gaps on
        # Windows.  Install one persistent logger before the first segment.
        model.set_logger(
            configure_logger(
                str(self.artifacts.log_dir),
                format_strings=["stdout", "tensorboard"],
            )
        )
        callbacks = self.build_callbacks(model)
        self.env.model = model
        self.env.data_path = str(self.artifacts.data_dir)

        total_timesteps = self.cfg.getint("options", "total_timesteps")
        eval_freq_steps = self.cfg.getint(
            "checkpoint", "eval_freq_steps", fallback=40_000
        )
        eval_episodes = self.cfg.getint(
            "checkpoint", "eval_episodes", fallback=50
        )
        eval_seed = self.cfg.getint(
            "checkpoint", "eval_seed", fallback=1_000_003
        )
        evaluator = DeterministicEvalBestModelCallback(
            self.artifacts.model_dir / "best_model" / "best_model",
            config=self.cfg,
            eval_freq_steps=eval_freq_steps,
            eval_episodes=eval_episodes,
            eval_seed=eval_seed,
            # AirSim has one configured vehicle.  Evaluation runs only after
            # a training segment has ended, then the VecEnv is resynchronized.
            eval_env=self.env,
        )
        evaluator.model = model
        evaluator._on_training_start()
        try:
            while model.num_timesteps < total_timesteps:
                segment_steps = min(
                    eval_freq_steps,
                    total_timesteps - model.num_timesteps,
                )
                model.learn(
                    segment_steps,
                    log_interval=4,
                    callback=callbacks,
                    reset_num_timesteps=False,
                )
                evaluator.num_timesteps = int(model.num_timesteps)
                evaluator._run_eval_if_due()

                # The evaluation deliberately reuses the single AirSim
                # vehicle, so restore the training VecEnv's initial
                # observation before the next segment.
                if model.num_timesteps < total_timesteps:
                    next_obs = model.env.reset()
                    model._last_obs = next_obs
                    model._last_original_obs = next_obs
            # Evaluate the final policy as well, even when total_timesteps is
            # not an exact multiple of eval_freq_steps.
            evaluator.num_timesteps = int(model.num_timesteps)
            if evaluator.last_eval_step != evaluator.num_timesteps:
                evaluator._run_eval_if_due(force=True)
        finally:
            evaluator._on_training_end()

        model_path = self.artifacts.model_dir / "model_sb3"
        model.save(str(model_path))
        print(f"training finished, model saved to: {model_path}")
        return model_path


class TrainingThread(QtCore.QThread):
    def __init__(self, config_file: str | Path):
        super().__init__()
        self.runner = TrainingRunner(config_file)
        self.cfg = self.runner.cfg
        self.env = self.runner.env

    def run(self):
        self.runner.run()
