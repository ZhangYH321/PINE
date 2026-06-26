from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch as th
import wandb
from PyQt5 import QtCore
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from scripts.core.config import PROJECT_ROOT, load_config, parse_net_arch
from scripts.core.registry import ensure_env_registered
from scripts.utils.pine_td3 import PINE_TD3


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
        if getattr(self.env, "episode_done", False):
            self.episode_count += 1
            if getattr(self.env, "crash_flag", False):
                self.crash_count += 1
        if self.step_counter % self.log_freq == 0 and self.episode_count > 0:
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


def build_training_artifacts(cfg) -> TrainingArtifacts:
    env_name = cfg.get("options", "env_name")
    now_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    root = PROJECT_ROOT / "logs" / env_name / f"{now_string}_{cfg.get('options', 'dynamic_name')}_{cfg.get('options', 'policy_name')}_{cfg.get('options', 'algo')}"
    return TrainingArtifacts(
        root=root,
        log_dir=root / "tb_logs",
        model_dir=root / "models",
        config_dir=root / "config",
        data_dir=root / "data",
    )


class TrainingRunner:
    """Owns environment creation, model construction, and training."""

    def __init__(self, config_file: str | Path):
        ensure_env_registered()
        self.config_file = Path(config_file)
        self.cfg = load_config(self.config_file)
        self.env = gym.make("airsim-env-v0")
        self.env.set_config(self.cfg)
        self.artifacts = build_training_artifacts(self.cfg)

        for path in (self.artifacts.log_dir, self.artifacts.model_dir, self.artifacts.config_dir, self.artifacts.data_dir):
            path.mkdir(parents=True, exist_ok=True)

        with open(self.artifacts.config_dir / "config.ini", "w", encoding="utf-8") as handle:
            self.cfg.write(handle)

        self.project_name = self.cfg.get("options", "env_name")

        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        if not self.cfg.getboolean("options", "use_wandb", fallback=False):
            return
        wandb.init(
            project=self.project_name,
            notes=self.cfg.get("wandb", "notes", fallback=""),
            name=self.cfg.get("wandb", "name", fallback=None),
            sync_tensorboard=True,
            save_code=False,
        )

    def build_model(self):
        activation_function = th.nn.Tanh if self.cfg.get("options", "activation_function") == "tanh" else th.nn.ReLU
        net_arch = parse_net_arch(self.cfg)
        algo = self.cfg.get("options", "algo").upper()
        n_actions = self.env.action_space.shape[-1]
        noise_sigma = self.cfg.getfloat("DRL", "action_noise_sigma") * np.ones(n_actions)
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
            physical_feature_dim = self.cfg.getint("PGVL", "physical_feature_dim", fallback=getattr(self.env.dynamic_model, "expert_feature_length", 4))
            policy_kwargs.update(
                {
                    "physical_feature_dim": physical_feature_dim,
                    "omega_spatial": self.cfg.getfloat("PGVL", "omega_spatial", fallback=0.7),
                    "omega_physical": self.cfg.getfloat("PGVL", "omega_physical", fallback=0.3),
                    "critic_mode": self.cfg.get("PGVL", "critic_mode", fallback="dual_stream"),
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
                pgvl_eta_rho=self.cfg.getfloat("PGVL", "eta_rho", fallback=1.0),
                seed=0,
                verbose=2,
            )

        if algo == "PPO":
            return PPO(
                policy_base,
                self.env,
                learning_rate=self.cfg.getfloat("DRL", "learning_rate"),
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(self.artifacts.log_dir),
                gamma=self.cfg.getfloat("DRL", "gamma"),
                seed=0,
                verbose=2,
            )

        raise ValueError(f"Unsupported algorithm: {algo}")

    def build_callbacks(self, model) -> list[BaseCallback]:
        callbacks: list[BaseCallback] = [
            EvalCallback(
                self.env,
                best_model_save_path=str(self.artifacts.model_dir / "best_model"),
                log_path=str(self.artifacts.log_dir / "eval"),
                eval_freq=20000,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
                verbose=1,
            ),
            CrashRateCallback(log_freq=1000, env=self.env),
        ]
        if self.cfg.getboolean("options", "use_wandb", fallback=False):
            from wandb.integration.sb3 import WandbCallback

            callbacks.append(WandbCallback(model_save_freq=0, gradient_save_freq=0, verbose=2))
            callbacks.append(CheckpointCallback(save_freq=100000, save_path=str(self.artifacts.model_dir / "checkpoints")))
        return callbacks

    def run(self) -> Path:
        model = self.build_model()
        callbacks = self.build_callbacks(model)
        self.env.model = model
        self.env.data_path = str(self.artifacts.data_dir)

        total_timesteps = self.cfg.getint("options", "total_timesteps")
        if self.cfg.getboolean("options", "use_wandb", fallback=False):
            model.learn(total_timesteps, log_interval=4, callback=callbacks)
        else:
            model.learn(total_timesteps)

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
