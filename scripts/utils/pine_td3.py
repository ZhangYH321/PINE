from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.td3.policies import TD3Policy


class PineReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    rho: th.Tensor


class PineReplayBuffer(ReplayBuffer):
    """Replay buffer that stores the PGVL sample weight rho_t from env infos."""

    def __init__(self, *args: Any, rho_default: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rho_default = float(rho_default)
        self.rhos = np.full((self.buffer_size, self.n_envs), self.rho_default, dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray,
            reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        rhos = np.full((self.n_envs,), self.rho_default, dtype=np.float32)
        for env_idx, info in enumerate(infos):
            if env_idx >= self.n_envs:
                break
            if isinstance(info, dict):
                rho_value = info.get("pgvl_rho", info.get("rho_t", self.rho_default))
                rhos[env_idx] = float(rho_value)

        self.rhos[self.pos] = np.nan_to_num(
            rhos,
            nan=self.rho_default,
            posinf=self.rho_default,
            neginf=self.rho_default,
        )
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> PineReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.rhos[batch_inds, env_indices].reshape(-1, 1),
        )
        return PineReplayBufferSamples(*tuple(map(self.to_torch, data)))


class PhysicsGuidedCritic(BaseModel):
    """
    TD3 critic with spatial and physical streams:
    Q_i = omega_spatial * Q_i^S(s_spatial, a) + omega_physical * Q_i^P(s_physical, a).
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        physical_feature_dim: int = 4,
        spatial_feature_dim: Optional[int] = None,
        omega_spatial: float = 0.7,
        omega_physical: float = 0.3,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.features_dim = int(features_dim)
        self.physical_feature_dim = int(physical_feature_dim)
        if self.physical_feature_dim <= 0:
            raise ValueError("PGVL requires physical_feature_dim > 0.")
        if self.physical_feature_dim >= self.features_dim:
            raise ValueError(
                f"physical_feature_dim={self.physical_feature_dim} must be smaller than features_dim={self.features_dim}."
            )

        inferred_spatial_dim = self.features_dim - self.physical_feature_dim
        self.spatial_feature_dim = inferred_spatial_dim if spatial_feature_dim is None else int(spatial_feature_dim)
        if self.spatial_feature_dim != inferred_spatial_dim:
            raise ValueError(
                "PGVL expects features ordered as [s_spatial, s_physical]; "
                f"spatial_feature_dim must be {inferred_spatial_dim}, got {self.spatial_feature_dim}."
            )

        self.omega_spatial = float(omega_spatial)
        self.omega_physical = float(omega_physical)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics

        action_dim = get_action_dim(self.action_space)
        self.spatial_q_networks: List[nn.Module] = []
        self.physical_q_networks: List[nn.Module] = []

        for idx in range(n_critics):
            spatial_net = nn.Sequential(
                *create_mlp(self.spatial_feature_dim + action_dim, 1, net_arch, activation_fn)
            )
            physical_net = nn.Sequential(
                *create_mlp(self.physical_feature_dim + action_dim, 1, net_arch, activation_fn)
            )
            self.add_module(f"qf{idx}_spatial", spatial_net)
            self.add_module(f"qf{idx}_physical", physical_net)
            self.spatial_q_networks.append(spatial_net)
            self.physical_q_networks.append(physical_net)

    def _split_features(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if features.dim() > 2:
            features = th.flatten(features, start_dim=1)
        if features.shape[1] != self.features_dim:
            raise RuntimeError(f"PGVL critic expected {self.features_dim} features, got {features.shape[1]}.")
        spatial_features = features[:, :self.spatial_feature_dim]
        physical_features = features[:, -self.physical_feature_dim:]
        return spatial_features, physical_features

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        spatial_features, physical_features = self._split_features(features)

        spatial_input = th.cat([spatial_features, actions], dim=1)
        physical_input = th.cat([physical_features, actions], dim=1)
        return tuple(
            self.omega_spatial * spatial_net(spatial_input)
            + self.omega_physical * physical_net(physical_input)
            for spatial_net, physical_net in zip(self.spatial_q_networks, self.physical_q_networks)
        )

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        spatial_features, physical_features = self._split_features(features)

        spatial_input = th.cat([spatial_features, actions], dim=1)
        physical_input = th.cat([physical_features, actions], dim=1)
        return (
            self.omega_spatial * self.spatial_q_networks[0](spatial_input)
            + self.omega_physical * self.physical_q_networks[0](physical_input)
        )


class PineTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        physical_feature_dim: int = 4,
        spatial_feature_dim: Optional[int] = None,
        omega_spatial: float = 0.7,
        omega_physical: float = 0.3,
        critic_mode: str = "dual_stream",
    ) -> None:
        self.physical_feature_dim = physical_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.omega_spatial = omega_spatial
        self.omega_physical = omega_physical
        self.critic_mode = critic_mode.lower()
        if self.critic_mode not in {"dual_stream", "single_stream"}:
            raise ValueError("critic_mode must be 'dual_stream' or 'single_stream'.")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[PhysicsGuidedCritic, ContinuousCritic]:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.critic_mode == "single_stream":
            return ContinuousCritic(**critic_kwargs).to(self.device)

        critic_kwargs.update(
            {
                "physical_feature_dim": self.physical_feature_dim,
                "spatial_feature_dim": self.spatial_feature_dim,
                "omega_spatial": self.omega_spatial,
                "omega_physical": self.omega_physical,
            }
        )
        return PhysicsGuidedCritic(**critic_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                physical_feature_dim=self.physical_feature_dim,
                spatial_feature_dim=self.spatial_feature_dim,
                omega_spatial=self.omega_spatial,
                omega_physical=self.omega_physical,
                critic_mode=self.critic_mode,
            )
        )
        return data


class PineCnnPolicy(PineTD3Policy):
    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        physical_feature_dim: int = 4,
        spatial_feature_dim: Optional[int] = None,
        omega_spatial: float = 0.7,
        omega_physical: float = 0.3,
        critic_mode: str = "dual_stream",
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            physical_feature_dim=physical_feature_dim,
            spatial_feature_dim=spatial_feature_dim,
            omega_spatial=omega_spatial,
            omega_physical=omega_physical,
            critic_mode=critic_mode,
        )


class PineMultiInputPolicy(PineTD3Policy):
    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        physical_feature_dim: int = 4,
        spatial_feature_dim: Optional[int] = None,
        omega_spatial: float = 0.7,
        omega_physical: float = 0.3,
        critic_mode: str = "dual_stream",
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            physical_feature_dim=physical_feature_dim,
            spatial_feature_dim=spatial_feature_dim,
            omega_spatial=omega_spatial,
            omega_physical=omega_physical,
            critic_mode=critic_mode,
        )


PineMlpPolicy = PineTD3Policy


class PINE_TD3(TD3):
    """TD3 with PGVL rho_t-weighted TD loss and selectable critic structure."""

    policy_aliases = {
        "MlpPolicy": PineMlpPolicy,
        "CnnPolicy": PineCnnPolicy,
        "MultiInputPolicy": PineMultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Any,
        learning_rate: Union[float, Any] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        pgvl_eta_rho: float = 1.0,
    ) -> None:
        self.pgvl_eta_rho = float(pgvl_eta_rho)
        if replay_buffer_class is None:
            replay_buffer_class = PineReplayBuffer

        replay_buffer_kwargs = dict(replay_buffer_kwargs or {})
        if replay_buffer_class is PineReplayBuffer or issubclass(replay_buffer_class, PineReplayBuffer):
            replay_buffer_kwargs.setdefault("rho_default", 1.0)

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        rho_means, rho_maxes = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            rho_t = getattr(replay_data, "rho", th.ones_like(target_q_values)).float().clamp_min(0.0)

            critic_loss = sum(
                (rho_t * F.mse_loss(current_q, target_q_values, reduction="none")).mean()
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            rho_means.append(rho_t.mean().item())
            rho_maxes.append(rho_t.max().item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/pgvl_rho_mean", np.mean(rho_means))
        self.logger.record("train/pgvl_rho_max", np.max(rho_maxes))


PineTD3 = PINE_TD3


__all__ = [
    "PineTD3",
    "PINE_TD3",
    "PineTD3Policy",
    "PineMlpPolicy",
    "PineCnnPolicy",
    "PineMultiInputPolicy",
    "PhysicsGuidedCritic",
    "PineReplayBuffer",
]
