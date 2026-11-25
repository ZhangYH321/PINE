from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import Actor, TD3Policy


class PineReplayBuffer(ReplayBuffer):
    pass


class PhysicsGuidedCritic(BaseModel):


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
        physical_feature_dim: int = 5,
        spatial_feature_dim: Optional[int] = None,
        use_physics_branch: Optional[bool] = None,
        use_adaptive_physics_gate: bool = True,
        obstacle_feature_dim: int = 25,
        obstacle_proximity_threshold: float = 0.60,
        obstacle_transition: float = 0.15,
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

        self.use_adaptive_physics_gate = bool(use_adaptive_physics_gate)


        self.use_physics_branch = (
            self.use_adaptive_physics_gate
            if use_physics_branch is None
            else bool(use_physics_branch)
        )
        self.obstacle_feature_dim = int(obstacle_feature_dim)
        self.obstacle_proximity_threshold = float(obstacle_proximity_threshold)
        self.obstacle_transition = float(obstacle_transition)
        if not 0 < self.obstacle_feature_dim <= self.spatial_feature_dim:
            raise ValueError("obstacle_feature_dim must fit inside spatial features.")
        if self.obstacle_proximity_threshold <= 0.0 or self.obstacle_transition <= 0.0:
            raise ValueError("Obstacle gate threshold and transition must be positive.")
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics

        action_dim = get_action_dim(self.action_space)
        self.spatial_q_networks: List[nn.Module] = []
        self.physical_q_networks: List[nn.Module] = []

        for idx in range(n_critics):
            spatial_net = nn.Sequential(
                *create_mlp(self.spatial_feature_dim + action_dim, 1, net_arch, activation_fn)
            )
            self.add_module(f"qf{idx}_spatial", spatial_net)
            self.spatial_q_networks.append(spatial_net)
            if self.use_physics_branch:
                physical_net = nn.Sequential(
                    *create_mlp(self.physical_feature_dim + action_dim, 1, net_arch, activation_fn)
                )
                self.add_module(f"qf{idx}_physical", physical_net)
                self.physical_q_networks.append(physical_net)

    def _split_features(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if features.dim() > 2:
            features = th.flatten(features, start_dim=1)
        if features.shape[1] != self.features_dim:
            raise RuntimeError(f"PGVL critic expected {self.features_dim} features, got {features.shape[1]}.")
        spatial_features = features[:, :self.spatial_feature_dim]
        physical_features = features[:, -self.physical_feature_dim:]
        return spatial_features, physical_features

    @staticmethod
    def _smooth_gate(value: th.Tensor, threshold: float, transition: float) -> th.Tensor:
        x = ((value - threshold) / transition).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _adaptive_gate(
        self, spatial_features: th.Tensor, physical_features: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        wind_along_goal = physical_features[:, 0:1].clamp(0.0, 1.0)
        headwind_strength = ((0.5 - wind_along_goal) * 2.0).clamp(0.0, 1.0)
        confidence = physical_features[:, 4:5].clamp(0.0, 1.0)
        obstacle_raw = spatial_features[:, : self.obstacle_feature_dim].amax(
            dim=1, keepdim=True
        ).clamp(0.0, 1.0)
        obstacle_proximity = self._smooth_gate(
            obstacle_raw,
            self.obstacle_proximity_threshold - self.obstacle_transition,
            self.obstacle_transition,
        )
        gate = confidence * headwind_strength * (1.0 - obstacle_proximity)
        return gate, headwind_strength, confidence, obstacle_proximity

    def physics_gate(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:

        features = self.extract_features(obs, self.features_extractor)
        spatial_features, physical_features = self._split_features(features)
        return self._adaptive_gate(spatial_features, physical_features)

    def _fuse_q(
        self,
        spatial_q: th.Tensor,
        physical_q: Optional[th.Tensor] = None,
        gate: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if not self.use_physics_branch:
            return spatial_q
        assert physical_q is not None
        if not self.use_adaptive_physics_gate:
            return spatial_q + physical_q
        assert gate is not None
        return spatial_q + gate * physical_q

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        spatial_features, physical_features = self._split_features(features)

        spatial_input = th.cat([spatial_features, actions], dim=1)
        if not self.use_physics_branch:
            return tuple(spatial_net(spatial_input) for spatial_net in self.spatial_q_networks)
        physical_input = th.cat([physical_features, actions], dim=1)
        gate = (
            self._adaptive_gate(spatial_features, physical_features)[0]
            if self.use_adaptive_physics_gate
            else None
        )
        return tuple(
            self._fuse_q(
                spatial_net(spatial_input),
                physical_net(physical_input),
                gate,
            )
            for spatial_net, physical_net in zip(self.spatial_q_networks, self.physical_q_networks)
        )

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        spatial_features, physical_features = self._split_features(features)

        spatial_input = th.cat([spatial_features, actions], dim=1)
        if not self.use_physics_branch:
            return self.spatial_q_networks[0](spatial_input)
        physical_input = th.cat([physical_features, actions], dim=1)
        gate = (
            self._adaptive_gate(spatial_features, physical_features)[0]
            if self.use_adaptive_physics_gate
            else None
        )
        return self._fuse_q(
            self.spatial_q_networks[0](spatial_input),
            self.physical_q_networks[0](physical_input),
            gate,
        )


class GatedWindResidualActor(Actor):


    def __init__(
        self,
        *args: Any,
        physical_feature_dim: int = 5,
        residual_hidden_dim: int = 16,
        residual_max_action: float = 0.20,
        residual_yaw_max_action: float = 0.20,
        obstacle_feature_dim: int = 25,
        obstacle_proximity_threshold: float = 0.60,
        obstacle_transition: float = 0.15,
        minimum_wind_strength: float = 0.08,
        wind_strength_transition: float = 0.10,
        minimum_wind_confidence: float = 0.25,
        confidence_transition: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.physical_feature_dim = int(physical_feature_dim)
        self.spatial_feature_dim = self.features_dim - self.physical_feature_dim
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.residual_max_action = float(residual_max_action)
        self.residual_yaw_max_action = float(residual_yaw_max_action)
        self.obstacle_feature_dim = int(obstacle_feature_dim)
        self.obstacle_proximity_threshold = float(obstacle_proximity_threshold)
        self.obstacle_transition = float(obstacle_transition)
        self.minimum_wind_strength = float(minimum_wind_strength)
        self.wind_strength_transition = float(wind_strength_transition)
        self.minimum_wind_confidence = float(minimum_wind_confidence)
        self.confidence_transition = float(confidence_transition)

        if self.physical_feature_dim != 5:
            raise ValueError("GatedWindResidualActor expects five physical features.")
        if self.spatial_feature_dim <= 0:
            raise ValueError("physical_feature_dim must be smaller than features_dim.")
        if not 0 < self.obstacle_feature_dim <= self.spatial_feature_dim:
            raise ValueError("obstacle_feature_dim must fit inside spatial features.")
        positive_values = (
            self.residual_hidden_dim,
            self.residual_max_action,
            self.residual_yaw_max_action,
            self.obstacle_transition,
            self.wind_strength_transition,
            self.confidence_transition,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("Residual dimensions, limits, and transitions must be positive.")

        action_dim = get_action_dim(self.action_space)
        self.mu = nn.Sequential(
            *create_mlp(
                self.spatial_feature_dim,
                action_dim,
                self.net_arch,
                self.activation_fn,
                squash_output=True,
            )
        )
        self.wind_residual = nn.Sequential(
            nn.Linear(self.features_dim, self.residual_hidden_dim),
            self.activation_fn(),
            nn.Linear(self.residual_hidden_dim, 2),
            nn.Tanh(),
        )
        nn.init.zeros_(self.wind_residual[-2].weight)
        nn.init.zeros_(self.wind_residual[-2].bias)

    @staticmethod
    def _smooth_gate(value: th.Tensor, threshold: float, transition: float) -> th.Tensor:
        x = ((value - threshold) / transition).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        if features.dim() > 2:
            features = th.flatten(features, start_dim=1)
        spatial = features[:, : self.spatial_feature_dim]
        physical = features[:, -self.physical_feature_dim :]
        navigation_action = self.mu(spatial)

        obstacle_proximity = spatial[:, : self.obstacle_feature_dim].amax(
            dim=1, keepdim=True
        )
        obstacle_near_gate = self._smooth_gate(
            obstacle_proximity,
            self.obstacle_proximity_threshold - self.obstacle_transition,
            self.obstacle_transition,
        )
        obstacle_far_gate = 1.0 - obstacle_near_gate
        normalized_wind = 2.0 * (physical[:, :2] - 0.5)
        wind_strength = th.linalg.vector_norm(normalized_wind, dim=1, keepdim=True)
        wind_gate = self._smooth_gate(
            wind_strength,
            self.minimum_wind_strength,
            self.wind_strength_transition,
        )
        confidence_gate = self._smooth_gate(
            physical[:, 4:5],
            self.minimum_wind_confidence,
            self.confidence_transition,
        )
        wind_confident_gate = wind_gate * confidence_gate
        residual = self.wind_residual(features)


        negative_speed = th.minimum(residual[:, 0:1], th.zeros_like(residual[:, 0:1]))
        speed_residual = self.residual_max_action * (

            obstacle_far_gate * wind_confident_gate * residual[:, 0:1]


            + obstacle_near_gate * negative_speed
        )


        yaw_residual = self.residual_yaw_max_action * wind_confident_gate * residual[:, 1:2]

        action = navigation_action.clone()
        action[:, 0:1] = (action[:, 0:1] + speed_residual).clamp(-1.0, 1.0)
        action[:, -1:] = (action[:, -1:] + yaw_residual).clamp(-1.0, 1.0)
        return action


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
        physical_feature_dim: int = 5,
        spatial_feature_dim: Optional[int] = None,
        use_physics_branch: Optional[bool] = None,
        use_adaptive_physics_gate: bool = True,
        use_wind_residual_actor: bool = True,
        residual_hidden_dim: int = 16,
        residual_max_action: float = 0.20,
        residual_yaw_max_action: float = 0.20,
        obstacle_feature_dim: int = 25,
        obstacle_proximity_threshold: float = 0.60,
        obstacle_transition: float = 0.15,
        minimum_wind_strength: float = 0.08,
        wind_strength_transition: float = 0.10,
        minimum_wind_confidence: float = 0.25,
        confidence_transition: float = 0.25,
    ) -> None:
        self.physical_feature_dim = physical_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.use_adaptive_physics_gate = bool(use_adaptive_physics_gate)
        self.use_physics_branch = (
            self.use_adaptive_physics_gate
            if use_physics_branch is None
            else bool(use_physics_branch)
        )
        self.use_wind_residual_actor = bool(use_wind_residual_actor)
        self.residual_actor_kwargs = {
            "residual_hidden_dim": int(residual_hidden_dim),
            "residual_max_action": float(residual_max_action),
            "residual_yaw_max_action": float(residual_yaw_max_action),
            "obstacle_feature_dim": int(obstacle_feature_dim),
            "obstacle_proximity_threshold": float(obstacle_proximity_threshold),
            "obstacle_transition": float(obstacle_transition),
            "minimum_wind_strength": float(minimum_wind_strength),
            "wind_strength_transition": float(wind_strength_transition),
            "minimum_wind_confidence": float(minimum_wind_confidence),
            "confidence_transition": float(confidence_transition),
        }
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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        if not self.use_wind_residual_actor:
            return Actor(**actor_kwargs).to(self.device)
        return GatedWindResidualActor(
            **actor_kwargs,
            physical_feature_dim=self.physical_feature_dim,
            **self.residual_actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PhysicsGuidedCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(
            {
                "physical_feature_dim": self.physical_feature_dim,
                "spatial_feature_dim": self.spatial_feature_dim,
                "use_physics_branch": self.use_physics_branch,
                "use_adaptive_physics_gate": self.use_adaptive_physics_gate,
                "obstacle_feature_dim": self.residual_actor_kwargs["obstacle_feature_dim"],
                "obstacle_proximity_threshold": self.residual_actor_kwargs[
                    "obstacle_proximity_threshold"
                ],
                "obstacle_transition": self.residual_actor_kwargs["obstacle_transition"],
            }
        )
        return PhysicsGuidedCritic(**critic_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                physical_feature_dim=self.physical_feature_dim,
                spatial_feature_dim=self.spatial_feature_dim,
                use_physics_branch=self.use_physics_branch,
                use_adaptive_physics_gate=self.use_adaptive_physics_gate,
                use_wind_residual_actor=self.use_wind_residual_actor,
                **self.residual_actor_kwargs,
            )
        )
        return data


PineMlpPolicy = PineTD3Policy


class PINE_TD3(TD3):


    policy_aliases = {
        "MlpPolicy": PineMlpPolicy,
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
    ) -> None:
        if replay_buffer_class is None:
            replay_buffer_class = PineReplayBuffer

        replay_buffer_kwargs = dict(replay_buffer_kwargs or {})

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

        actor_losses, critic_losses, gate_means = [], [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            if self.critic.use_adaptive_physics_gate:
                with th.no_grad():
                    gate_means.append(self.critic.physics_gate(replay_data.observations)[0].mean().item())

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
        if gate_means:
            self.logger.record("train/physics_gate_mean", np.mean(gate_means))


PineTD3 = PINE_TD3


__all__ = [
    "PineTD3",
    "PINE_TD3",
    "PineTD3Policy",
    "PineMlpPolicy",
    "GatedWindResidualActor",
    "PhysicsGuidedCritic",
    "PineReplayBuffer",
]
