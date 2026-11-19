"""PINE policy components for Soft Actor-Critic.

The SAC variant keeps SB3's stochastic actor and entropy-regularized update,
while using the same spatial/physical critic fusion and gated speed residual as
the PINE-TD3 implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy


class GatedWindResidualSACActor(Actor):
    """SAC actor with the same gated speed/yaw residual as PINE-TD3."""

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
        self.residual_max_action = float(residual_max_action)
        self.residual_yaw_max_action = float(residual_yaw_max_action)
        self.obstacle_feature_dim = int(obstacle_feature_dim)
        self.obstacle_proximity_threshold = float(obstacle_proximity_threshold)
        self.obstacle_transition = float(obstacle_transition)
        self.minimum_wind_strength = float(minimum_wind_strength)
        self.wind_strength_transition = float(wind_strength_transition)
        self.minimum_wind_confidence = float(minimum_wind_confidence)
        self.confidence_transition = float(confidence_transition)
        if self.physical_feature_dim != 5 or self.spatial_feature_dim <= 0:
            raise ValueError("PINE SAC expects five physical features after spatial features.")
        if not 0 < self.obstacle_feature_dim <= self.spatial_feature_dim:
            raise ValueError("obstacle_feature_dim must fit inside spatial features.")
        positive = (
            residual_hidden_dim,
            self.residual_max_action,
            self.residual_yaw_max_action,
            self.obstacle_transition,
            self.wind_strength_transition,
            self.confidence_transition,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("Residual dimensions, limits, and transitions must be positive.")
        self.wind_residual = nn.Sequential(
            nn.Linear(self.features_dim, int(residual_hidden_dim)),
            self.activation_fn(),
            nn.Linear(int(residual_hidden_dim), 2),
            nn.Tanh(),
        )
        nn.init.zeros_(self.wind_residual[-2].weight)
        nn.init.zeros_(self.wind_residual[-2].bias)

    @staticmethod
    def _smooth_gate(value: th.Tensor, threshold: float, transition: float) -> th.Tensor:
        x = ((value - threshold) / transition).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _residual(self, features: th.Tensor) -> th.Tensor:
        spatial = features[:, : self.spatial_feature_dim]
        physical = features[:, -self.physical_feature_dim :]
        obstacle = spatial[:, : self.obstacle_feature_dim].amax(dim=1, keepdim=True)
        obstacle_near_gate = self._smooth_gate(
            obstacle,
            self.obstacle_proximity_threshold - self.obstacle_transition,
            self.obstacle_transition,
        )
        obstacle_far_gate = 1.0 - obstacle_near_gate
        wind = 2.0 * (physical[:, :2] - 0.5)
        wind_gate = self._smooth_gate(
            th.linalg.vector_norm(wind, dim=1, keepdim=True),
            self.minimum_wind_strength,
            self.wind_strength_transition,
        )
        confidence_gate = self._smooth_gate(
            physical[:, 4:5], self.minimum_wind_confidence, self.confidence_transition
        )
        wind_confident_gate = wind_gate * confidence_gate
        residual = self.wind_residual(features)
        negative_speed = th.minimum(residual[:, 0:1], th.zeros_like(residual[:, 0:1]))
        speed_residual = self.residual_max_action * (
            obstacle_far_gate * wind_confident_gate * residual[:, 0:1]
            + obstacle_near_gate * negative_speed
        )
        yaw_residual = self.residual_yaw_max_action * wind_confident_gate * residual[:, 1:2]
        return th.cat([speed_residual, yaw_residual], dim=1)

    def get_action_dist_params(self, obs):
        mean_actions, log_std, kwargs = super().get_action_dist_params(obs)
        features = self.extract_features(obs, self.features_extractor)
        if features.dim() > 2:
            features = th.flatten(features, start_dim=1)
        residual = self._residual(features)
        mean_actions = mean_actions.clone()
        mean_actions[:, 0:1] = mean_actions[:, 0:1] + residual[:, 0:1]
        mean_actions[:, -1:] = mean_actions[:, -1:] + residual[:, 1:2]
        return mean_actions, log_std, kwargs


class PineSACCritic(BaseModel):
    """SAC critic with optional adaptive spatial/physical value fusion."""

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
        share_features_extractor: bool = False,
        physical_feature_dim: int = 5,
        spatial_feature_dim: Optional[int] = None,
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
        self.spatial_feature_dim = self.features_dim - self.physical_feature_dim if spatial_feature_dim is None else int(spatial_feature_dim)
        if self.spatial_feature_dim + self.physical_feature_dim != self.features_dim:
            raise ValueError("PINE SAC feature dimensions do not match the observation.")
        self.use_adaptive_physics_gate = bool(use_adaptive_physics_gate)
        self.obstacle_feature_dim = int(obstacle_feature_dim)
        self.obstacle_proximity_threshold = float(obstacle_proximity_threshold)
        self.obstacle_transition = float(obstacle_transition)
        action_dim = get_action_dim(action_space)
        self.n_critics = int(n_critics)
        self.share_features_extractor = bool(share_features_extractor)
        self.spatial_q_networks: List[nn.Module] = []
        self.physical_q_networks: List[nn.Module] = []
        for idx in range(self.n_critics):
            spatial_net = nn.Sequential(*create_mlp(self.spatial_feature_dim + action_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{idx}_spatial", spatial_net)
            self.spatial_q_networks.append(spatial_net)
            if self.use_adaptive_physics_gate:
                physical_net = nn.Sequential(*create_mlp(self.physical_feature_dim + action_dim, 1, net_arch, activation_fn))
                self.add_module(f"qf{idx}_physical", physical_net)
                self.physical_q_networks.append(physical_net)

    @staticmethod
    def _smooth_gate(value: th.Tensor, threshold: float, transition: float) -> th.Tensor:
        x = ((value - threshold) / transition).clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _split(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if features.dim() > 2:
            features = th.flatten(features, start_dim=1)
        return features[:, : self.spatial_feature_dim], features[:, -self.physical_feature_dim :]

    def _gate(self, spatial: th.Tensor, physical: th.Tensor) -> th.Tensor:
        wind = physical[:, 0:1].clamp(0.0, 1.0)
        headwind = ((0.5 - wind) * 2.0).clamp(0.0, 1.0)
        confidence = physical[:, 4:5].clamp(0.0, 1.0)
        obstacle = spatial[:, : self.obstacle_feature_dim].amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        obstacle_gate = self._smooth_gate(
            obstacle,
            self.obstacle_proximity_threshold - self.obstacle_transition,
            self.obstacle_transition,
        )
        return confidence * headwind * (1.0 - obstacle_gate)

    def _fuse(self, spatial_q: th.Tensor, physical_q: Optional[th.Tensor] = None, gate: Optional[th.Tensor] = None) -> th.Tensor:
        if not self.use_adaptive_physics_gate:
            return spatial_q
        assert physical_q is not None and gate is not None
        return spatial_q + gate * physical_q

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        spatial, physical = self._split(features)
        spatial_input = th.cat([spatial, actions], dim=1)
        if not self.use_adaptive_physics_gate:
            return tuple(s_net(spatial_input) for s_net in self.spatial_q_networks)
        gate = self._gate(spatial, physical)
        return tuple(
            self._fuse(s_net(th.cat([spatial, actions], dim=1)), p_net(th.cat([physical, actions], dim=1)), gate)
            for s_net, p_net in zip(self.spatial_q_networks, self.physical_q_networks)
        )

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        spatial, physical = self._split(features)
        spatial_input = th.cat([spatial, actions], dim=1)
        if not self.use_adaptive_physics_gate:
            return self.spatial_q_networks[0](spatial_input)
        gate = self._gate(spatial, physical)
        return self._fuse(
            self.spatial_q_networks[0](th.cat([spatial, actions], dim=1)),
            self.physical_q_networks[0](th.cat([physical, actions], dim=1)),
            gate,
        )


class PineSACPolicy(SACPolicy):
    """SAC policy using the PINE actor and critic components."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.physical_feature_dim = int(kwargs.pop("physical_feature_dim", 5))
        self.spatial_feature_dim = kwargs.pop("spatial_feature_dim", None)
        self.use_adaptive_physics_gate = bool(kwargs.pop("use_adaptive_physics_gate", True))
        self.use_wind_residual_actor = bool(kwargs.pop("use_wind_residual_actor", True))
        self.pine_kwargs = {
            "residual_hidden_dim": int(kwargs.pop("residual_hidden_dim", 16)),
            "residual_max_action": float(kwargs.pop("residual_max_action", 0.20)),
            "residual_yaw_max_action": float(kwargs.pop("residual_yaw_max_action", 0.20)),
            "obstacle_feature_dim": int(kwargs.pop("obstacle_feature_dim", 25)),
            "obstacle_proximity_threshold": float(kwargs.pop("obstacle_proximity_threshold", 0.60)),
            "obstacle_transition": float(kwargs.pop("obstacle_transition", 0.15)),
            "minimum_wind_strength": float(kwargs.pop("minimum_wind_strength", 0.08)),
            "wind_strength_transition": float(kwargs.pop("wind_strength_transition", 0.10)),
            "minimum_wind_confidence": float(kwargs.pop("minimum_wind_confidence", 0.25)),
            "confidence_transition": float(kwargs.pop("confidence_transition", 0.25)),
        }
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        if not self.use_wind_residual_actor:
            return Actor(**actor_kwargs).to(self.device)
        return GatedWindResidualSACActor(
            **actor_kwargs,
            physical_feature_dim=self.physical_feature_dim,
            **self.pine_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PineSACCritic(
            **critic_kwargs,
            physical_feature_dim=self.physical_feature_dim,
            spatial_feature_dim=self.spatial_feature_dim,
            use_adaptive_physics_gate=self.use_adaptive_physics_gate,
            obstacle_feature_dim=self.pine_kwargs["obstacle_feature_dim"],
            obstacle_proximity_threshold=self.pine_kwargs["obstacle_proximity_threshold"],
            obstacle_transition=self.pine_kwargs["obstacle_transition"],
        ).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "physical_feature_dim": self.physical_feature_dim,
                "spatial_feature_dim": self.spatial_feature_dim,
                "use_adaptive_physics_gate": self.use_adaptive_physics_gate,
                "use_wind_residual_actor": self.use_wind_residual_actor,
                **self.pine_kwargs,
            }
        )
        return data


__all__ = ["PineSACPolicy", "PineSACCritic", "GatedWindResidualSACActor"]
