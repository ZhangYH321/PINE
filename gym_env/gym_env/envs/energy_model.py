from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .power_model import MultirotorPowerModel, RotorcraftParams


@dataclass(frozen=True)
class ElectricalPowerSample:
    battery_power_w: float
    shaft_power_w: float
    source: str


def wind_assist_energy_per_progress(
    battery_power_w: float,
    no_wind_power_w: float,
    ground_velocity: Sequence[float],
    commanded_airspeed: Sequence[float],
    goal_direction: Sequence[float],
    goal_progress_m: float,
    minimum_progress_speed: float = 0.1,
) -> float:


    progress = max(float(goal_progress_m), 0.0)
    if progress <= 0.0:
        return 0.0

    direction = np.asarray(goal_direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if direction.size != 3 or not np.isfinite(norm) or norm <= 1e-9:
        return 0.0
    direction = direction / norm

    ground_progress_speed = float(
        np.dot(np.asarray(ground_velocity, dtype=float), direction)
    )
    air_progress_speed = float(
        np.dot(np.asarray(commanded_airspeed, dtype=float), direction)
    )
    epsilon = max(float(minimum_progress_speed), 1e-6)
    actual_cost_j_per_m = float(battery_power_w) / max(
        ground_progress_speed, epsilon
    )
    no_wind_cost_j_per_m = float(no_wind_power_w) / max(
        air_progress_speed, epsilon
    )
    return float((no_wind_cost_j_per_m - actual_cost_j_per_m) * progress)


class ForwardFlightPowerModel:


    def __init__(self, params: RotorcraftParams, profile_advance_coefficient: float = 4.6) -> None:
        self.params = params
        self.profile_advance_coefficient = float(profile_advance_coefficient)
        rotor_area = math.pi * params.rotor_radius**2
        self.total_disk_area = params.n_rotors * rotor_area
        self.rotor_tip_speed = params.rotor_speed * params.rotor_radius
        hover_thrust = params.mass * params.g
        self.hover_induced_velocity = math.sqrt(
            hover_thrust / (2.0 * params.rho * self.total_disk_area)
        )
        self.hover_induced_power = (
            params.induced_loss_factor * hover_thrust * self.hover_induced_velocity
        )
        profile_power_single = (
            (params.solidity * params.profile_drag_coeff / 8.0)
            * params.rho
            * rotor_area
            * self.rotor_tip_speed**3
        )
        self.hover_profile_power = params.n_rotors * profile_power_single

    def power(self, v_airspeed: Sequence[float]) -> float:
        va_n, va_e, va_d = np.asarray(v_airspeed, dtype=float)
        horizontal_speed = math.hypot(va_n, va_e)
        vertical_speed_up = -va_d

        v0 = max(self.hover_induced_velocity, 1e-6)
        forward_factor = math.sqrt(
            max(
                math.sqrt(1.0 + horizontal_speed**4 / (4.0 * v0**4))
                - horizontal_speed**2 / (2.0 * v0**2),
                0.0,
            )
        )
        induced_power = self.hover_induced_power * forward_factor
        if self.params.penalize_climb_only:
            climb_speed = max(0.0, vertical_speed_up)
        else:
            climb_speed = abs(vertical_speed_up)
        induced_power += (
            self.params.climb_power_decay
            * self.params.mass
            * self.params.g
            * climb_speed
        )

        advance_ratio = horizontal_speed / max(self.rotor_tip_speed, 1e-6)
        profile_power = self.hover_profile_power * (
            1.0 + self.profile_advance_coefficient * advance_ratio**2
        )
        parasitic_power = (
            0.5
            * self.params.rho
            * self.params.body_drag_area
            * horizontal_speed**3
        )
        return float(induced_power + profile_power + parasitic_power)


class AirSimRotorEnergyModel:


    AIRSIM_ROTOR = "airsim_rotor"
    HOVER_CALIBRATED_FORWARD = "hover_calibrated_forward"
    SUPPORTED_POWER_MODEL_MODES = {
        AIRSIM_ROTOR,
        HOVER_CALIBRATED_FORWARD,
    }

    def __init__(
        self,
        client,
        fallback_model: MultirotorPowerModel,
        params: RotorcraftParams,
        motor_esc_efficiency: float = 0.82,
        auxiliary_power_w: float = 10.0,
        expected_rotors: int = 4,
        profile_advance_coefficient: float = 4.6,
        power_model_mode: str = AIRSIM_ROTOR,
        fixed_hover_battery_power_w: float = 0.0,
    ) -> None:
        self.client = client
        self.fallback_model = fallback_model
        self.motor_esc_efficiency = float(motor_esc_efficiency)
        self.auxiliary_power_w = float(auxiliary_power_w)
        self.expected_rotors = int(expected_rotors)
        self.power_model_mode = str(power_model_mode).strip().lower()
        if not 0.0 < self.motor_esc_efficiency <= 1.0:
            raise ValueError("motor_esc_efficiency must be in (0, 1].")
        if self.auxiliary_power_w < 0.0:
            raise ValueError("auxiliary_power_w must be non-negative.")
        if self.expected_rotors <= 0:
            raise ValueError("expected_rotors must be positive.")
        if self.power_model_mode not in self.SUPPORTED_POWER_MODEL_MODES:
            supported = ", ".join(sorted(self.SUPPORTED_POWER_MODEL_MODES))
            raise ValueError(
                f"Unsupported energy.power_model_mode={power_model_mode!r}; "
                f"expected one of: {supported}."
            )

        self.counterfactual_model = ForwardFlightPowerModel(
            params,
            profile_advance_coefficient=profile_advance_coefficient,
        )
        self.previous_power_w: float | None = None
        self.total_energy_j = 0.0
        self.fallback_count = 0
        fixed_hover = float(fixed_hover_battery_power_w)
        if not np.isfinite(fixed_hover) or fixed_hover < 0.0:
            raise ValueError("fixed_hover_battery_power_w must be finite and non-negative.")
        if 0.0 < fixed_hover <= self.auxiliary_power_w:
            raise ValueError(
                "fixed_hover_battery_power_w must exceed auxiliary_power_w."
            )
        self.fixed_hover_battery_power_w = fixed_hover or None
        self.hover_battery_power_w: float | None = self.fixed_hover_battery_power_w

    def _fallback_sample(self, v_airspeed: Sequence[float]) -> ElectricalPowerSample:
        components = self.fallback_model.power_components(v_airspeed)
        shaft_power = float(sum(components[:3]))
        battery_power = shaft_power / self.motor_esc_efficiency + self.auxiliary_power_w
        self.fallback_count += 1
        return ElectricalPowerSample(battery_power, shaft_power, "analytical_fallback")

    def _sample_airsim_rotor(self, v_airspeed: Sequence[float]) -> ElectricalPowerSample:
        try:
            rotor_states = self.client.getRotorStates()
            rotors = list(getattr(rotor_states, "rotors", []))
            if len(rotors) != self.expected_rotors:
                raise ValueError(
                    f"expected {self.expected_rotors} rotors, received {len(rotors)}"
                )
            shaft_power = 0.0
            for rotor in rotors:
                torque = float(rotor["torque_scaler"])
                speed = float(rotor["speed"])
                if not np.isfinite(torque) or not np.isfinite(speed):
                    raise ValueError("non-finite rotor state")
                shaft_power += abs(torque * speed)
            if shaft_power <= 0.0:
                raise ValueError("non-positive shaft power while airborne")
            battery_power = shaft_power / self.motor_esc_efficiency + self.auxiliary_power_w
            return ElectricalPowerSample(
                float(battery_power),
                float(shaft_power),
                "airsim_rotor",
            )
        except Exception:
            return self._fallback_sample(v_airspeed)

    def _calibrated_forward_sample(
        self, v_airspeed: Sequence[float]
    ) -> ElectricalPowerSample:
        physical_power = self.counterfactual_model.power(v_airspeed)
        physical_hover = max(
            self.counterfactual_model.power(np.zeros(3, dtype=np.float32)),
            1e-6,
        )
        if self.hover_battery_power_w is None:
            hover_sample = self._sample_airsim_rotor(
                np.zeros(3, dtype=np.float32)
            )
            self.hover_battery_power_w = hover_sample.battery_power_w

        calibrated_propulsion_w = max(
            float(self.hover_battery_power_w) - self.auxiliary_power_w,
            0.0,
        ) * physical_power / physical_hover
        battery_power_w = calibrated_propulsion_w + self.auxiliary_power_w
        shaft_power_w = calibrated_propulsion_w * self.motor_esc_efficiency
        return ElectricalPowerSample(
            float(battery_power_w),
            float(shaft_power_w),
            self.HOVER_CALIBRATED_FORWARD,
        )

    def sample(self, v_airspeed: Sequence[float]) -> ElectricalPowerSample:
        if self.power_model_mode == self.HOVER_CALIBRATED_FORWARD:
            return self._calibrated_forward_sample(v_airspeed)
        return self._sample_airsim_rotor(v_airspeed)

    def reset(self, v_airspeed: Sequence[float]) -> ElectricalPowerSample:
        self.total_energy_j = 0.0
        self.fallback_count = 0
        if self.power_model_mode == self.HOVER_CALIBRATED_FORWARD:


            if self.hover_battery_power_w is None:
                hover_sample = self._sample_airsim_rotor(
                    np.zeros(3, dtype=np.float32)
                )
                self.hover_battery_power_w = hover_sample.battery_power_w
            sample = self._calibrated_forward_sample(v_airspeed)
        else:
            hover_sample = self._sample_airsim_rotor(v_airspeed)
            if self.hover_battery_power_w is None:
                self.hover_battery_power_w = hover_sample.battery_power_w
            sample = hover_sample
        self.previous_power_w = sample.battery_power_w
        return sample

    def integrate(self, sample: ElectricalPowerSample, dt: float) -> float:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        previous = (
            sample.battery_power_w
            if self.previous_power_w is None
            else self.previous_power_w
        )
        step_energy_j = 0.5 * (previous + sample.battery_power_w) * dt
        self.previous_power_w = sample.battery_power_w
        self.total_energy_j += step_energy_j
        return float(step_energy_j)

    def no_wind_battery_power(self, v_airspeed_command: Sequence[float]) -> float:
        physical_power = self.counterfactual_model.power(v_airspeed_command)
        physical_hover = max(
            self.counterfactual_model.power(np.zeros(3, dtype=np.float32)),
            1e-6,
        )
        if self.hover_battery_power_w is None:
            return (
                physical_power / self.motor_esc_efficiency
                + self.auxiliary_power_w
            )
        calibrated_propulsion = max(
            self.hover_battery_power_w - self.auxiliary_power_w,
            0.0,
        ) * physical_power / physical_hover
        return float(calibrated_propulsion + self.auxiliary_power_w)
