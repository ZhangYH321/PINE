from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RotorcraftParams:
    """Physical parameters used by the multirotor power estimator."""

    rho: float = 1.225
    g: float = 9.81

    mass: float = 1.5
    n_rotors: int = 4
    rotor_radius: float = 0.12
    rotor_speed: float = 520.0
    solidity: float = 0.08
    profile_drag_coeff: float = 0.012
    body_drag_area: float = 0.03
    climb_power_decay: float = 0.4

    induced_loss_factor: float = 1.15
    penalize_climb_only: bool = True


class MultirotorPowerModel:
    """Estimate induced, profile, and parasitic power for a multirotor."""

    def __init__(self, params: RotorcraftParams) -> None:
        self.params = params
        self.rotor_disk_area = math.pi * params.rotor_radius * params.rotor_radius
        self.total_disk_area = params.n_rotors * self.rotor_disk_area
        self.rotor_tip_speed = params.rotor_speed * params.rotor_radius

    def power_components(self, v_airspeed: Sequence[float]) -> tuple[float, float, float, dict[str, float]]:
        """Return `(induced, profile, parasitic, debug)` for the supplied airspeed."""

        params = self.params
        va_n, va_e, va_d = np.asarray(v_airspeed, dtype=float)

        horizontal_speed = math.hypot(va_n, va_e)
        vertical_speed_up = -va_d

        drag_force = 0.5 * params.rho * params.body_drag_area * (horizontal_speed**2)
        parasitic_power = 0.5 * params.rho * params.body_drag_area * (horizontal_speed**3)

        thrust = math.sqrt((params.mass * params.g) ** 2 + drag_force**2)
        induced_velocity = math.sqrt(thrust / (2.0 * params.rho * self.total_disk_area))
        induced_power = params.induced_loss_factor * thrust * induced_velocity

        if params.penalize_climb_only:
            climb_speed = max(0.0, vertical_speed_up)
        else:
            climb_speed = abs(vertical_speed_up)
        climb_power = params.climb_power_decay * thrust * climb_speed
        induced_power += climb_power

        advance_ratio = 0.0 if self.rotor_tip_speed < 1e-6 else horizontal_speed / self.rotor_tip_speed
        profile_power_single = (
            (params.solidity * params.profile_drag_coeff / 8.0)
            * params.rho
            * self.rotor_disk_area
            * (self.rotor_tip_speed**3)
            * (1.0 + 8.0 * advance_ratio * advance_ratio)
        )
        profile_power = params.n_rotors * profile_power_single

        debug = {
            "Vh": float(horizontal_speed),
            "Vz_up": float(vertical_speed_up),
            "D": float(drag_force),
            "T": float(thrust),
            "vi0": float(induced_velocity),
            "mu": float(advance_ratio),
            "Pi_induced": float(induced_power - climb_power),
            "Pi_climb": float(climb_power),
        }
        return float(induced_power), float(profile_power), float(parasitic_power), debug

