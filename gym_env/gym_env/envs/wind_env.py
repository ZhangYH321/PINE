from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class GridWindConfig:
    """Configuration for a 2D grid wind field."""

    map_range_xy: Sequence[float]
    m: int
    n: int
    wind_min: float = 0.0
    wind_max: float = 4.0
    z_range: Sequence[float] = (0.0, 20.0)
    z_layers: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    turbulence: bool = False
    turbulence_intensity: float = 1.2
    turbulence_persistence: float = 0.4


class GridWindField:
    """Discrete wind field used by the AirSim navigation environment."""

    def __init__(
        self,
        map_range_xy,
        M,
        N,
        z_range=(0, 20),
        z_layers=None,
        wind_min=0,
        wind_max=4,
        dynamic_model=None,
        seed=None,
        turbulence=False,
    ) -> None:
        self.min_x, self.max_x, self.min_y, self.max_y = map_range_xy
        self.min_z, self.max_z = z_range
        self.z_layers = list(z_layers) if z_layers is not None else [self.min_z, self.max_z]
        self.z_num = max(1, len(self.z_layers) - 1)
        self.use_z_layers = z_layers is not None

        self.dynamic_model = dynamic_model
        self.M = int(M)
        self.N = int(N)
        self.step_x = (self.max_x - self.min_x) / self.M
        self.step_y = (self.max_y - self.min_y) / self.N

        self.wind_min = float(wind_min)
        self.wind_max = float(wind_max)
        self.turbulence = bool(turbulence)
        self.turbulence_intensity = 1.2
        self.turbulence_persistence = 0.4
        self.last_turbulence = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        if seed is not None:
            np.random.seed(int(seed))
        else:
            np.random.seed(42)

        self._init_random_wind()

    def _init_random_wind(self) -> None:
        self.wind_data = np.zeros((self.z_num, self.M, self.N, 3), dtype=np.float32)
        for z_idx in range(self.z_num):
            for x_idx in range(self.M):
                for y_idx in range(self.N):
                    wind_speed = np.random.uniform(self.wind_min, self.wind_max)
                    wind_dir = np.random.uniform(0.0, 2.0 * np.pi)
                    self.wind_data[z_idx, x_idx, y_idx] = [
                        wind_speed * np.cos(wind_dir),
                        wind_speed * np.sin(wind_dir),
                        0.0,
                    ]

    def get_turbulence(self) -> np.ndarray:
        noise = np.random.normal(0.0, self.turbulence_intensity, 3)
        self.last_turbulence = (
            self.turbulence_persistence * self.last_turbulence
            + (1.0 - self.turbulence_persistence) * noise
        )
        return self.last_turbulence

    def get_grid_index(self, pos):
        x, y, z = pos
        x_idx = max(0, min(self.M - 1, int((x - self.min_x) / self.step_x)))
        y_idx = max(0, min(self.N - 1, int((y - self.min_y) / self.step_y)))

        z_idx = 0
        if self.use_z_layers:
            for idx in range(self.z_num):
                if self.z_layers[idx] <= z < self.z_layers[idx + 1]:
                    z_idx = idx
                    break
        z_idx = max(0, min(self.z_num - 1, z_idx))
        return z_idx, x_idx, y_idx

    def get_wind_at_pos(self, pos):
        z_idx, x_idx, y_idx = self.get_grid_index(pos)
        return self.wind_data[z_idx, x_idx, y_idx]

    def update_grid_wind(self, z_idx, x_idx, y_idx, new_wind):
        if 0 <= z_idx < self.z_num and 0 <= x_idx < self.M and 0 <= y_idx < self.N:
            self.wind_data[z_idx, x_idx, y_idx] = new_wind

    def update_wind_by_pos(self, pos):
        if self.dynamic_model is None:
            return
        wind_n, wind_e, wind_d = self.get_wind_at_pos(pos)
        if not self.turbulence:
            self.dynamic_model.set_wind_vec(wind_n, wind_e, wind_d)
            return
        turbulence = self.get_turbulence()
        self.dynamic_model.set_wind_vec(
            wind_n + turbulence[0],
            wind_e + turbulence[1],
            wind_d + turbulence[2],
        )

