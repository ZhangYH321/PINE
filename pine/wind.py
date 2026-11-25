from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def _as_float_vector(value: Sequence[float], size: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != size:
        raise ValueError(f"Expected a vector with {size} values, got {vector.size}.")
    return vector


class WindField:


    model_name = "base"

    def __init__(
        self,
        dynamic_model=None,
        dt: float = 0.1,
        field_seed: int = 42,
        turbulence_seed: int = 43,
        sensor_seed: int = 44,
        use_turbulence: bool = True,
        sensor_noise_std: float = 0.0,
        sensor_bias_std: float = 0.0,
        sensor_time_constant: float = 0.0,
    ) -> None:
        self.dynamic_model = dynamic_model
        self.dt = float(dt)
        self.field_seed = int(field_seed)
        self.turbulence_seed = int(turbulence_seed)
        self.sensor_seed = int(sensor_seed)
        self.use_turbulence = bool(use_turbulence)
        self.sensor_noise_std = float(sensor_noise_std)
        self.sensor_bias_std = float(sensor_bias_std)
        self.sensor_time_constant = max(0.0, float(sensor_time_constant))
        self.field_rng = np.random.default_rng(self.field_seed)
        self.turbulence_rng = np.random.default_rng(self.turbulence_seed)
        self.sensor_rng = np.random.default_rng(self.sensor_seed)
        self.current_field_seed = self.field_seed
        self.current_turbulence_seed = self.turbulence_seed
        self.current_sensor_seed = self.sensor_seed
        self.sensor_bias = np.zeros(3, dtype=np.float32)
        self.sensor_state = np.zeros(3, dtype=np.float32)
        self._sensor_initialized = False
        self.true_wind = np.zeros(3, dtype=np.float32)
        self.observed_wind = np.zeros(3, dtype=np.float32)
        self.time_s = 0.0
        self._last_sample_time: Optional[float] = None
        self._last_sample_position: Optional[np.ndarray] = None
        self.episode_seed = self.field_seed
        self.metadata: dict[str, Any] = {"model": self.model_name}

    def _reset_rngs(
        self,
        episode_seed: int,
        field_seed: Optional[int] = None,
        turbulence_seed: Optional[int] = None,
        sensor_seed: Optional[int] = None,
    ) -> None:


        field_sequence = (
            np.random.SeedSequence([int(episode_seed), self.field_seed, 0])
            if field_seed is None
            else np.random.SeedSequence(int(field_seed))
        )
        turbulence_sequence = (
            np.random.SeedSequence([int(episode_seed), self.turbulence_seed, 1])
            if turbulence_seed is None
            else np.random.SeedSequence(int(turbulence_seed))
        )
        sensor_sequence = (
            np.random.SeedSequence([int(episode_seed), self.sensor_seed, 2])
            if sensor_seed is None
            else np.random.SeedSequence(int(sensor_seed))
        )
        self.field_rng = np.random.default_rng(field_sequence)
        self.turbulence_rng = np.random.default_rng(turbulence_sequence)
        self.sensor_rng = np.random.default_rng(sensor_sequence)
        self.current_field_seed = (
            int(field_seed)
            if field_seed is not None
            else int(field_sequence.generate_state(1, dtype=np.uint32)[0])
        )
        self.current_turbulence_seed = (
            int(turbulence_seed)
            if turbulence_seed is not None
            else int(turbulence_sequence.generate_state(1, dtype=np.uint32)[0])
        )
        self.current_sensor_seed = (
            int(sensor_seed)
            if sensor_seed is not None
            else int(sensor_sequence.generate_state(1, dtype=np.uint32)[0])
        )

    def reset_episode(
        self,
        episode_seed: Optional[int] = None,
        field_seed: Optional[int] = None,
        turbulence_seed: Optional[int] = None,
        sensor_seed: Optional[int] = None,
    ) -> None:
        self.episode_seed = self.field_seed if episode_seed is None else int(episode_seed)
        self._reset_rngs(
            self.episode_seed,
            field_seed=field_seed,
            turbulence_seed=turbulence_seed,
            sensor_seed=sensor_seed,
        )
        self.sensor_bias = self.sensor_rng.normal(0.0, self.sensor_bias_std, 3).astype(np.float32)
        self.sensor_state = np.zeros(3, dtype=np.float32)
        self._sensor_initialized = False
        self.true_wind = np.zeros(3, dtype=np.float32)
        self.observed_wind = np.zeros(3, dtype=np.float32)
        self.time_s = 0.0
        self._last_sample_time = None
        self._last_sample_position = None

    def reset(
        self,
        field_seed: Optional[int] = None,
        turbulence_seed: Optional[int] = None,
        sensor_seed: Optional[int] = None,
    ) -> None:


        episode_seed = self.field_seed if field_seed is None else int(field_seed)
        self.reset_episode(
            episode_seed=episode_seed,
            field_seed=field_seed,
            turbulence_seed=turbulence_seed,
            sensor_seed=sensor_seed,
        )

    def sample_true(self, position: Sequence[float], time_s: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def sample_base(self, position: Sequence[float], time_s: Optional[float] = None) -> np.ndarray:


        return self.sample_true(position, time_s)

    def horizontal_slice(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        height: float,
        time_s: float = 0.0,
    ) -> np.ndarray:
        return np.asarray(
            [
                [self.sample_base([x, y, height], time_s) for y in y_values]
                for x in x_values
            ],
            dtype=np.float32,
        )

    def observe(self, true_wind: Sequence[float]) -> np.ndarray:
        true_wind = _as_float_vector(true_wind)
        noise = self.sensor_rng.normal(0.0, self.sensor_noise_std, 3).astype(np.float32)
        raw = true_wind + self.sensor_bias + noise
        if self.sensor_time_constant <= 1e-9:
            observed = raw
        else:
            alpha = float(np.exp(-self.dt / self.sensor_time_constant))
            if not self._sensor_initialized:
                self.sensor_state = raw.astype(np.float32, copy=True)
                self._sensor_initialized = True
            else:
                self.sensor_state = alpha * self.sensor_state + (1.0 - alpha) * raw
            observed = self.sensor_state
        return np.asarray(observed, dtype=np.float32)

    def update_wind_by_pos(self, position: Sequence[float], time_s: Optional[float] = None):
        if time_s is not None:
            self.time_s = float(time_s)
        position_array = _as_float_vector(position)
        reuse_sample = (
            self._last_sample_time is not None
            and abs(self.time_s - self._last_sample_time) <= 1e-12
            and self._last_sample_position is not None
            and np.allclose(position_array, self._last_sample_position, atol=1e-9, rtol=0.0)
        )
        if not reuse_sample:
            true_wind = self.sample_true(position_array, self.time_s)
            observed_wind = self.observe(true_wind)
            self.true_wind = np.asarray(true_wind, dtype=np.float32)
            self.observed_wind = np.asarray(observed_wind, dtype=np.float32)
            self._last_sample_time = self.time_s
            self._last_sample_position = position_array.copy()
            if self.dynamic_model is not None and hasattr(self.dynamic_model, "set_wind_vec"):
                self.dynamic_model.set_wind_vec(
                    *self.true_wind,
                    observed_wind=self.observed_wind,
                )
        return self.true_wind.copy(), self.observed_wind.copy()

    def get_wind_at_pos(self, position: Sequence[float]) -> np.ndarray:
        return self.sample_true(position, self.time_s)

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata["stream_episode_seeds"] = {
            "field": self.current_field_seed,
            "turbulence": self.current_turbulence_seed,
            "sensor": self.current_sensor_seed,
        }
        return metadata


class CorrelatedRandomWindField(WindField):


    model_name = "correlated"

    def __init__(
        self,
        map_range_xy,
        z_range,
        dynamic_model=None,
        dt=0.1,
        grid_shape=(20, 20, 8),
        speed_min=2.0,
        speed_max=4.0,
        mean_profile="power_law",
        reference_height=10.0,
        profile_exponent=0.25,
        mean_direction=None,
        horizontal_length_scale=30.0,
        vertical_length_scale=8.0,
        spatial_std_horizontal=0.8,
        spatial_std_vertical=0.3,
        turbulence_model="von_karman",
        sigma_n=0.5,
        sigma_e=0.5,
        sigma_d=0.2,
        turbulence_time_scale=0.7,
        turbulence_sequence_length=4096,
        use_turbulence=True,
        field_seed=100,
        turbulence_seed=200,
        sensor_seed=300,
        sensor_noise_std=0.1,
        sensor_bias_std=0.05,
        sensor_time_constant=0.2,
    ) -> None:
        super().__init__(
            dynamic_model=dynamic_model,
            dt=dt,
            field_seed=field_seed,
            turbulence_seed=turbulence_seed,
            sensor_seed=sensor_seed,
            use_turbulence=use_turbulence,
            sensor_noise_std=sensor_noise_std,
            sensor_bias_std=sensor_bias_std,
            sensor_time_constant=sensor_time_constant,
        )
        self.min_x, self.max_x, self.min_y, self.max_y = map(float, map_range_xy)
        self.min_z, self.max_z = map(float, z_range)
        self.nx, self.ny, self.nz = [max(2, int(value)) for value in grid_shape]
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        if self.speed_max < self.speed_min:
            raise ValueError("speed_max must be >= speed_min")
        self.mean_profile = str(mean_profile).lower()
        self.reference_height = max(float(reference_height), 1e-3)
        self.profile_exponent = float(profile_exponent)
        self.mean_direction = None if mean_direction is None else float(mean_direction)
        self.horizontal_length_scale = max(float(horizontal_length_scale), 1e-3)
        self.vertical_length_scale = max(float(vertical_length_scale), 1e-3)
        self.spatial_std_horizontal = max(float(spatial_std_horizontal), 0.0)
        self.spatial_std_vertical = max(float(spatial_std_vertical), 0.0)
        self.turbulence_model = str(turbulence_model).lower()
        self.sigma = np.asarray([sigma_n, sigma_e, sigma_d], dtype=np.float32)
        self.wind_min = self.speed_min
        self.wind_max = self.speed_max + 3.0 * max(
            self.spatial_std_horizontal,
            self.spatial_std_vertical,
            float(np.max(self.sigma)),
        )
        self.turbulence_time_scale = max(float(turbulence_time_scale), 1e-3)
        self.turbulence_sequence_length = max(256, int(turbulence_sequence_length))
        self.mean_speed = self.speed_min
        self.mean_direction_current = 0.0
        self.spatial_field = np.zeros((3, self.nz, self.nx, self.ny), dtype=np.float32)
        self.turbulence_sequence = np.zeros(
            (self.turbulence_sequence_length, 3),
            dtype=np.float32,
        )
        self.x_grid = np.linspace(self.min_x, self.max_x, self.nx)
        self.y_grid = np.linspace(self.min_y, self.max_y, self.ny)
        self.z_grid = np.linspace(self.min_z, self.max_z, self.nz)
        self.metadata.update(
            {
                "grid_shape": [self.nz, self.nx, self.ny],
                "horizontal_length_scale": self.horizontal_length_scale,
                "vertical_length_scale": self.vertical_length_scale,
                "turbulence_model": self.turbulence_model,
            }
        )
        self.reset_episode(self.field_seed)

    def reset_episode(
        self,
        episode_seed: Optional[int] = None,
        field_seed: Optional[int] = None,
        turbulence_seed: Optional[int] = None,
        sensor_seed: Optional[int] = None,
    ) -> None:
        super().reset_episode(
            episode_seed,
            field_seed=field_seed,
            turbulence_seed=turbulence_seed,
            sensor_seed=sensor_seed,
        )
        self.mean_speed = float(self.field_rng.uniform(self.speed_min, self.speed_max))
        if self.mean_direction is None:
            self.mean_direction_current = float(self.field_rng.uniform(-np.pi, np.pi))
        else:
            self.mean_direction_current = self.mean_direction
        self.spatial_field = self._generate_spatial_field()
        self.turbulence_sequence = self._generate_turbulence_sequence()
        self.metadata.update(
            {
                "episode_seed": int(self.episode_seed),
                "mean_speed": self.mean_speed,
                "mean_direction_rad": self.mean_direction_current,
            }
        )

    def _generate_spatial_field(self) -> np.ndarray:
        shape = (self.nz, self.nx, self.ny)
        noise = self.field_rng.normal(0.0, 1.0, (3,) + shape)
        spectrum = np.fft.fftn(noise, axes=(1, 2, 3))
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=max(self.z_grid[1] - self.z_grid[0], 1e-6))
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=max(self.x_grid[1] - self.x_grid[0], 1e-6))
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=max(self.y_grid[1] - self.y_grid[0], 1e-6))
        kz3 = kz[:, None, None]
        kx3 = kx[None, :, None]
        ky3 = ky[None, None, :]
        filter_gain = np.exp(
            -0.5
            * (
                (kx3 * self.horizontal_length_scale) ** 2
                + (ky3 * self.horizontal_length_scale) ** 2
                + (kz3 * self.vertical_length_scale) ** 2
            )
        )
        spectrum *= filter_gain[None, ...]


        k_squared = kx3 * kx3 + ky3 * ky3 + kz3 * kz3
        dot = kx3 * spectrum[0] + ky3 * spectrum[1] + kz3 * spectrum[2]
        nonzero = k_squared > 1e-12
        for component, wave_number in enumerate((kx3, ky3, kz3)):
            projected = spectrum[component].copy()
            correction = np.zeros_like(projected)
            np.divide(
                wave_number * dot,
                k_squared,
                out=correction,
                where=nonzero,
            )
            projected -= correction
            spectrum[component] = projected

        field = np.fft.ifftn(spectrum, axes=(1, 2, 3)).real.astype(np.float32)
        targets = np.asarray(
            [self.spatial_std_horizontal, self.spatial_std_horizontal, self.spatial_std_vertical],
            dtype=np.float32,
        )
        for component in range(3):
            field[component] -= np.mean(field[component])
            current_std = float(np.std(field[component]))
            if current_std > 1e-8 and targets[component] > 0.0:
                field[component] *= targets[component] / current_std
            elif targets[component] == 0.0:
                field[component].fill(0.0)
        return field

    def _interpolate(self, position: Sequence[float]) -> np.ndarray:
        x, y, z = [float(value) for value in position]
        x = float(np.clip(x, self.min_x, self.max_x))
        y = float(np.clip(y, self.min_y, self.max_y))
        z = float(np.clip(z, self.min_z, self.max_z))
        ix = int(np.clip(np.searchsorted(self.x_grid, x) - 1, 0, self.nx - 2))
        iy = int(np.clip(np.searchsorted(self.y_grid, y) - 1, 0, self.ny - 2))
        iz = int(np.clip(np.searchsorted(self.z_grid, z) - 1, 0, self.nz - 2))
        tx = (x - self.x_grid[ix]) / max(self.x_grid[ix + 1] - self.x_grid[ix], 1e-9)
        ty = (y - self.y_grid[iy]) / max(self.y_grid[iy + 1] - self.y_grid[iy], 1e-9)
        tz = (z - self.z_grid[iz]) / max(self.z_grid[iz + 1] - self.z_grid[iz], 1e-9)
        value = np.zeros(3, dtype=np.float32)
        for dz in (0, 1):
            for dx in (0, 1):
                for dy in (0, 1):
                    weight = (
                        (tz if dz else 1.0 - tz)
                        * (tx if dx else 1.0 - tx)
                        * (ty if dy else 1.0 - ty)
                    )
                    value += float(weight) * self.spatial_field[:, iz + dz, ix + dx, iy + dy]
        return value

    def _abl_mean(self, height: float) -> np.ndarray:
        height = max(float(height), 0.5)
        if self.mean_profile in {"log", "log_law", "log-law"}:
            roughness = max(self.reference_height * 0.1, 0.1)
            denominator = np.log(max(self.reference_height / roughness, 1.01))
            scale = np.log(max(height / roughness, 1.01)) / denominator
        else:
            scale = (height / self.reference_height) ** self.profile_exponent
        speed = self.mean_speed * max(float(scale), 0.0)
        return np.asarray(
            [
                speed * np.cos(self.mean_direction_current),
                speed * np.sin(self.mean_direction_current),
                0.0,
            ],
            dtype=np.float32,
        )

    def _generate_turbulence_sequence(self) -> np.ndarray:
        if not self.use_turbulence or self.turbulence_model in {"none", "off", "false"}:
            return np.zeros((self.turbulence_sequence_length, 3), dtype=np.float32)
        count = self.turbulence_sequence_length
        frequencies = np.fft.rfftfreq(count, d=self.dt)
        omega_tau = 2.0 * np.pi * frequencies * self.turbulence_time_scale


        target_psd = (1.0 + omega_tau * omega_tau) ** (-5.0 / 6.0)
        target_psd[0] = 0.0
        sequence = np.zeros((count, 3), dtype=np.float32)
        for component in range(3):
            real = self.turbulence_rng.normal(0.0, 1.0, frequencies.size)
            imag = self.turbulence_rng.normal(0.0, 1.0, frequencies.size)
            coefficients = (real + 1j * imag) * np.sqrt(target_psd)
            coefficients[0] = 0.0
            if count % 2 == 0:
                coefficients[-1] = coefficients[-1].real
            values = np.fft.irfft(coefficients, n=count)
            values -= np.mean(values)
            current_std = float(np.std(values))
            if current_std > 1e-12 and self.sigma[component] > 0.0:
                values *= float(self.sigma[component]) / current_std
            else:
                values.fill(0.0)
            sequence[:, component] = values.astype(np.float32)
        return sequence

    def _sample_von_karman(self, time_s: Optional[float]) -> np.ndarray:
        if self.turbulence_sequence.size == 0:
            return np.zeros(3, dtype=np.float32)
        sample_time = 0.0 if time_s is None else max(float(time_s), 0.0)
        index = int(round(sample_time / max(self.dt, 1e-9))) % self.turbulence_sequence.shape[0]
        return self.turbulence_sequence[index].copy()

    def sample_base(self, position, time_s=None):
        mean = self._abl_mean(float(position[2]))
        spatial = self._interpolate(position)
        return (mean + spatial).astype(np.float32)

    def sample_true(self, position, time_s=None):
        turbulence = self._sample_von_karman(time_s)
        return (self.sample_base(position, time_s) + turbulence).astype(np.float32)
