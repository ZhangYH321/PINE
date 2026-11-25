from pine.wind import CorrelatedRandomWindField


def _getfloat(cfg, option: str, fallback: float) -> float:
    return cfg.getfloat("wind", option, fallback=fallback)


def _getint(cfg, option: str, fallback: int) -> int:
    return cfg.getint("wind", option, fallback=fallback)


def _getbool(cfg, option: str, fallback: bool) -> bool:
    return cfg.getboolean("wind", option, fallback=fallback)


def build_wind_field(cfg, map_range_xy, z_range, dynamic_model=None, dt=0.1):


    model = cfg.get("wind", "model", fallback="correlated").strip().lower()
    if model != "correlated":
        raise ValueError("This release includes only wind.model=correlated.")
    use_turbulence = _getbool(cfg, "use_turbulence", True)
    common = {
        "dynamic_model": dynamic_model,
        "dt": dt,
        "field_seed": _getint(cfg, "field_seed", _getint(cfg, "seed", 42)),
        "turbulence_seed": _getint(cfg, "turbulence_seed", 200),
        "sensor_seed": _getint(cfg, "sensor_seed", 300),
        "sensor_noise_std": _getfloat(cfg, "sensor_noise_std", 0.1),
        "sensor_bias_std": _getfloat(cfg, "sensor_bias_std", 0.05),
        "sensor_time_constant": _getfloat(cfg, "sensor_time_constant", 0.2),
    }
    return CorrelatedRandomWindField(
        map_range_xy=map_range_xy,
        z_range=z_range,
        grid_shape=(
            _getint(cfg, "grid_nx", 20),
            _getint(cfg, "grid_ny", 20),
            _getint(cfg, "grid_nz", 8),
        ),
        speed_min=_getfloat(cfg, "speed_min", 2.0),
        speed_max=_getfloat(cfg, "speed_max", 4.0),
        mean_profile=cfg.get("wind", "mean_profile", fallback="power_law"),
        reference_height=_getfloat(cfg, "reference_height", 10.0),
        profile_exponent=_getfloat(cfg, "profile_exponent", 0.25),
        horizontal_length_scale=_getfloat(cfg, "horizontal_length_scale", 30.0),
        vertical_length_scale=_getfloat(cfg, "vertical_length_scale", 8.0),
        spatial_std_horizontal=_getfloat(cfg, "spatial_std_horizontal", 0.8),
        spatial_std_vertical=_getfloat(cfg, "spatial_std_vertical", 0.3),
        turbulence_model=cfg.get("wind", "turbulence_model", fallback="von_karman"),
        sigma_n=_getfloat(cfg, "sigma_n", 0.5),
        sigma_e=_getfloat(cfg, "sigma_e", 0.5),
        sigma_d=_getfloat(cfg, "sigma_d", 0.2),
        turbulence_time_scale=_getfloat(cfg, "turbulence_time_scale", 0.7),
        turbulence_sequence_length=_getint(cfg, "turbulence_sequence_length", 4096),
        use_turbulence=use_turbulence,
        **common,
    )
