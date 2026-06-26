# PINE: Physics-Informed TD3 for UAV Navigation

This repository contains a compact PINE training framework for AirSim UAV navigation. The current mainline uses TD3 with a physics-guided value critic and wind-aware replay weighting.

## Project Layout

```text
configs/                         Training configuration files
gym_env/                         Custom Gym environment package
  gym_env/envs/airsim_env.py      AirSim navigation environment
  gym_env/envs/wind_env.py        Grid wind field
  gym_env/envs/power_model.py     Multirotor power estimator
  gym_env/envs/dynamics/          Vehicle dynamics models
scripts/
  train.py                        Headless training entry point
  evaluation.py                   Headless evaluation entry point
  start_train_with_plot.py        PyQt training UI entry point
  start_evaluate_with_plot.py     PyQt evaluation UI entry point
  core/                           Shared training/evaluation runtime
  utils/pine_td3.py               PINE-TD3 implementation
airsim_settings/                  AirSim settings examples
```

## Algorithm

PINE extends TD3 with two additions:

1. `PineReplayBuffer` stores `pgvl_rho` from environment `info`.
2. `PINE_TD3.train()` applies `rho_t` to the TD loss, so headwind or physics-relevant samples can receive higher critic weight.
3. `PhysicsGuidedCritic` can split features into spatial and physical streams, then combine both Q estimates with `omega_spatial` and `omega_physical`.

The default config is `configs/config_NH_center_SimpleMultirotor_3D.ini`.

## Setup

Install the local environment package and Python dependencies:

```bash
pip install -e gym_env
pip install stable-baselines3 wandb torch opencv-python PyQt5 pyqtgraph pillow seaborn keyboard
```

AirSim must be running with a compatible multirotor scene before training or evaluation.

## Training

Run headless training:

```bash
python scripts/train.py --config config_NH_center_SimpleMultirotor_3D
```

Disable Weights & Biases for a local smoke run:

```bash
python scripts/train.py --config config_NH_center_SimpleMultirotor_3D --no-wandb
```

Run with the PyQt training UI:

```bash
python scripts/start_train_with_plot.py --config config_NH_center_SimpleMultirotor_3D
```

Training outputs are written under:

```text
logs/<env_name>/<timestamp>_<dynamic>_<policy>_<algo>/
```

Each run contains `config/`, `models/`, `tb_logs/`, and `data/`.

## Evaluation

Evaluate a saved run:

```bash
python scripts/evaluation.py --run-dir logs/NH_center/<run_folder> --episodes 50
```

By default, evaluation loads:

```text
<run-dir>/config/config.ini
<run-dir>/models/model_sb3.zip
```

You can override either path:

```bash
python scripts/evaluation.py --run-dir logs/NH_center/<run_folder> --model models/best_model/best_model.zip
```

Run evaluation with the PyQt UI:

```bash
python scripts/start_evaluate_with_plot.py --run-dir logs/NH_center/<run_folder> --episodes 50
```

## Important Config Fields

`[options]`

- `algo = TD3`: current supported mainline.
- `policy_name = mlp`: current supported policy path.
- `perception = vector`: vectorized depth features plus state features.
- `use_wind = True`: enables wind-aware physical features and PGVL weighting.
- `reward_type = reward_wind`: uses mission and energy-aware reward.

`[PGVL]`

- `critic_mode`: `dual_stream` or `single_stream`.
- `eta_rho`: scales the headwind-based replay weight.
- `physical_feature_dim`: number of physical features at the end of the observation vector.
- `omega_spatial`, `omega_physical`: dual-stream critic mixing weights.

## Notes

- Generated caches such as `__pycache__` are intentionally not part of the source tree.
- Older hard-coded evaluation scripts and one-off wind demos were removed. Use the parameterized CLI entry points above.
- The refactor keeps backward-compatible imports for `TrainingThread` and `EvaluateThread` under `scripts/utils/`.

