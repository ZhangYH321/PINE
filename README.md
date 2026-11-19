# PINE

PINE (Physics-Informed Navigation for Energy-Efficient Quadrotor Autonomy) combines wind-aware observations, gated speed/yaw residuals, and spatial–physical value fusion. This repository includes PINE-TD3/PINE-SAC, the main AirSim navigation environment, correlated wind and sensor models, energy estimation, a training entrypoint, and the main experiment configuration.

Use Python 3.8 in a dedicated environment. Install dependencies in this order because the AirSim package imports its RPC dependencies during installation:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-airsim.txt
```

Training requires a running AirSim simulator with the NH_center scene; the scene executable is not included. Before launching it, back up your existing AirSim settings and place `airsim_settings/settings_multirotor.json` at `~/Documents/AirSim/settings.json` (on Windows, `%USERPROFILE%\Documents\AirSim\settings.json`). Then run from the repository root:

```bash
python scripts/train.py --no-wandb
```

The default configuration is `configs/config_NH_center_SimpleMultirotor_3D.ini`; despite its historical filename, it uses the AirSim **Multirotor** backend, 3D vector observations, one goal per episode, and `reward_final`. To train PINE-SAC, copy this configuration, set `algo = SAC`, keep `model_variant = pine`, and pass its path with `--config`. Outputs are written to `logs/`. Optional W&B tracking requires a separate `wandb` installation and explicit opt-in in the configuration.

The released implementation uses TD3 exploration noise linearly decreasing from 0.15 to 0.05. Checkpoints are selected using 50 deterministic validation episodes every 40,000 training steps (seed 1000003), ordered by success rate, successful-episode energy, and successful-episode steps. Validation temporarily reuses the single simulator vehicle between training segments. Terminal rewards replace the dense reward. Energy is model-based, with a fixed 100 W battery-side hover reference and 0.05 m² drag area. These describe the code; manuscript descriptions must be reconciled with them.

This release corrects terminal classification so collisions/workspace violations cannot also count as successes. Historical paper results have not been regenerated. Software checks exercised TD3 and SAC updates, checkpoint selection, and save/reload through a local RPC fixture; this was **not** a real AirSim physics run or a reproduction of paper results.

`python -m scripts.report_saved_results --input-dir /path/to/results` summarizes your own matched episode CSV files. No datasets, recorded results, logs, weights, traditional planners, ablation drivers, or reviewer-response experiments are bundled. Only the main single-goal environment path is supported.
