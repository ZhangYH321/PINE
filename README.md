# PINE

Code for our paper, *PINE: A Physics-Informed Navigation Framework for Energy-Efficient Quadrotor Autonomy*.

PINE combines wind-aware observations, gated action residuals, and spatial–physical value fusion for energy-efficient quadrotor navigation. We implement the framework with TD3 and SAC in AirSim.

## How to run

Use Python 3.8 and install the dependencies:

```bash
git clone https://github.com/ZhangYH321/PINE.git
cd PINE
python -m pip install -r requirements.txt
python -m pip install -r requirements-airsim.txt
```

Copy `airsim_settings/settings_multirotor.json` to `%USERPROFILE%\Documents\AirSim\settings.json`, then start the NH_center scene in AirSim.

Train PINE-TD3 from the repository root:

```bash
python scripts/train.py
```

For PINE-SAC, copy `configs/config_NH_center_SimpleMultirotor_3D.ini` to `configs/pine_sac.ini`, set `algo = SAC`, and run:

```bash
python scripts/train.py --config configs/pine_sac.ini
```
