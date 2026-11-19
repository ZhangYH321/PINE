# PINE

PINE (Physics-Informed Navigation for Energy-Efficient Quadrotor Autonomy) studies wind-aware quadrotor navigation through structured observations, gated speed/yaw residuals, and spatial–physical value fusion.

This source-only release contains the PINE-TD3 and PINE-SAC policy components (`scripts/utils/`), a standalone correlated wind and sensor model (`pine/wind.py`), and a saved-episode statistics tool (`scripts/report_saved_results.py`).

Install the standalone module dependencies with `python -m pip install -r requirements.txt`. For locally generated episode metrics, run `python -m scripts.report_saved_results --input-dir /path/to/results`. The reporter reads an existing `episode_metrics.csv`; no results are bundled. Compared methods must use the same matched scenario indices.

The current training/environment pipeline and experiment configuration were withheld because the available source differs from the manuscript in several settings and evaluation details. Consequently, this is a partial method-code release, not a complete reproduction package. Constructor defaults are generic library defaults, not the paper's experiment preset. Datasets, logs, checkpoints, traditional planners, ablation experiments, and reviewer-response experiments are excluded.
