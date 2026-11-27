# Main experiment data

Recorded main test-set results for DDPG, PPO, SAC, TD3, PINE-SAC, and PINE-TD3: 200 matched scenarios per method, 1,200 episode records in total. Scenario indices are 0–199 and the base test seed is 46.

- `main/episode_metrics.csv`: recorded episode values, with scenario identifiers and random seeds.
- `main/summary.csv`: per-method summaries recomputed from the included records. Success rates use all 200 episodes; other means and population standard deviations use each method's successful episodes. Common-success intersections contain 96 episodes for SAC/PINE-SAC and 106 for TD3/PINE-TD3.

| Fields | Meaning and units |
| --- | --- |
| `algorithm`, `episode_index` | Method and matched scenario identifier |
| `status`, `success` | Terminal outcome and binary success indicator |
| `reward`, `steps` | Episode return and number of control steps |
| `energy_consumption_j` | Model-based episode electrical energy (J) |
| `speed_mps` | Episode mean ground speed (m/s) |
| `energy_consumption_per_step_j` | Episode energy divided by control steps (J/step) |
| `path_distance_per_step_m`, `goal_progress_per_step_m` | Travel distance and net goal progress per control step (m/step) |
| `goal_x`, `goal_y`, `goal_z` | Goal coordinates (m) |
| `goal_seed`, `field_seed`, `turbulence_seed`, `sensor_seed` | Scenario random seeds |

From the repository root, summarize the recorded episodes with:

```bash
python -m scripts.report_saved_results --input-dir data/main
```
