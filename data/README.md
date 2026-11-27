# Main experiment summary

The [summary table](main/summary.csv) contains results for DDPG, PPO, SAC, TD3, PINE-SAC, and PINE-TD3, evaluated on 200 matched test scenarios per method with base test seed 46.

Success rates use all 200 episodes. Other means and population standard deviations use each method's successful episodes. Columns ending in `_mean` and `_std` contain the mean and population standard deviation, respectively.

| Fields | Meaning and units |
| --- | --- |
| `algorithm` | Method |
| `episodes`, `successes`, `success_rate_pct` | Test count, successful count, and success rate (%) |
| `reward_*` | Episode return |
| `steps_*` | Number of control steps |
| `energy_j_*` | Model-based episode electrical energy (J) |
| `ground_speed_mps_*` | Episode mean ground speed (m/s) |
| `energy_per_step_j_*` | Episode energy per control step (J/step) |
| `goal_progress_per_step_m_*` | Net goal progress per control step (m/step) |
