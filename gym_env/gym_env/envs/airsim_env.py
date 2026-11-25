import gym
import json
from gym import spaces
import airsim
from configparser import NoOptionError
import keyboard
import time
import torch as th
import numpy as np
import math
import cv2
from pathlib import Path


from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .energy_model import AirSimRotorEnergyModel
from .power_model import MultirotorPowerModel, RotorcraftParams
from .wind_env import build_wind_field

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

class AirsimGymEnv(gym.Env, QtCore.QThread):


    metadata = {"render_modes": []}
    render_mode = None


    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    lgmd_signal = pyqtSignal(float, float, np.ndarray)
    wind_field_signal = pyqtSignal(object, int)


    def __init__(self) -> None:
        super().__init__()

        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("初始化AirSim-Gym环境.")


        self.model = None
        self.data_path = None
        self.lgmd = None
        self.goal_seed = None
        self.current_episode_seed = None
        self.current_goal_seed = None
        self.current_field_seed = None
        self.current_turbulence_seed = None
        self.current_sensor_seed = None
        self.base_seed = 42


        self.crash_flag = False
        self.episode_done = False
        self.episode_crash_count = 0
        self.episode_total_count = 0

    def set_config(self, cfg):


        for key, expected in (("env_name", "NH_center"), ("dynamic_name", "Multirotor"),
                              ("perception", "vector"), ("reward_type", "reward_final")):
            if cfg.get("options", key) != expected:
                raise ValueError(f"Main release requires options.{key}={expected}")
        if not cfg.getboolean("options", "navigation_3d"):
            raise ValueError("Main release requires 3D navigation.")
        if cfg.getint("mission", "waypoint_count", fallback=1) != 1:
            raise ValueError("Main release supports one goal per episode.")
        self.cfg = cfg
        self.base_seed = cfg.getint(
            'options',
            'seed',
            fallback=cfg.getint('wind', 'field_seed', fallback=42),
        )

        self.wind_apply_physics = cfg.getboolean('wind', 'apply_physics', fallback=True)
        self.log_wind_trajectory = cfg.getboolean('wind', 'log_trajectory', fallback=True)
        self.wind_reset_each_episode = cfg.getboolean('wind', 'reset_each_episode', fallback=False)
        self.wind_apply_on_reset = cfg.getboolean('wind', 'apply_on_reset', fallback=False)
        self.wind_apply_before_action = cfg.getboolean('wind', 'apply_before_action', fallback=False)

        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.perception_type = cfg.get('options', 'perception')

        print(f'环境: {self.env_name}  动力学模型: {self.dynamic_name}  感知类型: {self.perception_type}')


        self.dynamic_model = MultirotorDynamicsAirsim(cfg)


        if self.env_name == 'NH_center':
            self.mission_waypoint_bounds_xy = (-128.0, -128.0, 128.0, 128.0)
            start_position = [0, 0, 5]
            goal_rect = [-128, -128, 128, 128]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(random_angle=math.pi * 2, rect=goal_rect)

            self.work_space_x = [-140, 140]
            self.work_space_y = [-140, 140]
            self.work_space_z = [0.5, 20]
            self.max_episode_steps = 1000

            self.params = RotorcraftParams()
            self.power_model = MultirotorPowerModel(self.params)
            self.energy_model = AirSimRotorEnergyModel(
                self.dynamic_model.client,
                fallback_model=self.power_model,
                params=self.params,
                motor_esc_efficiency=cfg.getfloat(
                    'energy', 'motor_esc_efficiency', fallback=0.82
                ),
                auxiliary_power_w=cfg.getfloat(
                    'energy', 'auxiliary_power_w', fallback=10.0
                ),
                expected_rotors=cfg.getint('energy', 'expected_rotors', fallback=4),
                profile_advance_coefficient=cfg.getfloat(
                    'energy', 'profile_advance_coefficient', fallback=4.6
                ),
                power_model_mode=cfg.get(
                    'energy', 'power_model_mode', fallback='airsim_rotor'
                ),
                fixed_hover_battery_power_w=cfg.getfloat(
                    'energy', 'fixed_hover_battery_power_w', fallback=0.0
                ),
            )

            self.wind_field = build_wind_field(
                self.cfg,
                map_range_xy=self.work_space_x + self.work_space_y,
                z_range=self.work_space_z,
                dynamic_model=self.dynamic_model,
                dt=self.cfg.getfloat('multirotor', 'dt'),
            )


        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')


        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0
        self.CumulatedPower = 0
        self.CumulatedEnergy = 0
        self.current_power_w = 0.0
        self.current_shaft_power_w = 0.0
        self.current_energy_model_source = "uninitialized"
        self.net_wind_assist_energy_j = 0.0
        self.current_step_energy_j = 0.0
        self.current_wind_assist_energy_j = 0.0
        self.current_reward_base = 0.0
        self.current_reward_auxiliary = 0.0
        self.current_reward_energy_cost = 0.0
        self.current_reward_wind_savings = 0.0
        self.termination_reason = None
        self.termination_state = None
        self.mission_waypoints = np.empty((0, 3), dtype=np.float32)
        self.current_waypoint_index = 0
        self.mission_total_distance_m = 0.0
        self.waypoint_reached_this_step = False


        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')
        self.trajectory_list = []
        self.wind_episode_log = []


        if self.perception_type == 'vector' or self.perception_type == 'lgmd':

            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(1, self.cnn_feature_length + self.state_feature_length),
                dtype=np.float32
            )
        else:

            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.screen_height, self.screen_width, 2),
                dtype=np.uint8
            )


        self.action_space = self.dynamic_model.action_space
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)


        self.reward_type = None
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print(f'奖励函数类型: {self.reward_type}')
        except NoOptionError:
            self.reward_type = None


    @staticmethod
    def _seed_for_episode(base_seed, stream_id, episode_index):
        sequence = np.random.SeedSequence(
            [int(base_seed), int(stream_id), int(episode_index)]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _emit_wind_field_display(self):
        if not hasattr(self, "wind_field"):
            return
        grid_nx = max(1, self.cfg.getint('wind', 'display_grid_nx', fallback=10))
        grid_ny = max(1, self.cfg.getint('wind', 'display_grid_ny', fallback=10))
        step_x = (self.work_space_x[1] - self.work_space_x[0]) / grid_nx
        step_y = (self.work_space_y[1] - self.work_space_y[0]) / grid_ny
        x_values = np.linspace(
            self.work_space_x[0] + 0.5 * step_x,
            self.work_space_x[1] - 0.5 * step_x,
            grid_nx,
        )
        y_values = np.linspace(
            self.work_space_y[0] + 0.5 * step_y,
            self.work_space_y[1] - 0.5 * step_y,
            grid_ny,
        )
        position = self.dynamic_model.get_position()
        wind_slice = self.wind_field.horizontal_slice(
            x_values,
            y_values,
            height=float(position[2]),
            time_s=0.0,
        )
        if self.wind_field._last_sample_time is None:
            temporal_offset = np.zeros(3, dtype=np.float32)
        else:
            temporal_offset = self.wind_field.true_wind - self.wind_field.sample_base(
                position,
                time_s=0.0,
            )
        self.wind_field_signal.emit(wind_slice + temporal_offset[None, None, :], self.episode_num)


    def _get_hover_battery_power_reference(self, reward_section="reward_wind"):

        configured_reference = self.cfg.getfloat(
            reward_section, "power_reference_w", fallback=0.0
        )
        if np.isfinite(configured_reference) and configured_reference > 0.0:
            return float(configured_reference)

        energy_model = getattr(self, "energy_model", None)
        measured_reference = getattr(energy_model, "hover_battery_power_w", None)
        if (
            measured_reference is not None
            and np.isfinite(measured_reference)
            and measured_reference > 0.0
        ):
            return float(measured_reference)

        hover_components = self.power_model.power_components(
            np.zeros(3, dtype=np.float32)
        )
        hover_shaft_power = float(sum(hover_components[:3]))
        efficiency = float(
            getattr(
                energy_model,
                "motor_esc_efficiency",
                self.cfg.getfloat("energy", "motor_esc_efficiency", fallback=0.82),
            )
        )
        auxiliary_power = float(
            getattr(
                energy_model,
                "auxiliary_power_w",
                self.cfg.getfloat("energy", "auxiliary_power_w", fallback=10.0),
            )
        )
        return float(hover_shaft_power / max(efficiency, 1e-6) + auxiliary_power)


    @staticmethod
    def _compact_status_info(info):
        keys = (
            "is_success",
            "is_crash",
            "is_not_in_workspace",
            "is_step_limit",
            "termination_reason",
            "step_num",
            "episode_index",
            "SOC",
        )
        return {key: info.get(key) for key in keys}


    def set_goal_seed(self,goal_seed):
        self.goal_seed = goal_seed


    def _reset_waypoint_mission(self):
        start_position = np.asarray(self.dynamic_model.start_position, dtype=np.float32)
        self.mission_waypoints = np.asarray(
            [self.dynamic_model.goal_position], dtype=np.float32
        )

        anchors = np.vstack((start_position, self.mission_waypoints[:-1]))
        self.mission_total_distance_m = float(
            np.linalg.norm(self.mission_waypoints - anchors, axis=1).sum()
        )
        self.current_waypoint_index = 0
        self.waypoint_reached_this_step = False


    def _save_wind_episode_log(self):
        if not self.log_wind_trajectory or not self.wind_episode_log or self.data_path is None:
            return
        output_dir = Path(self.data_path) / "wind"
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = (
            self.wind_field.get_metadata()
            if hasattr(self, "wind_field")
            else {"model": "none"}
        )
        np.savez_compressed(
            output_dir / f"wind_episode_{self.episode_num:06d}.npz",
            time_s=np.asarray([row["time_s"] for row in self.wind_episode_log], dtype=np.float32),
            position=np.asarray([row["position"] for row in self.wind_episode_log], dtype=np.float32),
            true_wind=np.asarray([row["true_wind"] for row in self.wind_episode_log], dtype=np.float32),
            observed_wind=np.asarray([row["observed_wind"] for row in self.wind_episode_log], dtype=np.float32),
            commanded_airspeed=np.asarray([row["commanded_airspeed"] for row in self.wind_episode_log], dtype=np.float32),
            commanded_ground_velocity=np.asarray([row["commanded_ground_velocity"] for row in self.wind_episode_log], dtype=np.float32),
            actual_airspeed=np.asarray([row["actual_airspeed"] for row in self.wind_episode_log], dtype=np.float32),
            actual_ground_velocity=np.asarray([row["actual_ground_velocity"] for row in self.wind_episode_log], dtype=np.float32),
            battery_power_w=np.asarray([row["battery_power_w"] for row in self.wind_episode_log], dtype=np.float32),
            energy_model_source=np.asarray([row["energy_model_source"] for row in self.wind_episode_log], dtype=object),
            step_energy_j=np.asarray([row["step_energy_j"] for row in self.wind_episode_log], dtype=np.float32),
            wind_assist_energy_j=np.asarray([row["wind_assist_energy_j"] for row in self.wind_episode_log], dtype=np.float32),
            episode_index=np.asarray(self.current_episode_index, dtype=np.int64),
            goal_position=np.asarray(self.dynamic_model.goal_position, dtype=np.float32),
            mission_waypoints=np.asarray(self.mission_waypoints, dtype=np.float32),
            current_waypoint_index=np.asarray(self.current_waypoint_index, dtype=np.int64),
            mission_total_distance_m=np.asarray(self.mission_total_distance_m, dtype=np.float32),
            control_command_semantics=np.asarray(
                getattr(self.dynamic_model, "control_command_semantics", "unknown")
            ),
            episode_seed=np.asarray(self.current_episode_seed, dtype=np.uint32),
            goal_seed=np.asarray(self.current_goal_seed, dtype=np.uint32),
            field_seed=np.asarray(self.current_field_seed, dtype=np.uint32),
            turbulence_seed=np.asarray(self.current_turbulence_seed, dtype=np.uint32),
            sensor_seed=np.asarray(self.current_sensor_seed, dtype=np.uint32),
            energy_model_fallback_count=np.asarray(
                getattr(getattr(self, "energy_model", None), "fallback_count", 0),
                dtype=np.int64,
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    def seed(self, seed=None):

        if seed is not None:
            self.base_seed = int(seed)
        return [self.base_seed]

    def reset(self, *args, **kwargs):


        seed = kwargs.pop("seed", None)
        if args:
            if len(args) > 1:
                raise TypeError("reset() accepts at most one positional seed")
            seed = args[0]
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"reset() got an unexpected keyword argument {unexpected!r}")
        if seed is not None:
            self.seed(seed)
        self.current_episode_index = int(self.episode_num)
        self.current_episode_seed = self._seed_for_episode(
            self.base_seed, 0, self.current_episode_index
        )
        self.current_goal_seed = (
            int(self.goal_seed)
            if self.goal_seed is not None
            else self._seed_for_episode(self.base_seed, 1, self.current_episode_index)
        )
        self.current_field_seed = self._seed_for_episode(
            self.base_seed, 2, self.current_episode_index
        )
        self.current_turbulence_seed = self._seed_for_episode(
            self.base_seed, 3, self.current_episode_index
        )
        self.current_sensor_seed = self._seed_for_episode(
            self.base_seed, 4, self.current_episode_index
        )

        if hasattr(self.dynamic_model, "set_wind_vec"):
            self.dynamic_model.set_wind_vec(0, 0, 0)
        if self.dynamic_name == 'Multirotor':
            self.dynamic_model.reset(
                self.current_episode_seed,
                goal_seed=self.current_goal_seed,
            )
        else:
            self.dynamic_model.reset(self.current_goal_seed)
        self._reset_waypoint_mission()


        self.crash_flag = False
        self.episode_done = False
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.CumulatedPower = 0
        self.CumulatedEnergy = 0
        self.net_wind_assist_energy_j = 0.0
        self.current_step_energy_j = 0.0
        self.current_wind_assist_energy_j = 0.0
        self.current_reward_base = 0.0
        self.current_reward_auxiliary = 0.0
        self.current_reward_energy_cost = 0.0
        self.current_reward_wind_savings = 0.0
        self.termination_reason = None
        self.termination_state = None
        self.waypoint_reached_this_step = False
        self.trajectory_list = []
        self.wind_episode_log = []
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

        if hasattr(self, "energy_model"):
            initial_airspeed = self.dynamic_model.get_airspeed()
            initial_power = self.energy_model.reset(initial_airspeed)
            self.current_power_w = initial_power.battery_power_w
            self.current_shaft_power_w = initial_power.shaft_power_w
            self.current_energy_model_source = initial_power.source
            self.dynamic_model.power_observation_reference_w = (
                self._get_hover_battery_power_reference()
            )

        if hasattr(self, "wind_field") and self.wind_reset_each_episode:
            self.wind_field.reset_episode(
                episode_seed=self.current_episode_seed,
                field_seed=self.current_field_seed,
                turbulence_seed=self.current_turbulence_seed,
                sensor_seed=self.current_sensor_seed,
            )
        elif hasattr(self, "wind_field"):
            fixed_stream_seeds = self.wind_field.get_metadata().get(
                "stream_episode_seeds",
                {},
            )
            self.current_field_seed = fixed_stream_seeds.get(
                "field",
                self.current_field_seed,
            )
            self.current_turbulence_seed = fixed_stream_seeds.get(
                "turbulence",
                self.current_turbulence_seed,
            )
            self.current_sensor_seed = fixed_stream_seeds.get(
                "sensor",
                self.current_sensor_seed,
            )
        if hasattr(self, "wind_field") and self.wind_apply_on_reset:
            self.wind_field.update_wind_by_pos(
                self.dynamic_model.get_position(),
                time_s=0.0,
            )
        if hasattr(self, "wind_field"):
            self._emit_wind_field_display()


        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance


        obs = self.get_obs(self.current_power_w)

        return obs

    def step(self, action):


        self.termination_state = None
        wind_sample_position = self.dynamic_model.get_position()
        if hasattr(self, "wind_field") and self.wind_apply_before_action:
            self.wind_field.update_wind_by_pos(
                wind_sample_position,
                time_s=self.step_num * self.dynamic_model.dt,
            )


        self.dynamic_model.set_action(action)


        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        if hasattr(self, "wind_field") and not self.wind_apply_before_action:
            wind_sample_position = position_ue4
            self.wind_field.update_wind_by_pos(
                wind_sample_position,
                time_s=self.step_num * self.dynamic_model.dt,
            )

        v_airspeed = self.dynamic_model.get_airspeed() if hasattr(self.dynamic_model, "get_airspeed") else np.zeros(3)
        wind_ned = getattr(
            self.dynamic_model,
            "true_wind",
            getattr(self.dynamic_model, "wind_vec", np.zeros(3)),
        )
        observed_wind_ned = np.asarray(
            getattr(self.dynamic_model, "observed_wind", wind_ned),
            dtype=np.float32,
        )


        Pi, P0, Pp, dbg = self.power_model.power_components(v_airspeed=v_airspeed)
        if hasattr(self, "energy_model"):
            power_sample = self.energy_model.sample(v_airspeed)
            P = power_sample.battery_power_w
            self.current_shaft_power_w = power_sample.shaft_power_w
            self.current_energy_model_source = power_sample.source
            self.current_step_energy_j = self.energy_model.integrate(
                power_sample, self.dynamic_model.dt
            )
            self.CumulatedEnergy = self.energy_model.total_energy_j
        else:
            P = Pi + P0 + Pp
            self.current_shaft_power_w = P
            self.current_energy_model_source = "analytical"
            self.current_step_energy_j = P * self.dynamic_model.dt
            self.CumulatedEnergy += self.current_step_energy_j
        self.current_power_w = float(P)
        if hasattr(self.dynamic_model, "get_velocity_ground"):
            v_ground = self.dynamic_model.get_velocity_ground()
        else:
            v_ground = np.zeros(3, dtype=np.float32)
        if self.log_wind_trajectory:
            self.wind_episode_log.append(
                {
                    "time_s": float(self.step_num * self.dynamic_model.dt),
                    "position": np.asarray(wind_sample_position, dtype=np.float32),
                    "true_wind": np.asarray(wind_ned, dtype=np.float32),
                    "observed_wind": observed_wind_ned.copy(),
                    "commanded_airspeed": np.asarray(
                        getattr(self.dynamic_model, "commanded_airspeed_ned", np.zeros(3)),
                        dtype=np.float32,
                    ),
                    "commanded_ground_velocity": np.asarray(
                        getattr(self.dynamic_model, "commanded_ground_velocity_ned", np.zeros(3)),
                        dtype=np.float32,
                    ),
                    "actual_airspeed": np.asarray(v_airspeed, dtype=np.float32),
                    "actual_ground_velocity": np.asarray(v_ground, dtype=np.float32),
                    "battery_power_w": float(P),
                    "energy_model_source": self.current_energy_model_source,
                    "step_energy_j": float(self.current_step_energy_j),
                    "wind_assist_energy_j": 0.0,
                }
            )
        if hasattr(self.dynamic_model, "update_energy_j"):
            self.dynamic_model.update_energy_j(self.current_step_energy_j)
        elif hasattr(self.dynamic_model, "update_energy"):
            self.dynamic_model.update_energy(P, self.dynamic_model.dt)


        if hasattr(self, "wind_field") and self.wind_apply_before_action:
            self.wind_field.update_wind_by_pos(
                position_ue4,
                time_s=(self.step_num + 1) * self.dynamic_model.dt,
            )

        obs = self.get_obs(P)
        termination = self.get_termination_state()
        self.termination_reason = termination["reason"]
        done = self.termination_reason is not None

        info = {
            'is_success': termination["is_success"],
            'is_crash': termination["is_crash"],
            'is_not_in_workspace': termination["is_not_in_workspace"],
            'is_step_limit': termination["is_step_limit"],
            'termination_reason': self.termination_reason,
            'step_num': self.step_num + 1,
            "episode_index": self.current_episode_index,
            "SOC": getattr(self.dynamic_model, "soc", 1.0),
            "wind_true_ned": [float(value) for value in np.asarray(wind_ned).reshape(-1)],
            "wind_observed_ned": [
                float(value)
                for value in observed_wind_ned.reshape(-1)
            ],
            "wind_model": getattr(getattr(self, "wind_field", None), "model_name", "none"),
            "episode_seed": self.current_episode_seed,
            "goal_seed": self.current_goal_seed,
            "field_seed": self.current_field_seed,
            "turbulence_seed": self.current_turbulence_seed,
            "sensor_seed": self.current_sensor_seed,
            "goal_position": [float(value) for value in self.dynamic_model.goal_position],
            "mission_waypoints": self.mission_waypoints.tolist(),
            "waypoint_index": int(self.current_waypoint_index),
            "waypoint_count": int(len(self.mission_waypoints)),
            "mission_total_distance_m": float(self.mission_total_distance_m),
            "wind_position": [float(value) for value in wind_sample_position],
            "wind_time_s": float(self.step_num * self.dynamic_model.dt),
            "commanded_airspeed_ned": [
                float(value)
                for value in getattr(
                    self.dynamic_model, "commanded_airspeed_ned", np.zeros(3)
                )
            ],
            "commanded_ground_velocity_ned": [
                float(value)
                for value in getattr(
                    self.dynamic_model, "commanded_ground_velocity_ned", np.zeros(3)
                )
            ],
            "control_command_semantics": getattr(
                self.dynamic_model,
                "control_command_semantics",
                "unknown",
            ),
            "ground_velocity_ned": [float(value) for value in np.asarray(v_ground)],
            "airspeed_ned": [float(value) for value in np.asarray(v_airspeed)],
            "Instantaneous Power (W)": float(P),
            "Shaft Power (W)": float(self.current_shaft_power_w),
            "Energy Model Source": self.current_energy_model_source,
            "energy_model_source": self.current_energy_model_source,
            "Energy Model Fallback Count": int(
                getattr(getattr(self, "energy_model", None), "fallback_count", 0)
            ),
            "energy_model_fallback_count": int(
                getattr(getattr(self, "energy_model", None), "fallback_count", 0)
            ),


            "Cumulative Reward": f"{self.cumulated_episode_reward: 4.4f}",
            "Energy Consumed (J)": float(self.CumulatedEnergy),
            "Energy Consumed (Wh)": float(self.CumulatedEnergy / 3600.0),
        }


        self.crash_flag = info['is_crash']
        self.episode_done = done


        if done:
            self.episode_total_count += 1
            if self.crash_flag:
                self.episode_crash_count += 1

        reward = self.compute_reward_final(done, action)

        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        if self.wind_episode_log:
            self.wind_episode_log[-1]["wind_assist_energy_j"] = float(
                self.current_wind_assist_energy_j
            )


        self.cumulated_episode_reward += reward
        info["Cumulative Reward"] = f"{self.cumulated_episode_reward: 4.4f}"
        info["Energy Consumed (J)"] = float(self.CumulatedEnergy)
        info["Energy Consumed (Wh)"] = float(self.CumulatedEnergy / 3600.0)
        elapsed_s = max((self.step_num + 1) * self.dynamic_model.dt, self.dynamic_model.dt)
        info["Mean Power (W)"] = float(self.CumulatedEnergy / elapsed_s)
        info["episode_energy_j"] = float(self.CumulatedEnergy)
        info["mean_power_w"] = float(self.CumulatedEnergy / elapsed_s)
        info["net_wind_assist_energy_j"] = float(self.net_wind_assist_energy_j)
        info["step_wind_assist_energy_j"] = float(self.current_wind_assist_energy_j)
        if done:
            self._save_wind_episode_log()
            print(self._compact_status_info(info))


        self.print_train_info_airsim(action, obs, reward, info)


        self.set_pyqt_signal_multirotor(action, reward)


        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('=' * 100)
            print(f'回合 {self.episode_num}  步数 {self.step_num}  总步数 {self.total_step}')
            print(f'动作 {action_copy}')
            print(f'状态 {state_copy}')
            print(f'归一化状态 {self.dynamic_model.state_norm}')
            print(f'奖励 {reward:.3f}  累计奖励 {self.cumulated_episode_reward:.3f}')
            print(f'回合结束 {done}')
            keyboard.wait('a')


        if self.generate_q_map and (
                self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC'):
            if self.model is not None:
                with th.no_grad():
                    obs_copy = obs.copy()
                    if self.perception_type != 'vector':
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)

                    q_value_current = self.model.critic(
                        th.from_numpy(obs_copy[tuple([None])]).float().cuda(),
                        th.from_numpy(action[None]).float().cuda()
                    )
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]
                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)


        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info


    def get_obs(self,P=0):

        obs = self.get_obs_vector(P)

        return obs


    def get_depth_image(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        ])


        while responses[0].width == 0:
            print("获取深度图失败，重试...")
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
            )
            time.sleep(0.5)


        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                  responses[0].height)
        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_vector(self,P=0):


        image = self.get_depth_image()
        self.min_distance_to_obstacles = image.min()


        image_scaled = np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)
        split_row = 5
        split_col = 5
        v_split_list = np.vsplit(image_uint8, split_row)
        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())


        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature(predicted_power = P) / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)
        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))


        return feature_all


    def _compute_obstacle_penalty(self):

        mode = self.cfg.get("options", "obstacle_penalty_mode", fallback="legacy").strip().lower()
        distance = float(self.min_distance_to_obstacles)

        if mode in {"legacy", "linear"}:
            if distance < 10.0:
                return float(1.0 - np.clip((distance - self.crash_distance) / 5.0, 0.0, 1.0))
            return 0.0

        if mode == "barrier":
            d_safe = self.cfg.getfloat("options", "obstacle_safe_distance", fallback=12.0)
            d_crash = self.cfg.getfloat(
                "options", "obstacle_crash_distance", fallback=float(self.crash_distance)
            )
            if d_safe <= d_crash:
                raise ValueError(
                    "obstacle_safe_distance must be greater than obstacle_crash_distance."
                )
            ratio = np.clip((d_safe - distance) / (d_safe - d_crash), 0.0, 1.0)
            return float(ratio * ratio)

        raise ValueError(
            "Unsupported obstacle_penalty_mode={!r}; expected 'legacy' or 'barrier'.".format(mode)
        )

    def compute_reward_final(self, done, action):

        reward = 0
        reward_reach = 30
        reward_crash = -50
        reward_outside = -20


        if self.env_name == 'NH_center':
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:

            distance_now = self.get_distance_to_goal_3d()
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance
            self.previous_distance_from_des_point = distance_now


            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y, z = current_pose
            x_g, y_g, z_g = goal_pose
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)
            punishment_pose = punishment_xy + punishment_z


            punishment_obs = self._compute_obstacle_penalty()
            obstacle_mode = self.cfg.get(
                "options", "obstacle_penalty_mode", fallback="legacy"
            ).strip().lower()
            obstacle_lambda = (
                self.cfg.getfloat("options", "obstacle_penalty_lambda", fallback=2.0)
                if obstacle_mode == "barrier"
                else 0.5
            )


            punishment_action = 0
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = ((abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference) ** 2)
                punishment_action += (v_z_cost + z_err_cost)
            punishment_action += yaw_speed_cost


            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)


            reward = reward_distance - 0.1 * punishment_pose - obstacle_lambda * punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
        else:

            if self.termination_reason == "success":
                reward = reward_reach
            elif self.termination_reason == "crash":
                reward = reward_crash
            elif self.termination_reason == "outside":
                reward = reward_outside

        return reward


    def get_termination_state(self):


        if getattr(self, "termination_state", None) is not None:
            return dict(self.termination_state)

        goal_reached = bool(self.is_in_desired_pose())
        is_crash = bool(self.is_crashed())
        is_outside = bool(self.is_not_inside_workspace())
        is_step_limit = bool(self.step_num + 1 >= self.max_episode_steps)

        is_success = goal_reached and not is_crash and not is_outside
        if is_crash:
            reason = "crash"
        elif is_outside:
            reason = "outside"
        elif is_success:
            reason = "success"
        elif is_step_limit:
            reason = "step_limit"
        else:
            reason = None
        self.termination_state = {
            "reason": reason,
            "is_success": is_success,
            "is_crash": is_crash,
            "is_not_in_workspace": is_outside,
            "is_step_limit": is_step_limit,
        }
        return dict(self.termination_state)

    def is_done(self):

        termination = self.get_termination_state()
        self.termination_reason = termination["reason"]
        return self.termination_reason is not None

    def is_not_inside_workspace(self):

        if getattr(self, "termination_state", None) is not None:
            return bool(self.termination_state["is_not_in_workspace"])
        is_not_inside = False
        current_position = self.dynamic_model.get_position()


        if (current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or
                current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]):
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):

        if getattr(self, "termination_state", None) is not None:
            return bool(self.termination_state["is_success"])
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):

        if getattr(self, "termination_state", None) is not None:
            return bool(self.termination_state["is_crash"])
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True
        return is_crashed


    def get_distance_to_goal_3d(self):

        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):

        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

        return dis


    def print_train_info_airsim(self, action, obs, reward, info):

        msg_train_info = f"Episode: {self.episode_num}  Step: {self.step_num}  Total Steps: {self.total_step}"


        self.client.simPrintLogMessage('Training Info: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('Reward: ',
                                       f"{reward:4.4f}  Cumulative Reward: {self.cumulated_episode_reward:4.4f}")
        self.client.simPrintLogMessage(
            'Additional Info: ', str(self._compact_status_info(info))
        )
        self.client.simPrintLogMessage('Normalized State: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage('Raw State: ', str(self.dynamic_model.state_raw))
        self.client.simPrintLogMessage('Nearest Obstacle Distance: ', str(self.min_distance_to_obstacles))


    def set_pyqt_signal_multirotor(self, action, reward):

        step = int(self.total_step)


        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            action_output = np.array([action[0], 0, action[1]])
            state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])


        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude()),
                                  np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position),
                              np.asarray(self.dynamic_model.start_position),
                              np.asarray(self.dynamic_model.get_position()),
                              np.asarray(self.trajectory_list))

    def visual_log_q_value(self, q_value, action, reward):


        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x + 1, map_size_y + 1), np.nan)


        position = self.dynamic_model.get_position()
        pose_x, pose_y = position[0], position[1]
        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])


        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print(f'错误: X:{index_x} 和 Y:{index_y} 超出地图范围（visual_log_q_value）')


        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step + 1) % record_step == 0:
            if self.data_path is not None:
                np.save(self.data_path + f'/q_value_map_{self.total_step + 1}', self.q_value_map)

                self.q_value_map[5:, :, :] = np.nan
