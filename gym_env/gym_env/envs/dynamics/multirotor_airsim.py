import airsim
import numpy as np
import math
from gym import spaces
import time


class MultirotorDynamicsAirsim():


    def __init__(self, cfg) -> None:


        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')

        self.using_velocity_state = cfg.getboolean('options', 'using_velocity_state')

        self.dt = cfg.getfloat('multirotor', 'dt')

        self.observe_physical_features = cfg.getboolean(
            'wind', 'observe_in_policy', fallback=False
        )
        self.apply_wind_physics = cfg.getboolean('wind', 'apply_physics', fallback=True)
        self.action_low_pass = cfg.getboolean('options', 'action_low_pass', fallback=False)
        self.action_smoothing_beta = cfg.getfloat('options', 'action_smoothing_beta', fallback=0.6)
        if not 0.0 < self.action_smoothing_beta <= 1.0:
            raise ValueError('action_smoothing_beta must be in (0, 1].')

        self.wind_vec = (0.0, 0.0, 0.0)
        self.true_wind = np.zeros(3, dtype=np.float32)
        self.observed_wind = np.zeros(3, dtype=np.float32)
        self.wind_observation_scale = cfg.getfloat('wind', 'observation_scale', fallback=6.0)
        sensor_noise_std = cfg.getfloat('wind', 'sensor_noise_std', fallback=0.0)
        sensor_bias_std = cfg.getfloat('wind', 'sensor_bias_std', fallback=0.0)
        self.wind_sensor_uncertainty = math.hypot(
            max(sensor_noise_std, 0.0), max(sensor_bias_std, 0.0)
        )
        self.power_observation_reference_w = cfg.getfloat(
            'energy', 'power_observation_reference_w', fallback=200.0
        )
        self.airspeed_model = cfg.get(
            'wind',
            'airspeed_model',
            fallback='commanded_velocity',
        ).strip().lower()
        self._episode_rng = np.random.default_rng(
            cfg.getint('options', 'seed', fallback=cfg.getint('wind', 'field_seed', fallback=42))
        )

        self.battery_capacity_wh = cfg.getfloat(
            'multirotor', 'battery_capacity_wh', fallback=10.0
        )
        if self.battery_capacity_wh <= 0.0:
            raise ValueError('battery_capacity_wh must be positive.')
        self.battery_capacity = self.battery_capacity_wh * 3600.0
        self.current_energy = self.battery_capacity
        self.soc = 1.0


        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = None
        self.goal_random_angle = None
        self.goal_rect = None


        self.x = 0
        self.y = 0
        self.z = 0

        self.v_xy = 0
        self.v_z = 0

        self.yaw = 0
        self.yaw_rate = 0


        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        self.yaw_sp = 0
        self.commanded_airspeed_ned = np.zeros(3, dtype=np.float32)
        self.commanded_ground_velocity_ned = np.zeros(3, dtype=np.float32)
        self.control_command_semantics = "airspeed_command_with_true_wind_conversion"


        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5


        if self.navigation_3d:

            if self.using_velocity_state:
                self.state_feature_length = 6
            else:
                self.state_feature_length = 3

            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                dtype=np.float32
            )
        else:

            if self.using_velocity_state:
                self.state_feature_length = 4
            else:
                self.state_feature_length = 2

            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                dtype=np.float32
            )

        self.previous_executed_action = self._initial_executed_action()

        self.expert_feature_length = 0
        if self.observe_physical_features:
            self.expert_feature_length = 5
            self.state_feature_length += self.expert_feature_length

    def reset(self, seed=None, goal_seed=None):

        self._episode_rng = np.random.default_rng(seed)
        self.client.reset()
        self.previous_executed_action = self._initial_executed_action()
        self.commanded_airspeed_ned = np.zeros(3, dtype=np.float32)
        self.commanded_ground_velocity_ned = np.zeros(3, dtype=np.float32)

        goal_rng = np.random.default_rng(seed if goal_seed is None else goal_seed)
        self.update_goal_pose(rng=goal_rng)


        self.current_energy = self.battery_capacity
        self.soc = 1.0


        yaw_noise = self.start_random_angle * self._episode_rng.random()

        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = - self.start_position[2]

        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)


        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


        self.client.moveToZAsync(-self.start_position[2], 2).join()


        self.client.simPause(True)

    def update_energy(self, power_consumption, dt):

        consumed_energy = power_consumption * dt
        self.current_energy -= consumed_energy
        self.soc = np.clip(self.current_energy / self.battery_capacity, 0, 1)
        return self.soc

    def update_energy_j(self, consumed_energy_j):
        consumed_energy_j = max(0.0, float(consumed_energy_j))
        self.current_energy -= consumed_energy_j
        self.soc = np.clip(self.current_energy / self.battery_capacity, 0, 1)
        return self.soc

    def _initial_executed_action(self):
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        action[0] = self.v_xy_min
        return action

    def _filter_action(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {raw_action.shape}."
            )
        if self.action_low_pass:
            beta = self.action_smoothing_beta
            executed_action = beta * raw_action + (1.0 - beta) * self.previous_executed_action
        else:
            executed_action = raw_action
        executed_action = np.clip(executed_action, self.action_space.low, self.action_space.high)
        self.previous_executed_action = executed_action.astype(np.float32, copy=True)
        return self.previous_executed_action

    @staticmethod
    def airspeed_to_ground_velocity(commanded_airspeed_ned, wind_ned):

        airspeed = np.asarray(commanded_airspeed_ned, dtype=np.float32).reshape(-1)
        wind = np.asarray(wind_ned, dtype=np.float32).reshape(-1)
        if airspeed.size != 3 or wind.size != 3:
            raise ValueError("airspeed and wind must each contain 3 values")
        return airspeed + wind

    def set_action(self, action):

        action = self._filter_action(action)


        self.v_xy_sp = float(action[0])
        self.yaw_rate_sp = float(action[-1])
        if self.navigation_3d:
            self.v_z_sp = float(action[1])
        else:
            self.v_z_sp = 0


        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt


        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_air_sp = float(self.v_xy_sp * math.cos(self.yaw_sp))
        vy_air_sp = float(self.v_xy_sp * math.sin(self.yaw_sp))
        self.commanded_airspeed_ned = np.asarray(
            [vx_air_sp, vy_air_sp, -self.v_z_sp], dtype=np.float32
        )
        self.commanded_ground_velocity_ned = self.airspeed_to_ground_velocity(
            self.commanded_airspeed_ned, self.true_wind
        )
        vx_ground_sp = float(self.commanded_ground_velocity_ned[0])
        vy_ground_sp = float(self.commanded_ground_velocity_ned[1])
        vd_ground_sp = float(self.commanded_ground_velocity_ned[2])


        self.client.simPause(False)
        if len(action) == 2:

            self.client.moveByVelocityZAsync(
                vx_ground_sp, vy_ground_sp, -self.start_position[2], self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
            ).join()


        elif len(action) == 3:

            self.client.moveByVelocityAsync(
                vx_ground_sp, vy_ground_sp, vd_ground_sp, self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
            ).join()

        self.client.simPause(True)

    def get_commanded_airspeed(self):
        return self.commanded_airspeed_ned.copy()

    def get_commanded_ground_velocity(self):
        return self.commanded_ground_velocity_ned.copy()


    def set_wind_vec(self, wn, we, wd=0, observed_wind=None):


        wn, we, wd = float(wn), float(we), float(wd)
        if not self.apply_wind_physics:
            wn, we, wd = 0.0, 0.0, 0.0
            observed_wind = np.zeros(3, dtype=np.float32)
        self.current_wind = (wn, we, wd)
        self.wind_vec = self.current_wind
        self.true_wind = np.asarray(self.current_wind, dtype=np.float32)
        if observed_wind is None:
            self.observed_wind = self.true_wind.copy()
        else:
            observed = np.asarray(observed_wind, dtype=np.float32).reshape(-1)
            if observed.size != 3:
                raise ValueError(f"observed_wind must have 3 values, got {observed.size}")
            self.observed_wind = observed
        self.client.simPause(False)
        self.client.simSetWind(airsim.Vector3r(wn, we, wd))
        self.client.simPause(True)


    def get_airspeed(self):

        if self.airspeed_model in {"relative", "relative_ground_wind", "measured_relative"}:
            return self.get_velocity_ground() - self.true_wind
        v_n = self.v_xy_sp * math.cos(self.yaw_sp)
        v_e = self.v_xy_sp * math.sin(self.yaw_sp)
        v_d = -self.v_z_sp
        return np.asarray([v_n, v_e, v_d], dtype=np.float32)

    def get_velocity_ground(self):


        _,_,_, vN, vE, vD = self.get_velocity()
        return np.array([vN, vE, vD])


    def update_goal_pose(self, seed=None, rng=None):

        rng = rng or (np.random.default_rng(seed) if seed is not None else self._episode_rng)

        if self.goal_rect is None:
            distance = self.goal_distance
            noise = rng.random()
            angle = noise * self.goal_random_angle
            goal_x = distance * math.cos(angle) + self.start_position[0]
            goal_y = distance * math.sin(angle) + self.start_position[1]

        else:
            goal_x, goal_y = self.get_goal_from_rect(self.goal_rect, self.goal_random_angle, rng=rng)
            self.goal_distance = math.sqrt(goal_x * goal_x + goal_y * goal_y)

        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]


    def set_start(self, position, random_angle):

        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance=None, random_angle=0, rect=None):

        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect

    def get_goal_from_rect(self, rect_set, random_angle_set, seed=None, rng=None):

        rect = rect_set
        random_angle = random_angle_set
        rng = rng or (np.random.default_rng(seed) if seed is not None else self._episode_rng)
        noise = rng.random()
        angle = random_angle * noise - math.pi
        rect = [-128, -128, 128, 128]


        if abs(angle) == math.pi / 2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]
            else:
                goal_y = rect[1]
        elif abs(angle) <= math.pi / 4:
            goal_x = rect[2]
            goal_y = goal_x * math.tan(angle)
        elif abs(angle) > math.pi / 4 and abs(angle) <= math.pi / 4 * 3:
            if angle > 0:
                goal_y = rect[3]
                goal_x = goal_y / math.tan(angle)
            else:
                goal_y = rect[1]
                goal_x = goal_y / math.tan(angle)
        else:
            goal_x = rect[0]
            goal_y = goal_x * math.tan(angle)

        return goal_x, goal_y

    @staticmethod
    def _normalize_signed(value, reference):
        reference = max(float(reference), 1e-6)
        return float(0.5 + 0.5 * np.clip(float(value) / reference, -1.0, 1.0))

    def _goal_frame_xy(self):
        position = np.asarray(self.get_position(), dtype=np.float32)
        goal_delta = np.asarray(self.goal_position, dtype=np.float32)[:2] - position[:2]
        distance = float(np.linalg.norm(goal_delta))
        if distance <= 1e-6:
            yaw = float(self.get_attitude()[2])
            goal_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        else:
            goal_axis = goal_delta / distance
        cross_axis = np.asarray([-goal_axis[1], goal_axis[0]], dtype=np.float32)
        return goal_axis, cross_axis

    def _get_wind_observation_features(self, battery_power_w=0.0):

        goal_axis, cross_axis = self._goal_frame_xy()
        observed_wind = np.asarray(self.observed_wind, dtype=np.float32)
        ground_velocity = np.asarray(self.get_velocity_ground(), dtype=np.float32)
        estimated_airspeed = ground_velocity - observed_wind

        wind_scale = max(float(self.wind_observation_scale), 1e-6)
        ground_scale = max(float(self.v_xy_max) + wind_scale, 1e-6)
        airspeed_scale = max(float(self.v_xy_max), 1e-6)
        wind_magnitude = float(np.linalg.norm(observed_wind[:2]))
        uncertainty = max(float(self.wind_sensor_uncertainty), 1e-6)
        wind_confidence = float(
            np.clip(1.0 - uncertainty / max(wind_magnitude, uncertainty), 0.0, 1.0)
        )

        features = np.asarray(
            [
                self._normalize_signed(np.dot(observed_wind[:2], goal_axis), wind_scale),
                self._normalize_signed(np.dot(observed_wind[:2], cross_axis), wind_scale),
                self._normalize_signed(np.dot(ground_velocity[:2], goal_axis), ground_scale),
                self._normalize_signed(np.dot(estimated_airspeed[:2], goal_axis), airspeed_scale),
                wind_confidence,
            ],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)

    def _get_state_feature(self,predicted_power=0.0):


        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()

        relative_pose_z = self.get_position()[2] - self.goal_position[2]


        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255

        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255


        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]

        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = velocity[1]

        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255


        self.state_raw = np.array([
            distance,
            relative_pose_z,
            math.degrees(relative_yaw),
            linear_velocity_xy,
            linear_velocity_z,
            math.degrees(velocity[2])
        ])


        state_norm = np.array([
            distance_norm, vertical_distance_norm, relative_yaw_norm,
            linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm
        ])
        state_norm = np.clip(state_norm, 0, 255)


        self.state_norm = state_norm
        if self.navigation_3d:
            if self.using_velocity_state == False:
                self.state_norm = state_norm[:3]
        else:

            self.state_norm = np.array([state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.using_velocity_state == False:
                self.state_norm = state_norm[:2]

        self.final_state = self.state_norm

        if self.observe_physical_features:
            expert_features = self._get_wind_observation_features(predicted_power) * 255.0
            self.final_state = np.concatenate((self.state_norm, expert_features))

        return self.final_state

    def _get_relative_yaw(self):

        current_position = self.get_position()

        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)


        yaw_current = self.get_attitude()[2]


        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):

        position = self.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]


    def get_velocity(self):

        states = self.client.getMultirotorState()

        linear_velocity = states.kinematics_estimated.linear_velocity

        angular_velocity = states.kinematics_estimated.angular_velocity


        vN = linear_velocity.x_val
        vE = linear_velocity.y_val
        vD = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val


        velocity_xy = math.hypot(vN, vE)

        velocity_z = -vD


        return [velocity_xy, velocity_z, yaw_rate, vN, vE, vD]

    def get_attitude(self):

        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):

        return [0.0, 0.0, self.yaw_sp]

    def get_distance_to_goal_2d(self):

        return math.sqrt(
            pow(self.get_position()[0] - self.goal_position[0], 2) +
            pow(self.get_position()[1] - self.goal_position[1], 2)
        )
