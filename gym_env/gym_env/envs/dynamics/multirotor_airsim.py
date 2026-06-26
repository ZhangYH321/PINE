import airsim  # AirSim仿真器Python API
import numpy as np  # 数值计算
import math  # 数学运算
from gym import spaces  # Gym环境的动作/观测空间定义
import time

class MultirotorDynamicsAirsim():
    '''
    基于AirSim的多旋翼无人机动力学模型（用于视觉导航任务）
    控制器采用AirSim Simple Flight（https://microsoft.github.io/AirSim/simple_flight/）
    核心控制API：client.moveByVelocityZAsync(v_x, v_y, v_z, yaw_rate)
    '''

    def __init__(self, cfg) -> None:
        """
        初始化多旋翼动力学模型
        参数:
            cfg: ConfigParser对象，包含无人机控制参数、环境配置等
        """
        # -------------------------- 配置参数读取 --------------------------
        # 是否启用3D导航（Z轴可控），2D导航仅控制XY平面+偏航
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        # 是否将速度纳入状态特征（否则仅用位置/偏航）
        self.using_velocity_state = cfg.getboolean('options', 'using_velocity_state')
        # 控制步长（每次动作执行的时间，单位：秒）
        self.dt = cfg.getfloat('multirotor', 'dt')

        wind_config = cfg.getboolean('options', 'use_wind') if cfg.has_option('options', 'use_wind') else None
        # 判断值是否为 None，设置最终结果
        self.use_wind = wind_config if wind_config is not None else False

        self.wind_vec = (0.0, 0.0, 0.0)  # (N, E, D) m/s
        # [新增] 电池参数
        self.battery_capacity = 30000  # 电池总容量 (Joules)，例如 100Wh = 360000J
        self.current_energy = self.battery_capacity
        self.soc = 1.0  # State of Charge (0.0 - 1.0)


        # -------------------------- AirSim客户端初始化 --------------------------
        # 创建AirSim多旋翼客户端并确认连接
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()  # 验证与AirSim仿真器的连接
        self.client.enableApiControl(True)  # 启用API控制（接管无人机）
        self.client.armDisarm(True)  # 解锁无人机（允许电机启动）

        # -------------------------- 起点/目标点参数初始化 --------------------------
        self.start_position = [0, 0, 0]  # 无人机起始位置 [x, y, z]（z为高度，正数）
        self.start_random_angle = None  # 起始偏航角随机范围（弧度）
        self.goal_position = [0, 0, 0]  # 目标位置 [x, y, z]
        self.goal_distance = None  # 目标点距起点的距离（用于随机生成目标）
        self.goal_random_angle = None  # 目标点角度随机范围（弧度）
        self.goal_rect = None  # 目标点生成的矩形区域 [x_min, y_min, x_max, y_max]

        # -------------------------- 无人机状态变量 --------------------------
        # 位置状态（实时更新）
        self.x = 0  # X轴位置（世界坐标系）
        self.y = 0  # Y轴位置（世界坐标系）
        self.z = 0  # Z轴高度（正数，AirSim中z轴向下为正，需转换）
        # 速度状态
        self.v_xy = 0  # XY平面合速度（m/s）
        self.v_z = 0  # Z轴速度（m/s，向上为正）
        # 偏航状态
        self.yaw = 0  # 当前偏航角（弧度，世界坐标系）
        self.yaw_rate = 0  # 当前偏航率（弧度/秒）


        # -------------------------- 控制指令变量 --------------------------
        self.v_xy_sp = 0  # 期望XY平面合速度（m/s）
        self.v_z_sp = 0  # 期望Z轴速度（m/s）
        self.yaw_rate_sp = 0  # 期望偏航率（弧度/秒）
        self.yaw_sp = 0

        # -------------------------- 动作空间参数 --------------------------
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')  # XY平面最大加速度（m/s²）
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')  # XY平面最大速度（m/s）
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')  # XY平面最小速度（m/s）
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')  # Z轴最大速度（m/s）
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')  # 最大偏航率（度/秒）
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)  # 转换为弧度/秒
        self.max_vertical_difference = 5  # 最大垂直高度偏差（用于状态归一化）

        # -------------------------- 定义动作空间（适配Gym接口） --------------------------
        if self.navigation_3d:
            # 3D导航：动作维度为3 [v_xy, v_z, yaw_rate]
            if self.using_velocity_state:
                self.state_feature_length = 6  # 状态特征维度（距离、高度差、偏航差、xy速度、z速度、偏航率）
            else:
                self.state_feature_length = 3  # 状态特征维度（距离、高度差、偏航差）
            # 动作空间范围：v_xy [min, max], v_z [-max, max], yaw_rate [-max_rad, max_rad]
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                dtype=np.float32
            )
        else:
            # 2D导航：动作维度为2 [v_xy, yaw_rate]（z轴速度固定）
            if self.using_velocity_state:
                self.state_feature_length = 4  # 状态特征维度（距离、偏航差、xy速度、偏航率）
            else:
                self.state_feature_length = 2  # 状态特征维度（距离、偏航差）
            # 动作空间范围：v_xy [min, max], yaw_rate [-max_rad, max_rad]
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                dtype=np.float32
            )


        if self.use_wind:
            # [新增] 物理特征维度扩充
            # 原有维度 + 风速(3) + SOC(1) + 专家预测功率(1)
            # 假设原有 state_feature_length 是 6 (3D+速度)
            self.expert_feature_length = 4
            self.state_feature_length += self.expert_feature_length

    def reset(self,seed = None):
        """
        重置无人机状态和位置（强化学习环境重置时调用）
        1. 重置AirSim仿真器
        2. 重新生成目标位置
        3. 设置无人机起始位置和偏航角
        4. 解锁并起飞到指定高度
        """
        self.client.reset()  # 重置AirSim仿真器状态
        # 重新生成目标位置（随机化）
        self.update_goal_pose(seed)

        # [新增] 重置电池状态
        self.current_energy = self.battery_capacity
        self.soc = 1.0

        # 重置起始偏航角（添加随机噪声）
        yaw_noise = self.start_random_angle * np.random.random()  # 0~start_random_angle范围内随机偏航角
        # 设置无人机在AirSim中的初始位姿
        pose = self.client.simGetVehiclePose()  # 获取当前位姿（用于修改）
        pose.position.x_val = self.start_position[0]  # X轴起始位置
        pose.position.y_val = self.start_position[1]  # Y轴起始位置
        pose.position.z_val = - self.start_position[2]  # AirSim中z轴向下为正，转换为高度（正数）
        # 设置初始偏航角（俯仰/横滚为0，偏航为随机噪声）
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)  # 应用初始位姿（True表示忽略碰撞）

        # 恢复仿真并解锁无人机
        self.client.simPause(False)  # 取消仿真暂停
        self.client.enableApiControl(True)  # 重新启用API控制
        self.client.armDisarm(True)  # 重新解锁

        # 起飞到指定高度（异步执行并等待完成）
        self.client.moveToZAsync(-self.start_position[2], 2).join()  # 2为起飞速度（m/s）


        self.client.simPause(True)  # 暂停仿真（等待下一步动作）

    def update_energy(self, power_consumption, dt):
        """
        [新增] 更新电池状态
        """
        consumed_energy = power_consumption * dt
        self.current_energy -= consumed_energy
        self.soc = np.clip(self.current_energy / self.battery_capacity, 0, 1)
        return self.soc

    def set_action(self, action):
        """
        执行动作指令，控制无人机运动
        参数:
            action: 强化学习模型输出的动作向量，格式：
                    - 3D导航: [v_xy, v_z, yaw_rate]
                    - 2D导航: [v_xy, yaw_rate]
        处理逻辑：
            1. 解析动作指令，转换为无人机期望速度/偏航率
            2. 计算期望偏航角（基于当前偏航+偏航率*步长）
            3. 转换为机体坐标系下的速度指令
            4. 调用AirSim API执行控制指令
        """
        # 解析动作指令（添加缩放系数，适配实际控制范围）
        self.v_xy_sp = action[0] * 0.7  # XY平面期望速度（缩放系数0.7防止超调）
        self.yaw_rate_sp = action[-1] * 2  # 期望偏航率（缩放系数2增强响应）
        if self.navigation_3d:
            self.v_z_sp = float(action[1])  # 3D导航时解析Z轴速度指令
        else:
            self.v_z_sp = 0  # 2D导航时Z轴速度固定为0

        # 计算期望偏航角（当前偏航角 + 偏航率 * 控制步长）
        self.yaw = self.get_attitude()[2]  # 获取当前偏航角（弧度）
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        # 偏航角归一化到[-π, π]范围
        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        # 将机体坐标系的XY速度转换为世界坐标系速度
        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)  # X轴世界速度
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)  # Y轴世界速度

        # 恢复仿真并执行控制指令
        self.client.simPause(False)
        if len(action) == 2:
            # 2D导航：控制XY速度+固定Z高度+偏航率
            self.client.moveByVelocityZAsync(
                vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,  # 最大自由度（允许偏航）
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))  # 偏航率模式
            ).join()  # 等待动作执行完成
            # 注释的备选控制模式：仅前向驱动，固定偏航角
            # self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
            #                                 drivetrain=airsim.DrivetrainType.ForwardOnly,
            #                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(0))).join()
        elif len(action) == 3:
            # 3D导航：控制XYZ速度+偏航率
            self.client.moveByVelocityAsync(
                vx_local_sp, vy_local_sp, -self.v_z_sp, self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
            ).join()

        self.client.simPause(True)  # 再次暂停仿真


    def set_wind_vec(self,wn,we,wd=0):
        self.current_wind = (float(wn), float(we), float(wd))
        self.wind_vec = self.current_wind
        self.client.simPause(False)
        self.client.simSetWind(airsim.Vector3r(wn, we, wd))
        self.client.simPause(True)

    # def get_wind_vec(self):
    #     if self.current_wind:
    #         self.wind_vec = self.current_wind
    #     else:
    #         self.wind_vec = (0,0,0)

    def get_airspeed(self):
        """
        返回 NED 坐标系下的空速向量
        """
        vN = self.v_xy_sp * math.cos(self.yaw_sp)
        vE = self.v_xy_sp * math.sin(self.yaw_sp)
        vD = -self.v_z_sp
        return np.array([vN, vE, vD])

    def get_velocity_ground(self):
        """
        [新增] 获取NED坐标系下的地速
        """
        # velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        # [v_xy,v_z,yaw] = self.get_velocity()
        # vN = v_xy * math.cos(yaw)
        # vE = v_xy * math.sin(yaw)
        # vD = v_z
        _,_,_, vN, vE, vD = self.get_velocity()
        return np.array([vN, vE, vD])

    # def set_goal_pose(self,goal):
    #     self.goal_position = goal

    def update_goal_pose(self,seed = None):
        """
        随机生成目标位置
        两种模式：
            1. 距离+角度模式：基于起点距离和随机角度生成目标
            2. 矩形区域模式：在指定矩形区域内随机生成目标
        """
        # 模式1：距离+角度随机（无矩形区域限制）
        if self.goal_rect is None:
            distance = self.goal_distance  # 目标距起点的固定距离
            noise = np.random.random()  # 0~1随机数
            angle = noise * self.goal_random_angle  # 0~goal_random_angle随机角度（弧度）
            goal_x = distance * math.cos(angle) + self.start_position[0]  # 目标X坐标
            goal_y = distance * math.sin(angle) + self.start_position[1]  # 目标Y坐标
        # 模式2：矩形区域内随机（覆盖模式1）
        else:
            goal_x, goal_y = self.get_goal_from_rect(self.goal_rect, self.goal_random_angle,seed)
            self.goal_distance = math.sqrt(goal_x * goal_x + goal_y * goal_y)  # 重新计算目标距离
        # 设置目标位置（Z轴与起点一致）
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]
        # print('New goal pose: ', self.goal_position)  # 调试用：打印新目标位置

    def set_start(self, position, random_angle):
        """
        设置无人机起始位置和偏航角随机范围
        参数:
            position: 起始位置 [x, y, z]
            random_angle: 起始偏航角随机范围（弧度）
        """
        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance=None, random_angle=0, rect=None):
        """
        设置目标位置生成参数
        参数:
            distance: 目标距起点的距离（可选，用于距离+角度模式）
            random_angle: 目标角度随机范围（弧度）
            rect: 目标生成的矩形区域 [x_min, y_min, x_max, y_max]（可选，用于矩形模式）
        """
        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect

    def get_goal_from_rect(self, rect_set, random_angle_set,seed = None):
        """
        在指定矩形区域内基于随机角度生成目标位置
        参数:
            rect_set: 矩形区域 [x_min, y_min, x_max, y_max]
            random_angle_set: 角度随机范围（弧度）
        返回:
            goal_x, goal_y: 目标位置的X/Y坐标
        """
        rect = rect_set  # 矩形区域参数
        random_angle = random_angle_set  # 角度随机范围
        if seed is not None:
            np.random.seed(seed)
            # print(seed)
        noise = np.random.random()  # 0~1随机数
        angle = random_angle * noise - math.pi  # 角度归一化到[-π, π]
        rect = [-128, -128, 128, 128]  # 固定矩形区域（覆盖输入，可根据需求修改）

        # 根据角度分段计算目标位置（确保在矩形内）
        if abs(angle) == math.pi / 2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]  # 上边界
            else:
                goal_y = rect[1]  # 下边界
        elif abs(angle) <= math.pi / 4:
            goal_x = rect[2]  # 右边界
            goal_y = goal_x * math.tan(angle)  # 按角度计算Y坐标
        elif abs(angle) > math.pi / 4 and abs(angle) <= math.pi / 4 * 3:
            if angle > 0:
                goal_y = rect[3]  # 上边界
                goal_x = goal_y / math.tan(angle)  # 按角度计算X坐标
            else:
                goal_y = rect[1]  # 下边界
                goal_x = goal_y / math.tan(angle)
        else:
            goal_x = rect[0]  # 左边界
            goal_y = goal_x * math.tan(angle)

        return goal_x, goal_y

    def _get_state_feature(self,predicted_power=0.0):
        '''
        更新并获取归一化的无人机状态特征（0~255范围）
        返回:
            state_norm: 归一化后的状态特征数组
        状态特征说明（根据配置调整维度）：
            - 3D+速度: [距离归一化, 高度差归一化, 偏航差归一化, XY速度归一化, Z速度归一化, 偏航率归一化]
            - 3D+无速度: [距离归一化, 高度差归一化, 偏航差归一化]
            - 2D+速度: [距离归一化, 偏航差归一化, XY速度归一化, 偏航率归一化]
            - 2D+无速度: [距离归一化, 偏航差归一化]
        '''
        # 1. 计算核心状态量
        distance = self.get_distance_to_goal_2d()  # 到目标的2D距离
        relative_yaw = self._get_relative_yaw()  # 相对目标的偏航误差（-π~π弧度）
        # 垂直高度差（当前高度 - 目标高度）
        relative_pose_z = self.get_position()[2] - self.goal_position[2]
        # 2. 状态归一化（映射到0~255）
        # 垂直高度差归一化：-max_vertical_difference→0，+max_vertical_difference→255
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255
        # 2D距离归一化：0→0，goal_distance→255
        distance_norm = distance / self.goal_distance * 255
        # 偏航误差归一化：-π→0，+π→255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # 3. 速度/偏航率归一化（若启用速度状态）
        velocity = self.get_velocity()  # 获取当前速度 [xy速度, z速度, 偏航率]
        linear_velocity_xy = velocity[0]  # XY平面合速度
        # XY速度归一化：v_xy_min→0，v_xy_max→255
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = velocity[1]  # Z轴速度
        # Z速度归一化：-v_z_max→0，+v_z_max→255
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255  # 偏航率归一化

        # 4. 原始状态记录（未归一化，用于调试/奖励计算）
        self.state_raw = np.array([
            distance,  # 2D距离（米）
            relative_pose_z,  # 垂直高度差（米）
            math.degrees(relative_yaw),  # 偏航误差（度）
            linear_velocity_xy,  # XY合速度（米/秒）
            linear_velocity_z,  # Z轴速度（米/秒）
            math.degrees(velocity[2])  # 偏航率（度/秒）
        ])

        # 5. 拼接归一化状态并裁剪（防止超出0~255）
        state_norm = np.array([
            distance_norm, vertical_distance_norm, relative_yaw_norm,
            linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm
        ])
        state_norm = np.clip(state_norm, 0, 255)  # 确保所有值在0~255


        v_ground = self.get_velocity_ground()
        v_wind = self.wind_vec
        dot_product = np.dot(v_ground, v_wind)
        angle_norm = np.clip(np.tanh(dot_product / 25.0) * 255.0,0.0,255.0)

        soc_feature = np.array(self.soc) * 255.0
        # 3. 专家预测功率归一化 (假设最大功率 1000W)
        power_feature = np.clip(predicted_power / 200.0 * 255.0, 0, 255)

        # 6. 根据配置裁剪状态维度
        if self.navigation_3d:
            if self.using_velocity_state == False:
                self.state_norm = state_norm[:3]  # 3D无速度：仅保留距离、高度差、偏航差
        else:
            # 2D导航：重新排列维度（距离、偏航差、XY速度、偏航率）
            self.state_norm = np.array([state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.using_velocity_state == False:
                self.state_norm = state_norm[:2]  # 2D无速度：仅保留距离、偏航差

        self.final_state = self.state_norm  # 保存归一化状态

        if self.use_wind:
            expert_features = np.array([soc_feature, angle_norm, state_norm[3], state_norm[5]], dtype=object)
            self.final_state = np.concatenate((self.state_norm, expert_features))

        return self.final_state

    def _get_relative_yaw(self):
        '''
        计算无人机当前偏航角与目标方向的偏差（相对偏航角）
        返回:
            yaw_error: 相对偏航误差（弧度，范围[-π, π]）
        计算逻辑：
            1. 计算目标相对于无人机的方向角
            2. 计算当前偏航角与目标方向角的差值
            3. 将差值归一化到[-π, π]
        '''
        current_position = self.get_position()  # 获取当前位置 [x, y, z]
        # 1. 计算目标相对于无人机的方向角
        relative_pose_x = self.goal_position[0] - current_position[0]  # X轴相对距离
        relative_pose_y = self.goal_position[1] - current_position[1]  # Y轴相对距离
        angle = math.atan2(relative_pose_y, relative_pose_x)  # 目标方向角（弧度）

        # 2. 获取当前偏航角
        yaw_current = self.get_attitude()[2]  # [roll, pitch, yaw]中的偏航角

        # 3. 计算偏航误差并归一化到[-π, π]
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):
        """
        获取无人机当前位置（世界坐标系）
        返回:
            [x, y, z]: X/Y轴为平面坐标，Z轴为高度（正数，AirSim z轴向下为正，需取反）
        """
        position = self.client.simGetVehiclePose().position  # AirSim原始位置（z向下为正）
        return [position.x_val, position.y_val, -position.z_val]

    # def get_velocity(self):
    #     """
    #     获取无人机当前速度和偏航率
    #     返回:
    #         [velocity_xy, velocity_z, yaw_rate]:
    #             - velocity_xy: XY平面合速度（m/s）
    #             - velocity_z: Z轴速度（m/s，向上为正）
    #             - yaw_rate: 偏航率（弧度/秒）
    #     """
    #     states = self.client.getMultirotorState()  # 获取多旋翼状态
    #     # 线速度（AirSim原始值，z向下为正）
    #     linear_velocity = states.kinematics_estimated.linear_velocity
    #     # 角速度（AirSim原始值，z轴为偏航率）
    #     angular_velocity = states.kinematics_estimated.angular_velocity
    #
    #     # 计算XY平面合速度
    #     velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
    #     velocity_z = linear_velocity.z_val  # Z轴速度（向下为正）
    #     yaw_rate = angular_velocity.z_val  # 偏航率（弧度/秒）
    #
    #     return [velocity_xy, -velocity_z, yaw_rate]  # Z轴速度取反（向上为正）
    def get_velocity(self):
        """
        获取无人机当前速度和偏航率
        返回:
            [vN, vE, vD, yaw_rate]:
                - vN/vE/vD: NED坐标系下的对地速度（m/s，D向下为正）
                - yaw_rate: 偏航率（弧度/秒）
        """
        states = self.client.getMultirotorState()  # 获取多旋翼状态
        # 线速度（AirSim原始值，NED坐标系，z/D向下为正）
        linear_velocity = states.kinematics_estimated.linear_velocity
        # 角速度（AirSim原始值，z轴为偏航率）
        angular_velocity = states.kinematics_estimated.angular_velocity

        # 直接读取原始速度分量，避免二次计算的误差
        vN = linear_velocity.x_val
        vE = linear_velocity.y_val
        vD = linear_velocity.z_val  # D向下为正
        yaw_rate = angular_velocity.z_val  # 偏航率（弧度/秒）

        # 计算XY平面合速度
        velocity_xy = math.hypot(vN, vE)
        # Z轴速度（向上为正）
        velocity_z = -vD


        return [velocity_xy, velocity_z, yaw_rate,vN, vE, -vD]

    def get_attitude(self):
        """
        获取无人机当前姿态（欧拉角）
        返回:
            [roll, pitch, yaw]: 横滚、俯仰、偏航角（弧度）
        """
        self.state_current_attitude = self.client.simGetVehiclePose().orientation  # 四元数姿态
        return airsim.to_eularian_angles(self.state_current_attitude)  # 转换为欧拉角

    def get_attitude_cmd(self):
        """
        获取期望姿态（仅偏航角有期望值，横滚/俯仰为0）
        返回:
            [0.0, 0.0, yaw_sp]: 期望欧拉角（弧度）
        """
        return [0.0, 0.0, self.yaw_sp]

    def get_distance_to_goal_2d(self):
        """
        计算无人机到目标的2D欧氏距离（XY平面）
        返回:
            distance: 2D距离（米）
        """
        return math.sqrt(
            pow(self.get_position()[0] - self.goal_position[0], 2) +
            pow(self.get_position()[1] - self.goal_position[1], 2)
        )