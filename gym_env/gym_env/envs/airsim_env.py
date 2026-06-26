import gym
from gym import spaces
import airsim  # AirSim仿真器API
from configparser import NoOptionError  # 配置文件解析异常
import keyboard  # 键盘调试
import time
import torch as th  # PyTorch（用于Q值计算）
import numpy as np  # 数值计算
import math  # 数学运算
import cv2  # 图像处理（深度图/视觉特征）

# 导入不同无人机动力学模型
from .dynamics.multirotor_simple import MultirotorDynamicsSimple  # 简易多旋翼动力学
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim  # AirSim多旋翼动力学
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple  # 简易固定翼动力学
from .power_model import MultirotorPowerModel, RotorcraftParams
from .wind_env import GridWindField
# PyQt5信号（用于可视化）
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

class AirsimGymEnv(gym.Env, QtCore.QThread):
    """
    基于AirSim的无人机导航Gym环境
    继承gym.Env实现标准强化学习环境接口，继承QThread实现异步运行
    """
    # PyQt5信号定义（用于实时可视化）
    action_signal = pyqtSignal(int, np.ndarray)  # 动作信号：步数+动作数组
    state_signal = pyqtSignal(int, np.ndarray)  # 状态信号：步数+状态数组
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)  # 姿态信号：步数+实际姿态+期望姿态
    reward_signal = pyqtSignal(int, float, float)  # 奖励信号：步数+单步奖励+累计奖励
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # 位姿信号：目标点+起点+当前位置+轨迹
    lgmd_signal = pyqtSignal(float, float, np.ndarray)  # LGMD信号：最小障碍距离+LGMD输出+分割特征



    def __init__(self) -> None:
        super().__init__()
        # 设置numpy打印格式（便于调试）
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("初始化AirSim-Gym环境.")

        # 初始化变量
        self.model = None  # 强化学习模型（训练时赋值）
        self.data_path = None  # 数据保存路径
        self.lgmd = None  # LGMD视觉模型实例
        self.goal_seed = None

        # 新增：crash概率统计相关变量
        self.crash_flag = False  # 单回合是否碰撞
        self.episode_done = False  # 单回合是否结束
        self.episode_crash_count = 0  # 累计碰撞回合数
        self.episode_total_count = 0  # 累计总回合数

    def set_config(self, cfg):
        """
        从配置文件加载环境配置
        参数:
            cfg: ConfigParser对象，包含环境/动力学/感知等配置
        """
        self.cfg = cfg
        # 读取核心配置
        self.env_name = cfg.get('options', 'env_name')  # 环境名称（如NH_center/Trees）
        self.dynamic_name = cfg.get('options', 'dynamic_name')  # 动力学模型名称
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')  # 键盘调试开关
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')  # Q值地图生成开关
        self.perception_type = cfg.get('options', 'perception')  # 感知类型（vector/lgmd/image）

        # 初始化LGMD视觉模型（若启用）
        if self.perception_type == 'lgmd':
            self.lgmd = LGMD(type='origin', p_threshold=50, s_threshold=0, Ki=2,
                             i_layer_size=3, activate_coeff=1, use_on_off=True)
            self.split_out_last = np.array([0, 0, 0, 0, 0])  # LGMD特征缓存

        print(f'环境: {self.env_name}  动力学模型: {self.dynamic_name}  感知类型: {self.perception_type}')

        # 初始化无人机动力学模型
        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_model = FixedwingDynamicsSimple(cfg)  # 简易固定翼
        elif self.dynamic_name == 'SimpleMultirotor':
            self.dynamic_model = MultirotorDynamicsSimple(cfg)  # 简易多旋翼
        elif self.dynamic_name == 'Multirotor':
            self.dynamic_model = MultirotorDynamicsAirsim(cfg)  # AirSim多旋翼
        else:
            raise Exception("无效的动力学模型名称!", self.dynamic_name)

        # 根据不同环境设置起点/终点/工作空间/最大步数
        if self.env_name == 'NH_center':
            start_position = [0, 0, 5]
            goal_rect = [-128, -128, 128, 128]  # 目标区域（矩形）
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)  # 随机起始角度
            self.dynamic_model.set_goal(random_angle=math.pi * 2, rect=goal_rect)

            self.work_space_x = [-140, 140]  # X轴工作空间范围
            self.work_space_y = [-140, 140]  # Y轴工作空间范围
            self.work_space_z = [0.5, 20]  # Z轴工作空间范围
            self.max_episode_steps = 1000  # 单回合最大步数

            # power model
            self.params = RotorcraftParams()
            self.power_model = MultirotorPowerModel(self.params)

            #wind
            self.wind_field = GridWindField(
                map_range_xy=[-140, 140, -140, 140],
                M=10,
                N=10,
                wind_min=0,
                wind_max=4,
                dynamic_model = self.dynamic_model,
                seed = self.cfg.getint('wind', 'seed', fallback=42),
                turbulence = self.cfg.getboolean('wind', 'use_turbulence', fallback=True),
            )


        elif self.env_name == 'NH_tree':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [start_position[0], start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
            self.max_episode_steps = 400
        elif self.env_name == 'City':
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 400
        elif self.env_name == 'City_400':
            start_position = [0, 0, 50]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 800
        elif self.env_name == 'Tree_200':
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 600
        elif self.env_name == 'SimpleAvoid':
            start_position = [0, 0, 5]
            goal_distance = 50
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(distance=goal_distance, random_angle=math.pi * 2)
            self.work_space_x = [start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 400
        elif self.env_name == 'Forest':
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 300
        elif self.env_name == 'Trees':
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(start_position, random_angle=math.pi * 2)
            self.dynamic_model.set_goal(distance=goal_distance, random_angle=math.pi * 2)
            self.work_space_x = [start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 500
        else:
            raise Exception("无效的环境名称!", self.env_name)

        # 初始化AirSim客户端和环境参数
        self.client = self.dynamic_model.client  # AirSim客户端实例
        self.state_feature_length = self.dynamic_model.state_feature_length  # 状态特征维度
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')  # CNN特征维度

        # 训练状态变量
        self.episode_num = 0  # 当前回合数
        self.total_step = 0  # 总步数
        self.step_num = 0  # 当前回合步数
        self.cumulated_episode_reward = 0  # 当前回合累计奖励
        self.previous_distance_from_des_point = 0  # 上一步到目标的距离
        self.CumulatedPower = 0

        # 环境参数
        self.crash_distance = cfg.getint('environment', 'crash_distance')  # 碰撞距离阈值
        self.accept_radius = cfg.getint('environment', 'accept_radius')  # 目标到达半径
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')  # 深度图最大距离
        self.screen_height = cfg.getint('environment', 'screen_height')  # 图像高度
        self.screen_width = cfg.getint('environment', 'screen_width')  # 图像宽度
        self.trajectory_list = []  # 轨迹记录

        # 定义观测空间（根据感知类型）
        if self.perception_type == 'vector' or self.perception_type == 'lgmd':
            # 向量观测：CNN特征+状态特征
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(1, self.cnn_feature_length + self.state_feature_length),
                dtype=np.float32
            )
        else:
            # 图像观测：深度图+状态特征（H,W,2）
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.screen_height, self.screen_width, 2),
                dtype=np.uint8
            )

        # 动作空间（由动力学模型定义）
        self.action_space = self.dynamic_model.action_space

        # 奖励函数类型（读取配置，无则使用默认）
        self.reward_type = None
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print(f'奖励函数类型: {self.reward_type}')
        except NoOptionError:
            self.reward_type = None


    def set_goal_seed(self,goal_seed):
        self.goal_seed = goal_seed

    def reset(self):
        """
        重置环境（增加安全检查，防止开局即炸）
        """

        if hasattr(self.dynamic_model, "set_wind_vec"):
            self.dynamic_model.set_wind_vec(0, 0, 0)
        self.dynamic_model.reset(self.goal_seed)

        # 2. 重置状态标志
        self.crash_flag = False
        self.episode_done = False
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.CumulatedPower = 0
        self.trajectory_list = []


        # 4. 更新目标距离
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        # 5. 获取初始观测
        obs = self.get_obs()

        return obs

    def step(self, action):
        """
        执行动作（强化学习标准接口）
        参数:
            action: 模型输出的动作
        返回:
            obs: 新观测
            reward: 单步奖励
            done: 回合是否结束
            info: 附加信息（成功/碰撞/步数等）
        """


        # 设置动作到动力学模型
        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_model.set_action(action, self.step_num)  # 固定翼需传入步数
        else:
            self.dynamic_model.set_action(action)  # 多旋翼直接设置

        # 记录当前位置（轨迹）
        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        #设置风速
        if hasattr(self, "wind_field"):
            self.wind_field.update_wind_by_pos(position_ue4)


        v_airspeed = self.dynamic_model.get_airspeed() if hasattr(self.dynamic_model, "get_airspeed") else np.zeros(3)
        wind_ned = getattr(self.dynamic_model, "wind_vec", np.zeros(3))
        # vN = self.dynamic_model.v_xy_sp * math.cos(self.dynamic_model.yaw_sp)
        # vE = self.dynamic_model.v_xy_sp * math.sin(self.dynamic_model.yaw_sp)
        # vD = -self.dynamic_model.v_z_sp  # down positive for AirSim
        Pi, P0, Pp, dbg = self.power_model.power_components(
            v_airspeed=v_airspeed,
        )
        P = Pi + P0 + Pp
        if hasattr(self.dynamic_model, "get_velocity_ground"):
            v_ground = self.dynamic_model.get_velocity_ground()
        else:
            v_ground = np.zeros(3, dtype=np.float32)
        wind_vec = np.asarray(wind_ned, dtype=np.float32)
        v_ground_vec = np.asarray(v_ground, dtype=np.float32)
        wind_dot = float(np.dot(v_ground_vec, wind_vec))
        v_xy_max = float(getattr(self.dynamic_model, "v_xy_max", 1.0))
        wind_max = float(getattr(getattr(self, "wind_field", None), "wind_max", 4.0))
        eta_rho = self.cfg.getfloat('PGVL', 'eta_rho', fallback=1.0)
        denom = max(v_xy_max * wind_max, 1e-6)
        headwind_ratio = float(np.clip(max(0.0, -wind_dot) / denom, 0.0, 1.0))
        pgvl_rho = 1.0 + eta_rho * headwind_ratio
        # E = P * self.dynamic_model.dt
        # 更新无人机电池状态
        if hasattr(self.dynamic_model, "update_energy"):
            self.dynamic_model.update_energy(P, self.dynamic_model.dt)
        # 获取新观测
        obs = self.get_obs(P)
        # 判断回合是否结束
        done = self.is_done()
        # 附加信息
        info = {
            'is_success': self.is_in_desired_pose(),  # 是否到达目标
            'is_crash': self.is_crashed(),  # 是否碰撞
            'is_not_in_workspace': self.is_not_inside_workspace(),  # 是否超出工作空间
            'step_num': self.step_num, # 当前步数,
            "SOC": getattr(self.dynamic_model, "soc", 1.0),
            "pgvl_rho": float(pgvl_rho),
            "pgvl_headwind_ratio": float(headwind_ratio),
            "pgvl_wind_dot": float(wind_dot),
            # "wind_x":self.x,
            # "wind_y":self.y,
            "Cumulative Reward": f"{self.cumulated_episode_reward: 4.4f}",
            "Cumulated Power": f"{self.CumulatedPower: 4.4f}"
        }

        # 新增：更新crash_flag和episode_done
        self.crash_flag = info['is_crash']
        self.episode_done = done

        # 新增：累计回合数和碰撞数（用于crash概率统计）
        if done:
            self.episode_total_count += 1
            if self.crash_flag:
                self.episode_crash_count += 1

        # 计算奖励（根据奖励类型选择函数）
        if self.dynamic_name == 'SimpleFixedwing':
            reward = self.compute_reward_final_fixedwing(done, action)
        elif self.reward_type == 'reward_with_action':
            reward = self.compute_reward_with_action(done, action)
        elif self.reward_type == 'reward_new':
            reward = self.compute_reward_multirotor_new(done, action)
        elif self.reward_type == 'reward_lqr':
            reward = self.compute_reward_lqr(done, action)
        elif self.reward_type == 'reward_final':
            reward = self.compute_reward_final(done, action)
        elif self.reward_type == "reward_wind":
             reward = self.compute_reward_wind(done, action, P, v_ground, wind_ned)
        else:
            reward = self.compute_reward(done, action)

        # 累计奖励
        self.cumulated_episode_reward += reward
        #累计能源
        self.CumulatedPower += P
        if done:
            print(info)

        # 打印训练信息
        self.print_train_info_airsim(action, obs, reward, info)

        # 发送PyQt5可视化信号
        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            self.set_pyqt_signal_fixedwing(action, reward, done)
        else:
            self.set_pyqt_signal_multirotor(action, reward)

        # 键盘调试（暂停并打印详细信息）
        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])  # 角度转换（弧度→度）
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('=' * 100)
            print(f'回合 {self.episode_num}  步数 {self.step_num}  总步数 {self.total_step}')
            print(f'动作 {action_copy}')
            print(f'状态 {state_copy}')
            print(f'归一化状态 {self.dynamic_model.state_norm}')
            print(f'奖励 {reward:.3f}  累计奖励 {self.cumulated_episode_reward:.3f}')
            print(f'回合结束 {done}')
            keyboard.wait('a')  # 按a键继续

        # 生成Q值地图（仅TD3/SAC算法）
        if self.generate_q_map and (
                self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC'):
            if self.model is not None:
                with th.no_grad():  # 禁用梯度计算
                    obs_copy = obs.copy()
                    if self.perception_type != 'vector':
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)
                    # 计算当前动作的Q值
                    q_value_current = self.model.critic(
                        th.from_numpy(obs_copy[tuple([None])]).float().cuda(),
                        th.from_numpy(action[None]).float().cuda()
                    )
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]
                    q_value = min(q_1, q_2)[0]  # TD3取两个Q网络的最小值
                    # 记录Q值
                    self.visual_log_q_value(q_value, action, reward)

        # 更新步数
        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    # ! -------------------------观测获取模块------------------------------------------
    def get_obs(self,P=0):
        """
        根据感知类型获取观测
        返回:
            obs: 观测数据（向量/图像/LGMD特征）
        """
        if self.perception_type == 'vector':
            obs = self.get_obs_vector(P)
        elif self.perception_type == 'lgmd':
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()

        return obs

    def get_obs_image(self):
        """
        获取图像观测（深度图+状态特征）
        返回:
            image_with_state: (H,W,2) 深度图+状态特征图像
        """
        # 1. 获取深度图像并归一化到0-255
        image = self.get_depth_image()  # 原始深度图（0-6550400.0，单位：厘米→转换为米）
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))  # 调整尺寸
        self.min_distance_to_obstacles = image.min()  # 记录最近障碍距离
        # 深度值映射：0→255（近），max_depth→0（远）
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # 2. 获取状态特征并转换为图像格式
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()  # 动力学模型状态特征
        state_feature_array[0, 0:self.state_feature_length] = state_feature  # 仅左上角填充状态

        # 3. 合并深度图和状态特征
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    def get_depth_gray_image(self):
        """
        获取深度图和灰度图（LGMD感知专用）
        返回:
            depth_meter: 深度图（米）
            img_gray: 灰度图
        """
        # 请求AirSim返回深度图和场景图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),  # 深度可视化图（浮点型）
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # 场景图（RGB）
        ])

        # 重试直到获取有效图像
        while responses[0].width == 0:
            print("获取图像失败，重试...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ])

        # 解析深度图（转换为米）
        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                  responses[0].height)
        depth_meter = depth_img * 100  # AirSim深度值单位为厘米→转换为米

        # 解析RGB图并转换为灰度图
        img_1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        return depth_meter, img_gray

    def get_depth_image(self):
        """
        仅获取深度图（用于vector/image感知）
        返回:
            depth_meter: 深度图（米）
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        ])

        # 重试直到获取有效图像
        while responses[0].width == 0:
            print("获取深度图失败，重试...")
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
            )
            print(123)
            time.sleep(0.5)

        # 解析深度图并转换为米
        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                  responses[0].height)
        depth_meter = depth_img * 100

        return depth_meter

    def get_obs_vector(self,P=0):
        """
        获取向量观测（深度图分割特征+状态特征）
        返回:
            feature_all: (1, N) 归一化的向量观测
        """
        # 获取深度图
        image = self.get_depth_image()
        self.min_distance_to_obstacles = image.min()

        # 深度图归一化并分割为5列，取每列最大值作为特征
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
                split_final.append(h_split_list[j].max())  # 每列最大值

        # 归一化图像特征和状态特征
        img_feature = np.array(split_final) / 255.0
        # state_feature = self.dynamic_model._get_state_feature() / 255
        state_feature = self.dynamic_model._get_state_feature(predicted_power = P) / 255
        # 合并特征
        feature_all = np.concatenate((img_feature, state_feature), axis=0)
        self.feature_all = feature_all
        # 调整形状为(1, N)
        feature_all = np.reshape(feature_all, (1, len(feature_all)))


        return feature_all

    def get_obs_lgmd(self):
        """
        获取LGMD视觉特征观测（模拟昆虫视觉神经）
        返回:
            feature_all: (1, N) LGMD特征+状态特征
        """
        # 获取深度图和灰度图
        depth_meter, img_gray = self.get_depth_gray_image()
        self.min_distance_to_obstacles = depth_meter.min()

        # 更新LGMD模型
        self.lgmd.update(img_gray)

        # 将LGMD输出分割为5列，计算每列归一化特征
        split_col_num = 5
        s_layer = self.lgmd.s_layer  # LGMD的S层输出 (192, 320)
        s_layer_split = np.hsplit(s_layer, split_col_num)  # 水平分割为5部分
        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192 * 64)  # 激活系数
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2  # 归一化到[-1,1]
            lgmd_out_list.append(lgmd_out_norm)

        # 可视化LGMD输出（热力图）
        heatmapshow = cv2.normalize(s_layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow('gray image', img_gray)
        cv2.imshow('depth image', np.clip(depth_meter, 0, 255) / 255)
        cv2.imshow('s-layer', heatmapshow)
        cv2.waitKey(1)  # 刷新显示

        # 对LGMD特征进行低通滤波（平滑）
        split_final = np.array(lgmd_out_list)
        filter_coeff = 0.8
        split_final_filter = filter_coeff * split_final + (1 - filter_coeff) * self.split_out_last
        self.split_out_last = split_final_filter

        # 合并LGMD特征和状态特征
        img_feature = np.array(split_final_filter)
        state_feature = self.dynamic_model._get_state_feature() / 255
        feature_all = np.concatenate((img_feature, state_feature), axis=0)
        self.feature_all = feature_all
        # 调整形状为(1, N)
        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    # ! ---------------------奖励计算模块-------------------------------------
    def compute_reward(self, done, action):
        """
        默认奖励函数（多旋翼）
        参数:
            done: 回合是否结束
            action: 执行的动作
        返回:
            reward: 单步奖励
        """
        reward = 0
        reward_reach = 10  # 到达目标奖励
        reward_crash = -20  # 碰撞惩罚
        reward_outside = -10  # 超出工作空间惩罚

        if not done:
            # 距离奖励（向目标移动则奖励，远离则惩罚）
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance * 500  # 归一化奖励
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # 偏航率惩罚（动作平滑性）
            yaw_speed_cost = 0.1 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # 3D导航时增加垂直速度和高度误差惩罚
                v_z_cost = 0.1 * ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.05 * (
                            (abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference) ** 2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            # 偏航误差惩罚
            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            # 回合结束时的奖励/惩罚
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_final(self, done, action):
        """
        最终版多旋翼奖励函数（优化惩罚项）
        """
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        # 不同环境调整距离奖励系数
        if self.env_name == 'NH_center':
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            # 1. 距离奖励
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2. 位置惩罚（偏离目标方向）
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y, z = current_pose
            x_g, y_g, z_g = goal_pose
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)
            punishment_pose = punishment_xy + punishment_z

            # 3. 障碍惩罚（近距离障碍）
            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            # 4. 动作惩罚（偏航率/垂直速度）
            punishment_action = 0
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = ((abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference) ** 2)
                punishment_action += (v_z_cost + z_err_cost)
            punishment_action += yaw_speed_cost

            # 5. 偏航误差惩罚
            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            # 总奖励 = 距离奖励 - 各项惩罚（加权）
            reward = reward_distance - 0.1 * punishment_pose - 0.2 * punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
        else:
            # 回合结束奖励/惩罚
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_wind(self, done, action, power, v_ground, v_wind):
        """
        融合能耗的多旋翼奖励函数：在原距离/惩罚项基础上，叠加能耗惩罚，鼓励低能耗飞行
        参数:
            done: 回合是否结束
            action: 执行的动作
            E: 单步能量消耗（J），由功率模型计算得到 E = P * dt
        返回:
            reward: 融合能耗的单步奖励
        """
        reward = 0
        reward_reach = 10  # 到达目标奖励
        reward_crash = -20  # 碰撞惩罚
        reward_outside = -10  # 超出工作空间惩罚

        # 气动智能体权重 (Aero Agent Weight)
        w_power = 0.5  # 功率惩罚权重
        w_wind = 2.0  # 借风奖励权重

        # ================= 1. 计算博弈门控系数 alpha =================
        # 使用 Sigmoid 或 线性函数，根据 SOC 动态调整
        # 当 SOC 高 (1.0) -> alpha 趋向 1 (偏向任务)
        # 当 SOC 低 (0.1) -> alpha 趋向 0 (偏向节能)
        soc = self.dynamic_model.soc
        alpha = 1.0 / (1.0 + np.exp(-10 * (soc - 0.4)))  # Sigmoid centered at SOC=0.4
        # 或者简单的线性： alpha = soc

        beta = 1.0 - alpha  # 节能权重

        # ========== 1. 环境适配：不同场景设置不同的能耗惩罚系数 ==========
        if self.env_name == 'NH_center':
            distance_reward_coef = 500
            energy_punish_coef = 0.02  # 能耗惩罚系数（可根据实际能耗量级调整）
            max_energy = 50  # 能耗上限（裁剪极端值）
        elif self.env_name in ['City', 'City_400']:
            distance_reward_coef = 50
            energy_punish_coef = 0.01
            max_energy = 30
        else:  # Trees/SimpleAvoid/Forest等
            distance_reward_coef = 50
            energy_punish_coef = 0.015
            max_energy = 40

        if not done:
            # ========== 2. 基础距离奖励（核心：向目标移动） ==========
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance
            self.previous_distance_from_des_point = distance_now

            # ========== 3. 位置惩罚（偏离目标方向） ==========
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y, z = current_pose
            x_g, y_g, z_g = goal_pose
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g) / 5, 0, 1)
            punishment_pose = punishment_xy + punishment_z

            # ========== 4. 障碍惩罚（近距离障碍） ==========
            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            # ========== 5. 动作惩罚（偏航率/垂直速度） ==========
            punishment_action = 0
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = ((abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference) ** 2)
                punishment_action += (v_z_cost + z_err_cost)
            punishment_action += yaw_speed_cost

            # ========== 6. 偏航误差惩罚 ==========
            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            # ================= 3. 气动奖励 (R_aero) - 专家知识 =================
            # A. 功率惩罚 (专家模型计算出的 P_total)
            # 归一化功率 (假设悬停功率约400W，最大1000W)
            p_norm = power / 200.0
            r_power = -w_power * p_norm

            # B. 借风奖励 (Wind Harvesting)
            # 计算地速与风速的点积： v . w = |v||w|cos(theta)
            # 正值表示顺风，负值表示逆风
            dot_product = np.dot(v_ground, v_wind)
            # 归一化：除以最大速度和风速的积，或者直接用数值
            r_wind_harvest = w_wind * np.tanh(dot_product / 25.0)  # tanh限制在 -1 到 1

            # C. 专家效率比 (Efficiency Ratio)
            # 简单的悬停功率估算 (mg)^1.5 / ...
            # 如果当前功率 P_total 接近理论悬停功率 P_hover，说明效率高
            # r_efficiency = P_hover / (P_total + 1e-5)

            R_aero = r_power + r_wind_harvest

            # ================= 4. 融合 (博弈) =================
            # 混合奖励
            # print("效率奖励：",reward_distance)
            # print("能源奖励：",R_aero)
            reward = alpha * reward_distance + beta * R_aero

            # ========== 8. 总奖励：距离奖励 - 各项惩罚（加权） ==========
            reward = (
                    reward
                    - 0.1 * punishment_pose  # 位置惩罚（加权0.1）
                    - 0.2 * punishment_obs  # 障碍惩罚（加权0.2）
                    - 0.1 * punishment_action  # 动作惩罚（加权0.1）
                    - 0.5 * yaw_error_cost  # 偏航误差惩罚（加权0.5）
            )



        else:
            # ========== 9. 回合结束奖励/惩罚（保持原有逻辑） ==========
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward


    def compute_reward_final_fixedwing(self, done, action):
        """
        固定翼专用最终奖励函数
        """
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            # 1. 距离奖励
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = 300 * (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2. 位置惩罚（仅XY平面）
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y = current_pose[0], current_pose[1]
            x_g, y_g = goal_pose[0], goal_pose[1]
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 50, 0, 1)
            punishment_pose = punishment_xy

            # 3. 障碍惩罚
            if self.min_distance_to_obstacles < 20:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 15, 0, 1)
            else:
                punishment_obs = 0

            # 4. 动作惩罚（滚转率）
            punishment_action = abs(action[0]) / self.dynamic_model.roll_rate_max

            # 5. 偏航误差惩罚
            yaw_error = self.dynamic_model.state_raw[1]
            yaw_error_cost = abs(yaw_error / 90)

            # 总奖励
            reward = reward_distance - 0.1 * punishment_pose - 0.5 * punishment_obs - 0.1 * punishment_action - 0.1 * yaw_error_cost
        else:
            # 回合结束奖励/惩罚
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_test(self, done, action):
        """
        测试用奖励函数（简化版）
        """
        reward = 0
        reward_reach = 10
        reward_crash = -100
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance * 100
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0
            yaw_speed_cost = 0.1 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.1 * abs(action[1]) / self.dynamic_model.v_z_max
                z_err_cost = 0.05 * abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost
            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        """
        固定翼基础奖励函数
        """
        reward = 0
        reward_reach = 10
        reward_crash = -50
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance * 300
            self.previous_distance_from_des_point = distance_now

            action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max
            yaw_error_deg = self.dynamic_model.state_raw[1]
            yaw_error_cost = 0.1 * abs(yaw_error_deg / 180)

            reward = reward_distance - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                yaw_error_deg = self.dynamic_model.state_raw[1]
                reward = reward_reach * (1 - abs(yaw_error_deg / 180))
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        """
        多旋翼新版奖励函数（侧重偏航和障碍）
        """
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (
                                          self.previous_distance_from_des_point - distance_now) / self.dynamic_model.goal_distance * 5
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = self.dynamic_model.state_raw[1]
            relative_yaw_cost = abs(yaw_error_deg / 180)
            action_cost = abs(action[1]) / self.dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 1 - (self.min_distance_to_obstacles - self.crash_distance) / (
                            obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost ** 2
            reward = - (2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self.is_in_desired_pose():
                reward = reward_reach * (1 - abs(self.dynamic_model.state_norm[1]))
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        """
        含动作成本的奖励函数（惩罚速度偏差）
        """
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                              self.dynamic_model.goal_distance * 10
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # 速度惩罚（偏离巡航速度5m/s）
            v_xy_cost = 0.02 * abs(action[0] - 5) / 4
            yaw_rate_cost = 0.02 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += (v_xy_cost + yaw_rate_cost)

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        """
        模仿LQR的奖励函数（二次惩罚）
        """
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            # 偏航率二次惩罚
            yaw_speed_cost = 0.2 * ((action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2)

            if self.dynamic_model.navigation_3d:
                # 垂直速度和高度误差二次惩罚
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max) ** 2)
                z_err_cost = 0.1 * ((self.dynamic_model.state_raw[1] / self.dynamic_model.max_vertical_difference) ** 2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            # 偏航误差裁剪后二次惩罚
            yaw_error_clip = min(max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip ** 2)

            reward = - (action_cost + yaw_error_cost)
        else:
            if self.is_in_desired_pose():
                # 到达目标后根据偏航误差调整奖励
                yaw_error_clip = min(max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip ** 2)
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    # ! ------------------ 回合终止条件判断模块-----------------------------------------------
    def is_done(self):
        """
        判断回合是否结束
        返回:
            episode_done: True/False
        """
        episode_done = False

        # 检查终止条件：超出工作空间/到达目标/碰撞/步数超限
        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose = self.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed()

        episode_done = is_not_inside_workspace_now or has_reached_des_pose or too_close_to_obstable or self.step_num >= self.max_episode_steps

        return episode_done

    def is_not_inside_workspace(self):
        """
        检查无人机是否超出工作空间
        返回:
            is_not_inside: True/False
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()
        # print(current_position)

        # 检查X/Y/Z是否在工作空间范围内
        if (current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or
                current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]):
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        """
        检查是否到达目标位置（距离小于接受半径）
        返回:
            in_desired_pose: True/False
        """
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        """
        检查是否碰撞（AirSim碰撞检测+近距离障碍）
        返回:
            is_crashed: True/False
        """
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()  # AirSim碰撞信息
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True
        #     return is_crashed  # 物理碰撞直接返回
        #
        #     # 2. 视觉避障检测 (基于深度图)
        #     # 修正：在前 10 步忽略视觉距离判断，防止起飞倾斜或相机初始化噪声导致的误判
        # if self.step_num > 10:
        #     if self.min_distance_to_obstacles < self.crash_distance:
        #         # 打印调试信息，确认是视觉误判
        #         # print(f"Visual Crash detected: Dist {self.min_distance_to_obstacles:.2f} < Threshold {self.crash_distance}")
        #         is_crashed = True

        return is_crashed

    # ! ----------- 工具函数模块-------------------------------------------
    def get_distance_to_goal_3d(self):
        """
        计算无人机到目标的3D欧氏距离
        返回:
            distance: 3D距离（米）
        """
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        计算点到直线的距离（用于位置惩罚）
        参数:
            pointX/Y: 点坐标
            lineX1/Y1: 直线起点
            lineX2/Y2: 直线终点
        返回:
            dis: 点到直线的垂直距离
        '''
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

        return dis

    # ! ----------- 可视化/日志模块------------------------------------------------------------------
    def print_train_info_airsim(self, action, obs, reward, info):
        """
        打印训练信息到AirSim仿真器日志
        """
        msg_train_info = f"Episode: {self.episode_num}  Step: {self.step_num}  Total Steps: {self.total_step}"

        # Send log messages to AirSim
        self.client.simPrintLogMessage('Training Info: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('Reward: ',
                                       f"{reward:4.4f}  Cumulative Reward: {self.cumulated_episode_reward:4.4f}")
        self.client.simPrintLogMessage('Additional Info: ', str(info))
        self.client.simPrintLogMessage('Normalized State: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage('Raw State: ', str(self.dynamic_model.state_raw))
        self.client.simPrintLogMessage('Nearest Obstacle Distance: ', str(self.min_distance_to_obstacles))

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        发送固定翼可视化信号
        """
        step = int(self.total_step)
        # 动作转换（弧度→度）
        action_plot = np.array([10, 0, math.degrees(action[0])])
        # 状态转换（适配可视化格式）
        state = self.dynamic_model.state_raw
        state_output = np.array([state[0], 0, state[1], 10, 0, state[2]])

        # 发射信号
        self.action_signal.emit(step, action_plot)
        self.state_signal.emit(step, state_output)
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude()),
                                  np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position),
                              np.asarray(self.dynamic_model.start_position),
                              np.asarray(self.dynamic_model.get_position()),
                              np.asarray(self.trajectory_list))
        self.lgmd_signal.emit(self.min_distance_to_obstacles, 0, self.feature_all[:-1])

    def set_pyqt_signal_multirotor(self, action, reward):
        """
        发送多旋翼可视化信号
        """
        step = int(self.total_step)

        # 2D→3D状态/动作转换（适配可视化）
        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            action_output = np.array([action[0], 0, action[1]])
            state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])

        # 发射信号
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
        '''
        记录Q值地图（用于可视化策略分布）
        参数:
            q_value: 当前状态-动作对的Q值
            action: 执行的动作
            reward: 单步奖励
        '''
        # 初始化Q值地图（9层：Q值/动作0/动作1/步数/奖励/周期Q值/周期动作0/周期动作1/周期奖励）
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x + 1, map_size_y + 1), np.nan)

        # 获取当前位置并转换为地图索引
        position = self.dynamic_model.get_position()
        pose_x, pose_y = position[0], position[1]
        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # 记录Q值/动作/步数/奖励到地图
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

        # 定期保存Q值地图
        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step + 1) % record_step == 0:
            if self.data_path is not None:
                np.save(self.data_path + f'/q_value_map_{self.total_step + 1}', self.q_value_map)
                # 重置周期数据层
                self.q_value_map[5:, :, :] = np.nan
