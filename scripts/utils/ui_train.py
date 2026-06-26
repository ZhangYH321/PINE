'''
@Author: Lei He
@Date: 2020-06-01 22:52:40
LastEditTime: 2022-12-02 17:30:59
@Description:
@Github: https://github.com/heleidsn
'''
import sys
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets,QtGui
from PIL import Image
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import seaborn as sns

from configparser import ConfigParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrainingUi(QWidget):
    """PyQt5 GUI used to show the data during training

    Args:
        QWidget (_type_): _description_
    """

    def __init__(self, config):
        super(TrainingUi, self).__init__()
        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)

        # 新增：风场显示数据初始化
        self.wind_display_data = None
        self.wind_arrows_initialized = False
        self.wind_arrows = []  # 存储箭头对象
        self.wind_texts = []  # 存储风速文本
        self.pending_wind_data = None  # 待处理的风场数据
        self.pending_map_range = None  # 待处理的地图范围

        self.init_ui()

        # 使用定时器延迟初始化风场箭头，避免阻塞UI
        self.init_timer = QTimer()
        self.init_timer.timeout.connect(self.delayed_wind_init)
        self.init_timer.setSingleShot(True)
        self.init_timer.start(500)  # 延迟500ms初始化风场

    def init_ui(self):
        """ init UI
            Include four parts:
                Action
                State
                Attitude
                Reward and trajectory
        """
        self.setWindowTitle("Training UI")
        # self.showFullScreen()

        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')  # best performance

        self.max_len = 100
        self.ptr = -self.max_len

        # action, state_feature, attitude, trajectory
        self.dynamics = self.cfg.get('options', 'dynamic_name')
        if self.dynamics == 'SimpleFixedwing':
            action_plot_gb = self.create_actionPlot_groupBox_fixed_wing()
        else:
            action_plot_gb = self.create_actionPlot_groupBox_multirotor()
        state_plot_gb = self.create_state_plot_groupbox()
        attitude_plot_gb = self.create_attitude_plot_groupbox()
        reward_plot_gb = self.create_reward_plot_groupbox()
        lgmd_plot_gb = self.create_lgmd_plot_groupbox()
        traj_plot_gb = self.create_traj_plot_groupbox()

        right_widget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addWidget(reward_plot_gb)

        if self.cfg.has_option('options', 'perception'):
            if self.cfg.get('options', 'perception') == 'lgmd':
                vlayout.addWidget(lgmd_plot_gb)

        vlayout.addWidget(traj_plot_gb)
        right_widget.setLayout(vlayout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(action_plot_gb)
        main_layout.addWidget(state_plot_gb)
        main_layout.addWidget(attitude_plot_gb)
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)

        self.pen_red = pg.mkPen(color='r', width=2)    # used for cmd
        self.pen_blue = pg.mkPen(color='b', width=1)   # used for real data
        self.pen_green = pg.mkPen(color='g', width=2)

    def update_value_list(self, list, value):
        '''
        @description: update value list for plot
        @param {type}
        @return:
        '''
        list[:-1] = list[1:]
        list[-1] = float(value)
        return list

# action plot groupbox
    def create_actionPlot_groupBox_multirotor(self):
        """ groupbox for multirotor action plot
            For each action, cmd and real state should be ploted
            Action:
                v_xy, v_z, yaw_rate
        """
        actionPlotGroupBox = QGroupBox('Action (multirotor)')
        self.v_xy_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_xy_real_list = np.linspace(0, 0, self.max_len)

        self.v_z_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_z_real_list = np.linspace(0, 0, self.max_len)

        self.yaw_rate_cmd_list = np.linspace(0, 0, self.max_len)
        self.yaw_rate_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.plotWidget_v_xy = pg.PlotWidget(title='v_xy (m/s)')
        self.plotWidget_v_xy.setYRange(max=self.cfg.getfloat(
            'multirotor', 'v_xy_max'), min=self.cfg.getfloat('multirotor', 'v_xy_min'))
        self.plotWidget_v_xy.showGrid(x=True, y=True)
        self.plot_v_xy = self.plotWidget_v_xy.plot()      # get plot object
        self.plot_v_xy_cmd = self.plotWidget_v_xy.plot()

        self.plotWidget_v_z = pg.PlotWidget(title='v_z (m/s)')
        self.plotWidget_v_z.setYRange(max=self.cfg.getfloat(
            'multirotor', 'v_z_max'), min=-self.cfg.getfloat('multirotor', 'v_z_max'))
        self.plotWidget_v_z.showGrid(x=True, y=True)
        self.plot_v_z = self.plotWidget_v_z.plot()
        self.plot_v_z_cmd = self.plotWidget_v_z.plot()

        self.plotWidget_yaw_rate = pg.PlotWidget(title='yaw_rate (deg/s)')
        self.plotWidget_yaw_rate.setYRange(max=self.cfg.getfloat(
            'multirotor', 'yaw_rate_max_deg'), min=-self.cfg.getfloat('multirotor', 'yaw_rate_max_deg'))
        self.plotWidget_yaw_rate.showGrid(x=True, y=True)
        self.plot_yaw_rate = self.plotWidget_yaw_rate.plot()
        self.plot_yaw_rate_cmd = self.plotWidget_yaw_rate.plot()

        layout.addWidget(self.plotWidget_v_xy)
        layout.addWidget(self.plotWidget_v_z)
        layout.addWidget(self.plotWidget_yaw_rate)

        actionPlotGroupBox.setLayout(layout)

        return actionPlotGroupBox

    def create_actionPlot_groupBox_fixed_wing(self):
        """
        state and action groupbox
        state: roll, pitch, yaw, airspeed
        action: roll_cmd, pitch_cmd, yaw_cmd, airspeed_cmd
        """
        actionPlotGroupBox = QGroupBox('Real time action and state')

        self.v_xy_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_xy_real_list = np.linspace(0, 0, self.max_len)

        self.v_z_cmd_list = np.linspace(0, 0, self.max_len)
        self.v_z_real_list = np.linspace(0, 0, self.max_len)

        self.roll_list_a = np.linspace(0, 0, self.max_len)
        self.roll_cmd_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.plotWidget_v_xy = pg.PlotWidget(title='v_xy (m/s)')
        self.plotWidget_v_xy.setYRange(max=self.cfg.getfloat(
            'fixedwing', 'v_xy_max'), min=self.cfg.getfloat('fixedwing', 'v_xy_min'))
        self.plotWidget_v_xy.showGrid(x=True, y=True)
        self.plot_v_xy = self.plotWidget_v_xy.plot()      # get plot object
        self.plot_v_xy_cmd = self.plotWidget_v_xy.plot()

        self.plotWidget_v_z = pg.PlotWidget(title='v_z (m/s)')
        self.plotWidget_v_z.setYRange(max=self.cfg.getfloat(
            'fixedwing', 'v_z_max'), min=-self.cfg.getfloat('fixedwing', 'v_z_max'))
        self.plotWidget_v_z.showGrid(x=True, y=True)
        self.plot_v_z = self.plotWidget_v_z.plot()
        self.plot_v_z_cmd = self.plotWidget_v_z.plot()

        self.plotWidget_roll = pg.PlotWidget(title='roll (deg)')
        self.plotWidget_roll.setYRange(max=45, min=-45)
        self.plotWidget_roll.showGrid(x=True, y=True)
        self.plot_roll_a = self.plotWidget_roll.plot()
        self.plot_roll_cmd_a = self.plotWidget_roll.plot()

        layout.addWidget(self.plotWidget_v_xy)
        layout.addWidget(self.plotWidget_v_z)
        layout.addWidget(self.plotWidget_roll)

        actionPlotGroupBox.setLayout(layout)

        return actionPlotGroupBox

    def action_cb(self, step, action):
        if self.dynamics == 'SimpleFixedwing':
            self.action_cb_fixed_wing(step, action)
        else:
            self.action_cb_multirotor(step, action)

    def action_cb_multirotor(self, step, action):
        self.update_value_list(self.v_xy_cmd_list, action[0])
        self.update_value_list(self.v_z_cmd_list, action[1])
        self.update_value_list(self.yaw_rate_cmd_list, math.degrees(action[2]))

        self.plot_v_xy_cmd.setData(self.v_xy_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_yaw_rate_cmd.setData(
            self.yaw_rate_cmd_list, pen=self.pen_blue)

    def action_cb_fixed_wing(self, step, action):
        """
        call back function used for fixed wing plot
        """
        self.update_value_list(self.v_xy_cmd_list, action[0])
        self.update_value_list(self.v_z_cmd_list, action[1])
        self.update_value_list(self.roll_cmd_list, action[2])

        # plot action
        self.plot_v_xy_cmd.setData(self.v_xy_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_roll_cmd_a.setData(self.roll_cmd_list, pen=self.pen_blue)

# state feature plot groupbox
    def create_state_plot_groupbox(self):
        state_plot_groupbox = QGroupBox(title='State feature')

        self.distance_list = np.linspace(0, 0, self.max_len)
        self.vertical_dis_list = np.linspace(0, 0, self.max_len)
        self.relative_yaw_list = np.linspace(0, 0, self.max_len)

        layout = QVBoxLayout()

        self.pw1 = pg.PlotWidget(title='distance_xy (m)')
        self.pw1.showGrid(x=True, y=True)
        self.p1 = self.pw1.plot()

        self.pw2 = pg.PlotWidget(title='distance_z (m)')
        self.pw2.showGrid(x=True, y=True)
        self.p2 = self.pw2.plot()

        self.pw3 = pg.PlotWidget(title='relative yaw (deg)')
        self.pw3.showGrid(x=True, y=True)
        self.p3 = self.pw3.plot()

        layout.addWidget(self.pw1)
        layout.addWidget(self.pw2)
        layout.addWidget(self.pw3)

        state_plot_groupbox.setLayout(layout)
        return state_plot_groupbox

    def state_cb(self, step, state_raw):

        # update state
        self.update_value_list(self.distance_list, state_raw[0])
        self.update_value_list(self.vertical_dis_list, state_raw[1])
        self.update_value_list(self.relative_yaw_list, state_raw[2])

        self.p1.setData(self.distance_list, pen=self.pen_red)
        self.p2.setData(self.vertical_dis_list, pen=self.pen_red)
        self.p3.setData(self.relative_yaw_list, pen=self.pen_red)

        # update action real
        self.update_value_list(self.v_xy_real_list, state_raw[3])
        self.update_value_list(self.v_z_real_list, state_raw[4])
        self.plot_v_xy.setData(self.v_xy_real_list, pen=self.pen_red)
        self.plot_v_z.setData(self.v_z_real_list, pen=self.pen_red)

        if self.dynamics == 'SimpleFixedwing':
            self.update_value_list(self.roll_list_a, state_raw[5])
            self.plot_roll_a.setData(self.roll_list_a, pen=self.pen_red)
        else:
            self.update_value_list(self.yaw_rate_list, state_raw[5])
            self.plot_yaw_rate.setData(self.yaw_rate_list, pen=self.pen_red)

# attitude plot groupbox
    def create_attitude_plot_groupbox(self):
        plot_gb = QGroupBox(title='Attitude')
        layout = QVBoxLayout()

        self.roll_list = np.linspace(0, 0, self.max_len)
        self.roll_cmd_list = np.linspace(0, 0, self.max_len)

        self.pitch_list = np.linspace(0, 0, self.max_len)
        self.pitch_cmd_list = np.linspace(0, 0, self.max_len)

        self.yaw_list = np.linspace(0, 0, self.max_len)
        self.yaw_cmd_list = np.linspace(0, 0, self.max_len)

        self.pw_roll = pg.PlotWidget(title='roll (deg)')
        self.pw_roll.setYRange(max=45, min=-45)
        self.pw_roll.showGrid(x=True, y=True)
        self.plot_roll = self.pw_roll.plot()
        self.plot_roll_cmd = self.pw_roll.plot()

        self.pw_pitch = pg.PlotWidget(title='pitch (deg)')
        self.pw_pitch.setYRange(max=25, min=-25)
        self.pw_pitch.showGrid(x=True, y=True)
        self.plot_pitch = self.pw_pitch.plot()
        self.plot_pitch_cmd = self.pw_pitch.plot()

        self.pw_yaw = pg.PlotWidget(title='yaw (deg)')
        # self.pw_yaw.setYRange(max=180, min=-180)
        self.pw_yaw.showGrid(x=True, y=True)
        self.plot_yaw = self.pw_yaw.plot()
        self.plot_yaw_cmd = self.pw_yaw.plot()

        layout.addWidget(self.pw_roll)
        layout.addWidget(self.pw_pitch)
        layout.addWidget(self.pw_yaw)

        plot_gb.setLayout(layout)
        return plot_gb

    def attitude_plot_cb(self, step, attitude, attitude_cmd):
        """ plot attitude (pitch, roll, yaw) and the cmd data
        """
        self.update_value_list(self.pitch_list, math.degrees(attitude[0]))
        self.update_value_list(self.roll_list, math.degrees(attitude[1]))
        self.update_value_list(self.yaw_list, math.degrees(attitude[2]))

        self.plot_pitch.setData(self.pitch_list, pen=self.pen_red)
        self.plot_roll.setData(self.roll_list, pen=self.pen_red)
        self.plot_yaw.setData(self.yaw_list, pen=self.pen_red)

# reward plot groupbox
    def create_reward_plot_groupbox(self):
        reward_plot_groupbox = QGroupBox(title='Reward')
        layout = QHBoxLayout()
        # reward_plot_groupbox.setFixedHeight(300)
        reward_plot_groupbox.setFixedWidth(600)

        self.reward_list = np.linspace(0, 0, self.max_len)
        self.total_reward_list = np.linspace(0, 0, self.max_len)

        self.rw_pw_1 = pg.PlotWidget(title='reward')
        self.rw_pw_1.showGrid(x=True, y=True)
        self.rw_p_1 = self.rw_pw_1.plot()

        self.rw_pw_2 = pg.PlotWidget(title='total reward')
        self.rw_pw_2.showGrid(x=True, y=True)
        self.rw_p_2 = self.rw_pw_2.plot()

        layout.addWidget(self.rw_pw_1)
        layout.addWidget(self.rw_pw_2)

        reward_plot_groupbox.setLayout(layout)
        return reward_plot_groupbox

    def reward_plot_cb(self, step, reward, total_reward):
        self.update_value_list(self.reward_list, reward)
        self.update_value_list(self.total_reward_list, total_reward)

        self.rw_p_1.setData(self.reward_list, pen=self.pen_red)
        self.rw_p_2.setData(self.total_reward_list, pen=self.pen_red)

# lgmd plot groupbox
    def create_lgmd_plot_groupbox(self):
        lgmd_plot_groupbox = QGroupBox(title='Lgmd info')
        layout = QHBoxLayout()
        # lgmd_plot_groupbox.setFixedHeight(300)
        lgmd_plot_groupbox.setFixedWidth(600)

        self.min_dist_to_obs_list = np.linspace(0, 0, self.max_len)
        self.lgmd_out_list = np.linspace(0, 0, self.max_len)

        self.lgmd_pw_1 = pg.PlotWidget(title='min distance to obs')
        self.lgmd_pw_1.showGrid(x=True, y=True)
        self.lgmd_p_1 = self.lgmd_pw_1.plot()

        self.lgmd_pw_2 = pg.PlotWidget(title='lgmd output')
        self.lgmd_pw_2.showGrid(x=True, y=True)
        self.lgmd_p_2 = self.lgmd_pw_2.plot()

        self.lgmd_pw_3 = MatplotlibWidget()
        # self.lgmd_pw_3.showGrid(x=True,y=True)
        # self.lgmd_p_3 = self.lgmd_pw_3.plot()

        layout.addWidget(self.lgmd_pw_1)
        layout.addWidget(self.lgmd_pw_2)
        # layout.addWidget(self.lgmd_pw_3)

        lgmd_plot_groupbox.setLayout(layout)
        return lgmd_plot_groupbox

    def lgmd_plot_cb(self, min_dist, lgmd_out, lgmd_split):
        self.update_value_list(self.min_dist_to_obs_list, min_dist)
        self.update_value_list(self.lgmd_out_list, lgmd_out)

        self.lgmd_p_1.setData(self.min_dist_to_obs_list, pen=self.pen_red)
        self.lgmd_p_2.setData(self.lgmd_out_list, pen=self.pen_red)

        # add feature bar
        # x = np.arange(len(lgmd_split))
        # self.lgmd_pw_3.getFigure().clf()
        # subplot1 = self.lgmd_pw_3.getFigure().add_subplot(111)
        # sns.barplot(x=x, y=lgmd_split, ax=subplot1)
        # subplot1.set(title='lgmd out split')
        # self.lgmd_pw_3.draw()

# trajectory plot groupbox
    def create_traj_plot_groupbox(self):
        traj_plot_groupbox = QGroupBox('Trajectory')
        traj_plot_groupbox.setFixedHeight(600)
        traj_plot_groupbox.setFixedWidth(600)
        layout = QVBoxLayout()

        self.traj_pw = pg.PlotWidget(title='trajectory')
        self.traj_pw.showGrid(x=True, y=True)
        # self.traj_pw.setXRange(max=350, min=-100)
        # self.traj_pw.setYRange(max=100, min=-300)
        self.traj_pw.setXRange(max=140, min=-140)
        self.traj_pw.setYRange(max=140, min=-140)
        self.traj_plot = self.traj_pw.plot()
        self.traj_pw.invertY()

        if self.cfg.get('options', 'env_name') == 'SimpleAvoid':
            background_image_path = PROJECT_ROOT / 'resources' / 'env_maps' / 'simple_world_light.png'
            img_data = Image.open(background_image_path)
            image = np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            # make sure image is behind other data
            self.background_img.setZValue(-100)
            self.background_img.setRect(pg.QtCore.QRectF(-60, -60, 120, 120))
            self.traj_pw.setXRange(max=60, min=-60)
            self.traj_pw.setYRange(max=60, min=-60)
        elif self.cfg.get('options', 'env_name') == 'NH_center':
            background_image_path = PROJECT_ROOT / 'resources' / 'env_maps' / 'NH_center.png'
            img_data = Image.open(background_image_path)
            image = np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            # make sure image is behind other data
            self.background_img.setZValue(-100)
            self.background_img.setRect(pg.QtCore.QRectF(-135, -135, 270, 270))
            self.traj_pw.setXRange(max=135, min=-135)
            self.traj_pw.setYRange(max=135, min=-135)
        elif self.cfg.get('options', 'env_name') == 'City_400':
            background_image_path = PROJECT_ROOT / 'resources' / 'env_maps' / 'city_400.png'
            img_data = Image.open(background_image_path)
            image = np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            # make sure image is behind other data
            self.background_img.setZValue(-100)
            self.background_img.setRect(pg.QtCore.QRectF(-220, -220, 440, 440))
            self.traj_pw.setXRange(max=220, min=-220)
            self.traj_pw.setYRange(max=220, min=-220)
        elif self.cfg.get('options', 'env_name') == 'Tree_200':
            background_image_path = PROJECT_ROOT / 'resources' / 'env_maps' / 'trees_200_200.png'
            img_data = Image.open(background_image_path)
            image = np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            # make sure image is behind other data
            self.background_img.setZValue(-100)
            self.background_img.setRect(pg.QtCore.QRectF(-100, -100, 200, 200))
            self.traj_pw.setXRange(max=100, min=-100)
            self.traj_pw.setYRange(max=100, min=-100)
        elif self.cfg.get('options', 'env_name') == 'Forest':
            background_image_path = PROJECT_ROOT / 'resources' / 'env_maps' / 'Forest.png'
            img_data = Image.open(background_image_path)
            image = np.copy(img_data)
            self.background_img = pg.ImageItem(image)
            self.traj_pw.addItem(self.background_img)
            # make sure image is behind other data
            self.background_img.setZValue(-100)
            self.background_img.setRect(pg.QtCore.QRectF(-100, -100, 200, 200))
            self.traj_pw.setXRange(max=100, min=-100)
            self.traj_pw.setYRange(max=100, min=-100)
        elif self.cfg.get('options', 'env_name') == 'City':
            self.traj_pw.setXRange(max=300, min=0)
            self.traj_pw.setYRange(max=0, min=-300)

        layout.addWidget(self.traj_pw)

        traj_plot_groupbox.setLayout(layout)
        return traj_plot_groupbox

    def init_wind_arrows(self, wind_data):
        """
        初始化风场箭头（只调用一次）
        :param wind_data: (M, N, 3) 的风场数组
        """
        if wind_data is None or self.wind_arrows_initialized:
            return

        M, N, _ = wind_data.shape
        self.wind_arrows = []  # 存储箭头对象

        # 获取地图范围（从配置文件或硬编码）
        if self.cfg.get('options', 'env_name') == 'NH_center':
            map_range_xy = [-140, 140, -140, 140]
        else:
            # 默认范围
            map_range_xy = [-140, 140, -140, 140]

        min_x, max_x, min_y, max_y = map_range_xy
        step_x = (max_x - min_x) / M
        step_y = (max_y - min_y) / N

        print(f"绘制风场箭头: {M}x{N} 网格")

        for i in range(M):
            for j in range(N):
                # 获取网格风速向量
                wind_N, wind_E, _ = wind_data[i, j]
                wind_speed = np.sqrt(wind_N ** 2 + wind_E ** 2)

                # 风速太小则不显示箭头
                if wind_speed < 0.1:
                    continue

                # 计算网格中心坐标
                center_x = min_x + (i + 0.5) * step_x
                center_y = min_y + (j + 0.5) * step_y

                # 计算风向角度（从正北方向顺时针计算）
                wind_angle = np.arctan2(wind_E, wind_N)  # 弧度

                # 根据风速调整箭头大小
                arrow_size = min(8, max(3, wind_speed * 2))

                # 创建红色箭头
                arrow = pg.ArrowItem(
                    angle=180 - np.degrees(wind_angle),  # PyQtGraph角度调整
                    tipAngle=25,  # 箭头尖端角度
                    headLen=arrow_size,
                    tailLen=0,
                    pen=pg.mkPen(color='r', width=1.5),
                    brush=pg.mkBrush('r')
                )
                arrow.setPos(center_x, center_y)
                arrow.setZValue(-50)  # 确保箭头在背景之上，但在轨迹之下
                self.traj_pw.addItem(arrow)
                self.wind_arrows.append(arrow)

                # 可选：显示风速数值（小字体）
                if wind_speed > 1.0:  # 只显示较大风速
                    text = pg.TextItem(
                        text=f'{wind_speed:.1f}',
                        color=(200, 0, 0, 180),  # 半透明红色
                        anchor=(0.5, 0.5)
                    )
                    text.setFont(QtGui.QFont('Arial', 7))
                    text.setPos(center_x, center_y - 4)
                    text.setZValue(-49)
                    self.traj_pw.addItem(text)
                    self.wind_arrows.append(text)

        self.wind_arrows_initialized = True
        print(f"风场箭头初始化完成，共绘制 {len(self.wind_arrows)} 个箭头")

    # 新增：接收风场显示数据的槽函数
    def set_wind_display_data(self, wind_data):
        """
        接收风场显示数据并初始化箭头
        """
        if not self.wind_arrows_initialized and wind_data is not None:
            self.wind_display_data = wind_data
            self.init_wind_arrows(wind_data)

    # def traj_plot_cb(self, goal, start, current_pose, trajectory_list):
    #     """
    #     Plot trajectory
    #     """
    #     # clear plot
    #     self.traj_pw.clear()
    #
    #     # set background image
    #     background_list = ['SimpleAvoid', 'NH_center',
    #                        'City_400', 'Tree_200', 'Forest']
    #     if self.cfg.get('options', 'env_name') in background_list:
    #         self.traj_pw.addItem(self.background_img)
    #
    #
    #
    #     # plot start, goal and trajectory
    #     self.traj_pw.plot([start[0]], [start[1]], symbol='o')
    #     self.traj_pw.plot([goal[0]], [goal[1]], symbol='o')
    #     self.traj_pw.plot(
    #         trajectory_list[..., 0], trajectory_list[..., 1], pen=self.pen_red)
    def delayed_wind_init(self):
        """延迟初始化风场箭头"""
        if self.cfg.getboolean('options', 'use_wind', fallback=False):
            self.init_wind_arrows_from_config()

    def traj_plot_cb(self, goal, start, current_pose, trajectory_list):
        """
        Plot trajectory - 优化版本：避免频繁清空和重绘
        """
        # 方法1：使用视图矩形来更新轨迹，而不清除所有项目
        # 我们只更新轨迹线，不清除风场箭头

        # 检查是否已经有轨迹线
        if hasattr(self, 'traj_line'):
            # 更新现有轨迹线
            if len(trajectory_list) > 0:
                self.traj_line.setData(trajectory_list[..., 0], trajectory_list[..., 1])
        else:
            # 创建新的轨迹线
            if len(trajectory_list) > 0:
                self.traj_line = self.traj_pw.plot(
                    trajectory_list[..., 0], trajectory_list[..., 1],
                    pen=self.pen_red, name='trajectory')

        # 更新起点和终点标记
        if hasattr(self, 'start_point'):
            self.start_point.setData([start[0]], [start[1]])
        else:
            self.start_point = self.traj_pw.plot([start[0]], [start[1]],
                                                 pen=None, symbol='o',
                                                 symbolPen=None, symbolBrush='g',
                                                 name='start')

        if hasattr(self, 'goal_point'):
            self.goal_point.setData([goal[0]], [goal[1]])
        else:
            self.goal_point = self.traj_pw.plot([goal[0]], [goal[1]],
                                                pen=None, symbol='o',
                                                symbolPen=None, symbolBrush='r',
                                                name='goal')

        # 更新当前无人机位置
        if hasattr(self, 'drone_point'):
            self.drone_point.setData([current_pose[0]], [current_pose[1]])
        else:
            self.drone_point = self.traj_pw.plot([current_pose[0]], [current_pose[1]],
                                                 pen=None, symbol='x',
                                                 symbolPen='b', symbolBrush='b', symbolSize=10,
                                                 name='drone')

    def init_wind_arrows_from_config(self):
        """
        从配置文件初始化风场箭头 - 优化版本：使用更少的箭头
        """
        if not self.cfg.getboolean('options', 'use_wind', fallback=False):
            return

        # 获取地图范围
        env_name = self.cfg.get('options', 'env_name')
        if env_name == 'NH_center':
            map_range_xy = [-140, 140, -140, 140]
            # 减少网格密度以提高性能
            M, N = 10, 10  # 从20x20减少到10x10
        elif env_name == 'SimpleAvoid':
            map_range_xy = [-60, 60, -60, 60]
            M, N = 5, 5  # 从10x10减少到5x5
        elif env_name in ['Tree_200', 'Forest']:
            map_range_xy = [-100, 100, -100, 100]
            M, N = 8, 8  # 从15x15减少到8x8
        elif env_name == 'City_400':
            map_range_xy = [-220, 220, -220, 220]
            M, N = 12, 12  # 从25x25减少到12x12
        else:
            return

        # 设置随机种子
        seed = self.cfg.getint('wind', 'seed', fallback=42)
        np.random.seed(seed)

        min_x, max_x, min_y, max_y = map_range_xy
        step_x = (max_x - min_x) / M
        step_y = (max_y - min_y) / N

        print(f"初始化风场箭头: {M}x{N} 网格")

        # 使用更简单的箭头以提高性能
        for i in range(M):
            for j in range(N):
                # 随机生成风速大小（0~4m/s）
                wind_speed = np.random.uniform(0, 4)
                # 随机生成风向（0~2π弧度）
                wind_dir = np.random.uniform(0, 2 * np.pi)

                # 风速太小则不显示箭头
                if wind_speed < 0.5:  # 提高阈值，减少箭头数量
                    continue

                # 计算网格中心坐标
                center_x = min_x + (i + 0.5) * step_x
                center_y = min_y + (j + 0.5) * step_y

                # 根据风速调整箭头大小
                arrow_size = min(6, max(2, wind_speed * 1.5))  # 减小箭头大小

                # 创建更简单的箭头（使用线段而不是复杂箭头）
                # 计算箭头端点
                end_x = center_x + arrow_size * np.cos(wind_dir)
                end_y = center_y + arrow_size * np.sin(wind_dir)

                # 绘制箭头线
                line = pg.PlotCurveItem(
                    x=[center_x, end_x],
                    y=[center_y, end_y],
                    pen=pg.mkPen(color='r', width=1.5)
                )
                self.traj_pw.addItem(line)
                self.wind_arrows.append(line)

                # 添加箭头头部（小三角形）
                head_size = arrow_size * 0.3
                # 计算箭头头部的两个点
                angle1 = wind_dir + np.pi * 0.8
                angle2 = wind_dir - np.pi * 0.8

                head_x1 = end_x + head_size * np.cos(angle1)
                head_y1 = end_y + head_size * np.sin(angle1)
                head_x2 = end_x + head_size * np.cos(angle2)
                head_y2 = end_y + head_size * np.sin(angle2)

                # 绘制箭头头部
                head_line1 = pg.PlotCurveItem(
                    x=[end_x, head_x1],
                    y=[end_y, head_y1],
                    pen=pg.mkPen(color='b', width=1.5)
                )
                head_line2 = pg.PlotCurveItem(
                    x=[end_x, head_x2],
                    y=[end_y, head_y2],
                    pen=pg.mkPen(color='b', width=1.5)
                )

                self.traj_pw.addItem(head_line1)
                self.traj_pw.addItem(head_line2)
                self.wind_arrows.append(head_line1)
                self.wind_arrows.append(head_line2)

                # 只在风速较大时显示数值
                if wind_speed > 2.0:
                    text = pg.TextItem(
                        text=f'{wind_speed:.2f}',  # 只显示整数风速
                        color=(255, 0, 0, 200),
                        anchor=(0.5, 0.5)
                    )
                    text.setFont(QtGui.QFont('Arial', 6))  # 减小字体
                    text.setPos(center_x, center_y - 3)
                    self.traj_pw.addItem(text)
                    self.wind_texts.append(text)

        # 重置随机种子
        np.random.seed(None)

        self.wind_arrows_initialized = True
        print(f"风场箭头初始化完成，共绘制 {len(self.wind_arrows)} 个图形项")


def main():
    config_file = 'configs/config_new.ini'
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_file)
    gui.show()

    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()
