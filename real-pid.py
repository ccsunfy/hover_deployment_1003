#!/usr/bin/env python

import rospy
import time
import torch as th
import numpy as np
import cv2
import time
import math, json
import onnxruntime as ort

from sensor_msgs.msg import Image,Imu
# from std_msgs.msg import Float32 
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Twist
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from quadrotor_msgs.msg import Command,TRPYCommand
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from utils.type import bound

MODE_CHANNEL = 6 
HOVER_ACC = 9.81
MODE_SHIFT_VALUE = 0.25

th.set_grad_enabled(False)

class RealEnv:
    def __init__(self, log_prefix=""):
        self.log_prefix = log_prefix
        # self.model = ort.InferenceSession("src/sim_to_real/waypoint_state_49.onnx")
        # self.model = ort.InferenceSession("src/sim_to_real/waypoint_state_327.onnx")
        # self.model = ort.InferenceSession("waypoint_state_412_low_velocity_dynamic_1.onnx")
        # self.model = ort.InferenceSession("/home/drone/sfy/src/sim_to_real/waypoint_state_413_5_2_1.onnx")

        rospy.loginfo(f"{self.log_prefix} preheating")

        self.device = th.device("cpu")
        self.position = None
        self.actions = []
        self.actions_time = []
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None
        self.yaw = 0

        self.start = False
        self.current_test_dim = 2

        rospy.Subscriber("/bfctrl/local_odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        rospy.Subscriber('/mavros/imu/data', Imu, self.angular_callback, queue_size=10)
        # pub
        self.ctbr_pub = rospy.Publisher('/bfctrl/cmd', Command, queue_size=10, tcp_nodelay=True)
        self.load("/home/drone/sfy/src/sim_to_real/example.json")

        self.t = 0
        
    def rc_callback(self, data: RCIn):
        self.mode_enable = (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE
            
    def angular_callback(self, data):
        self.angular_velocity = th.tensor([data.angular_velocity.x, 
                                         data.angular_velocity.y, 
                                         data.angular_velocity.z], dtype=th.float32)

    def odom_callback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        t = data.header.stamp

        self.orientation = th.tensor([orientation.w, orientation.x, 
                                    orientation.y, orientation.z], 
                                    dtype=th.float32)
        if orientation.w < 0: 
            self.orientation = -self.orientation
        self.position = th.tensor([position.x, position.y, position.z], 
                                 dtype=th.float32)
        self.velocity = th.tensor([velocity.x, velocity.y, velocity.z],
                                dtype=th.float32)
        # self.t = th.tensor(data.t.to_sec())
        # print("rospytime",rospy.Time.now().to_sec())
        # t = th.tensor([t.secs + t.nsecs * 1e-9],dtype=th.float32)
        self.t = rospy.Time.now().to_sec()
        # print(self.orientation)
        # rospy.loginfo("timestep:%.9f" % self.t)
        # print(self.t)

    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )

    
    def process(self):
        if self.position is None or self.orientation is None:
            return

        amplitude = 0.1  
        total_duration_times = 2 + 3 + 1 + 1  
        single_duration = 0.2
        period = single_duration * 7
        div_t = self.t % period
        # print(div_t, self.t)
        action = th.zeros(4)
        if self.start:
            if div_t <= single_duration * 2 or single_duration*5<div_t <= single_duration * 6:
                action[self.current_test_dim] = amplitude
            elif single_duration*2 < div_t <= single_duration * 5 or single_duration*6<div_t <= single_duration * 7:
                action[self.current_test_dim] = -amplitude
            else:
                raise TimeoutError
        else:
            if div_t <= 0.01:
                self.start=True
        command = th.tensor(action)
        cmd_msg = Command()
        cmd_msg.thrust = command[0].item()+9.85 
        cmd_msg.angularVel.x, cmd_msg.angularVel.y, cmd_msg.angularVel.z = command[1:4]
        cmd_msg.mode = Command.ANGULAR_MODE
        self.ctbr_pub.publish(cmd_msg)
        # t = np.linspace(0, total_duration, int(total_duration * 5))  # 30Hz采样
        # ctbr_z = np.zeros_like(t)
        
        # time_segments = [
        #     (0, 2, amplitude),    # 0-2秒 正阶跃
        #     (2, 5, -amplitude),   # 2-5秒 负阶跃 
        #     (5, 6, amplitude),    # 5-6秒 正阶跃
        #     (6, 7, -amplitude)    # 6-7秒 负阶跃
        # ]
        
        # for start, end, value in time_segments:
        #     mask = (t >= start) & (t < end)
        #     ctbr_z[mask] = value
        
        # actions = np.zeros((len(t), 4), dtype=np.float32)  # [thrust, ctbr_x, ctbr_y, ctbr_z]
        # actions[:, 2] = ctbr_z
        # actions[:, 0] = HOVER_ACC  # 保持悬停推力

        # self.actions = actions

        # for action in self.actions: 
        #     command = th.tensor(action)
        #     cmd_msg = Command()
        #     cmd_msg.thrust = command[0].item()
        #     cmd_msg.angularVel.x, cmd_msg.angularVel.y, cmd_msg.angularVel.z = command[1:4]
        #     cmd_msg.mode = Command.ANGULAR_MODE
        #     self.ctbr_pub.publish(cmd_msg)


if __name__ == "__main__":
    rospy.init_node("real_end2end")
    agent = RealEnv(log_prefix="[导航控制器]")
    rate = rospy.Rate(30)

    try:
        while not rospy.is_shutdown():
            agent.process()
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("控制器已关闭")
