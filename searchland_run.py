#!/usr/bin/env python
import rospy
import time
import torch as th
import numpy as np
import cv2
import json
import onnxruntime as ort
import os
import threading
from collections import deque

from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32
from std_msgs.msg import Bool, Float32MultiArray, UInt8
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from quadrotor_msgs.msg import Command, TRPYCommand
from cv_bridge import CvBridge, CvBridgeError
from utils.type import bound
from ActionSmoother import Smoother

MODE_CHANNEL = 6
HOVER_ACC = 9.81
# HOVER_ACC = 10.2
MODE_SHIFT_VALUE = 0.25
HOVER_WHITE_FRACTION = 0.5
HOVER_SPEED_LIMIT = 0.50
# HOVER_HOLD_SECONDS = 1.0
HOVER_HOLD_SECONDS = 0.4
HOVER_BOOST_ACC = 0.8          # 悬停确认瞬间附带的推力增量
HOVER_BOOST_DECAY_SECONDS = 1.0  # 该增量指数衰减到 ~0 所需时间
# 悬停确认后是否切换成程序手动给出的悬停指令；False = 只发标志位，继续跑策略
HOVER_USE_MANUAL_COMMAND = False
ODOM_TIMEOUT = 0.5
th.set_grad_enabled(False)


class RealEnv:
    def __init__(self, log_prefix="", env="demo1"):
        self.log_prefix = log_prefix
        # 确保此模型对应的输入 state 是 13 维 (3位pixel + 4位ori + 3位vel + 3位omega)
        # self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/BPTT_no_pos_pixel_reward_131.onnx") 
        # self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/BPTT_imu_pixel_center_reward_22.onnx")
        # self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/BPTT_imu_pixel_center_reward_24.onnx")
        # self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/Abalation_BPTT_noise_obs_32.onnx")
        self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/BPTT_searchLand-926-random-scene_2_ckpt_20000000.onnx")
        # self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/BPTT_dynamic_cube_327.onnx")
        # 模型预热 (state 维度为 13)
        
        x1 = np.random.randn(1, 1, 48, 64).astype(np.float32)
        x2 = np.random.randn(1, 13).astype(np.float32)
        inputs = {'semantic': x1, 'state': x2}

        rospy.loginfo(f"{self.log_prefix} 模型预热中...")
        for _ in range(6):
            self.model.run(None, inputs)

        self.bridge = CvBridge()

        self.semantic_image = None
        self.semantic_image_lock = threading.Lock()  # 线程锁
        self.semantic_image_ready = False
        self.last_semantic_time = 0
        self.semantic_timeout = 0.5  # 500ms超时

        self.device = th.device("cpu")
        self.position = None
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None
        self.last_odom_time = 0.0
        self.hover_started_at = None
        self.hover_confirmed = False
        self.hover_confirmed_at = None
        self.num_envs = 1
        self.max_sense_radius = 10.0
        # self.target = th.as_tensor([[1.58, -0.07, 0.16]])
        self.m = 0.5

        rospy.Subscriber("/bfctrl/local_odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        rospy.Subscriber('/mavros/imu/data', Imu, self.angular_callback, queue_size=10)
        rospy.Subscriber("/semantic/image", Image, self.semantic_callback, 
                        queue_size=1, buff_size=2**24, tcp_nodelay=True)

        self.ctbr_pub = rospy.Publisher('/bfctrl/cmd', Command, queue_size=10, tcp_nodelay=True)
        self.hover_success_pub = rospy.Publisher('/hover/success', UInt8, queue_size=1, latch=True)
        self.hover_success_pub.publish(UInt8(data=0))

        self.load("/home/drone/sfy/diff_land/src/sim_to_real/example.json")

        self.last_command = None
        self.command_history = deque(maxlen=5)

        self.processing_count = 0
        self.last_stat_time = time.time()

        rospy.loginfo(f"{self.log_prefix} 初始化完成，等待传感器数据...")

    def get_pixel_obs(self, img):
        """
        对应仿真环境 VisualLandingEnv 中的像素观测逻辑
        返回: [u_norm, v_norm, visible]
        """
        # img 预期形状为 (48, 64)
        indices = np.where(img > 0)
        if len(indices[0]) > 0:
            # 仿真中逻辑: y_center = np.mean(indices[1]), x_center = np.mean(indices[2])
            # 注意: np.where 返回 (row_indices, col_indices)，对应 (y, x)
            y_center = np.mean(indices[0])
            x_center = np.mean(indices[1])
            H, W = img.shape
            u_norm = (x_center - W/2) / (W/2)
            v_norm = (y_center - H/2) / (H/2)
            visable = 1.0
        else:
            u_norm = 0.0
            v_norm = 0.0
            visable = 0.0
        return th.tensor([u_norm, v_norm, visable], device=self.device, dtype=th.float32)

    def semantic_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            if cv_image is None or not isinstance(cv_image, np.ndarray):
                rospy.logwarn(f"{self.log_prefix} 图像转换失败")
                return
            if cv_image.shape == (48, 64):
                with self.semantic_image_lock:
                    self.semantic_image = cv_image
                    self.semantic_image_ready = True
                    self.last_semantic_time = time.time()
            else:
                rospy.logwarn_once(f"{self.log_prefix} 语义图像尺寸异常: {cv_image.shape}")
                try:
                    cv_image = cv2.resize(cv_image, (64, 48)) # W, H
                    with self.semantic_image_lock:
                        self.semantic_image = cv_image
                        self.semantic_image_ready = True
                        self.last_semantic_time = time.time()
                except Exception as e:
                    rospy.logerr(f"{self.log_prefix} 图像resize失败: {e}")
        except CvBridgeError as e:
            rospy.logerr_throttle(1, f"{self.log_prefix} CvBridge错误: {e}")
        except Exception as e:
            rospy.logerr_throttle(1, f"{self.log_prefix} 语义回调异常: {e}")

    def rc_callback(self, data: RCIn):
        self.mode_enable = (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE

    def angular_callback(self, data):
        self.angular_velocity = th.tensor([
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z
        ], dtype=th.float32)
        # self.orientation = th.tensor([
        #     data.orientation.w,
        #     data.orientation.x,
        #     data.orientation.y,
        #     data.orientation.z
        # ], dtype=th.float32)
        
    def odom_callback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        self.orientation = th.tensor([orientation.w, orientation.x, orientation.y, orientation.z], dtype=th.float32)
        self.position = th.tensor([position.x, position.y, position.z], dtype=th.float32)
        self.velocity = th.tensor([velocity.x, velocity.y, velocity.z], dtype=th.float32)
        self.last_odom_time = time.monotonic()

    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"]))

    def de_normalize(self, command):
        thrust_scale = (self.m * HOVER_ACC) / self.m
        thrust_bias = (self.m * HOVER_ACC) / self.m
        bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / 2.0
        bodyrate_bias = self._bd_rate.max - bodyrate_scale * 1.0
        command = th.hstack([
            (command[:, :1] * thrust_scale + thrust_bias),
            command[:, 1:] * bodyrate_scale + bodyrate_bias
        ])
        return command.T

    def hover_success(self, mask, now):
        if self.hover_confirmed:
            return True
        if self.velocity is None or now - self.last_odom_time > ODOM_TIMEOUT:
            self.hover_started_at = None
            return False
        white_fraction = np.count_nonzero(mask == 255) / mask.size
        speed = self.velocity.norm().item()
        if white_fraction <= HOVER_WHITE_FRACTION or speed >= HOVER_SPEED_LIMIT:
            self.hover_started_at = None
            return False
        if self.hover_started_at is None:
            self.hover_started_at = now
        if now - self.hover_started_at >= HOVER_HOLD_SECONDS:
            self.hover_confirmed = True
            self.hover_confirmed_at = now
            self.hover_success_pub.publish(UInt8(data=1))
            rospy.loginfo(f"{self.log_prefix} 悬停成功：白色占比>60%、速度<0.5m/s 持续1秒")
        return self.hover_confirmed

    def hover_boost(self):
        """悬停确认后附加的推力增量，从 HOVER_BOOST_ACC 指数衰减到 0。"""
        if self.hover_confirmed_at is None:
            return HOVER_BOOST_ACC
        elapsed = time.monotonic() - self.hover_confirmed_at
        if elapsed >= HOVER_BOOST_DECAY_SECONDS:
            return 0.0
        # 1 秒内衰减到 e^-5 ≈ 0.7%，之后直接归零
        tau = HOVER_BOOST_DECAY_SECONDS / 5.0
        return float(HOVER_BOOST_ACC * np.exp(-elapsed / tau))

    def _world_to_body_velocity(self):
        """
        将世界坐标系下的线性速度转换为机体坐标系下的速度。
        包含 BPTT 安全性修复：Clone 和 Normalization。
        """
        # 确保 orientation 和 velocity 是二维张量 [batch_size, dim]
        if self.orientation.dim() == 1:
            vel = self.velocity.unsqueeze(0).clone()  # 形状从 [3] 变为 [1, 3]
            ori = self.orientation.unsqueeze(0).clone()  # 形状从 [4] 变为 [1, 4]
        else:
            vel = self.velocity.clone()
            ori = self.orientation.clone()

        # 归一化四元数，防止梯度爆炸/NaN
        norm = ori.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        q = ori / norm

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx_w, vy_w, vz_w = vel[:, 0], vel[:, 1], vel[:, 2]

        # 计算旋转矩阵的逆 (World -> Body)
        # row 1 (Body X axis projection)
        bx = (1 - 2*y**2 - 2*z**2) * vx_w + (2*x*y + 2*w*z) * vy_w + (2*x*z - 2*w*y) * vz_w

        # row 2 (Body Y axis projection)
        by = (2*x*y - 2*w*z) * vx_w + (1 - 2*x**2 - 2*z**2) * vy_w + (2*y*z + 2*w*x) * vz_w

        # row 3 (Body Z axis projection)
        bz = (2*x*z + 2*w*y) * vx_w + (2*y*z - 2*w*x) * vy_w + (1 - 2*x**2 - 2*y**2) * vz_w

        return th.stack([bx, by, bz], dim=1)  # 返回形状为 [1, 3]

    def process(self):
        current_time = time.time()
        self.hover_success_pub.publish(UInt8(data=int(self.hover_confirmed)))
        if self.position is None or self.orientation is None:
            rospy.logwarn_throttle(2, f"{self.log_prefix} waiting for data...")
            return

        semantic_image_copy = None
        use_fallback_image = False
        with self.semantic_image_lock:
            semantic_ready = self.semantic_image_ready
            semantic_time_ok = (current_time - self.last_semantic_time) < self.semantic_timeout
            if semantic_ready and semantic_time_ok and self.semantic_image is not None:
                semantic_image_copy = self.semantic_image.copy()
            else:
                # 与仿真保持一致：不可见/超时应对应全 0 mask（visible=0）
                semantic_image_copy = np.zeros((48, 64), dtype=np.uint8)
                use_fallback_image = True
                rospy.logwarn_throttle(2, f"{self.log_prefix} 语义图像不可用/超时，使用全黑图像")

        # 只负责判定并发布悬停成功标志位，指令始终由策略给出
        hovered = self.hover_success(semantic_image_copy, time.monotonic())
        if hovered and HOVER_USE_MANUAL_COMMAND:
            hover_cmd = Command()
            hover_cmd.thrust = HOVER_ACC + self.hover_boost()
            hover_cmd.angularVel.x = hover_cmd.angularVel.y = hover_cmd.angularVel.z = 0
            hover_cmd.mode = Command.ANGULAR_MODE
            hover_cmd.header.stamp = rospy.Time.now()
            self.ctbr_pub.publish(hover_cmd)
            return

        try:

            pixel_obs = self.get_pixel_obs(semantic_image_copy)

            body_velocity = self._world_to_body_velocity()

            state = th.hstack([
                pixel_obs.unsqueeze(0),           # (1, 3)
                self.orientation.unsqueeze(0),    # (1, 4)
                # self.velocity.unsqueeze(0) / 10,  # (1, 3)
                body_velocity / 10,  # (1, 3)
                self.angular_velocity.unsqueeze(0) / 10, # (1, 3)
            ]).to(self.device).float()

            state = state.reshape(1, -1)

            if semantic_image_copy.dtype != np.float32:
                semantic_image_copy = semantic_image_copy.astype(np.float32)
            if len(semantic_image_copy.shape) == 2:
                semantic_image_copy = np.expand_dims(semantic_image_copy, axis=0)
            semantic_tensor = np.expand_dims(semantic_image_copy, axis=0)

            if use_fallback_image:
                rospy.logwarn_throttle(1, f"{self.log_prefix} 使用全黑语义图像作为输入")

            obs = {"semantic": semantic_tensor, "state": state.float().cpu().numpy()}

            start_inference = time.time()
            action = self.model.run(None, obs)[0]
            action = np.clip(action, -1, 1)
            inference_time = (time.time() - start_inference) * 1000
            command = self.de_normalize(th.tensor(action))

            if self.last_command is not None:
                alpha = 0.3
                command = alpha * command + (1 - alpha) * self.last_command

            self.last_command = command
            self.command_history.append(command)

            cmd_msg = Command()
            cmd_msg.thrust = command[0].item()
            cmd_msg.angularVel.x = command[1].item()
            cmd_msg.angularVel.y = command[2].item()
            cmd_msg.angularVel.z = command[3].item()
            cmd_msg.mode = Command.ANGULAR_MODE
            cmd_msg.header.stamp = rospy.Time.now()
            self.ctbr_pub.publish(cmd_msg)

            self.processing_count += 1
            if current_time - self.last_stat_time > 2.0:
                rospy.loginfo(
                    f"{self.log_prefix} 运行中 | 推理: {inference_time:.1f}ms | "
                    f"像素观测: [u:{pixel_obs[0].item():.2f}, v:{pixel_obs[1].item():.2f}, vis:{pixel_obs[2].item():.1f}] | "
                    f"位置: [{self.position[0].item():.2f}, {self.position[1].item():.2f}, {self.position[2].item():.2f}] | "
                    f"图像源: {'全黑回退图像' if use_fallback_image else '正常图像'}"
                )
                self.processing_count = 0
                self.last_stat_time = current_time
            # 不再在每步中重置 semantic_image_ready：
            # 用 last_semantic_time + semantic_timeout 控制是否超时兜底，避免反复进入不一致观测分布。

        except Exception as e:
            rospy.logerr(f"{self.log_prefix} 处理循环异常: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

            safety_cmd = Command()
            safety_cmd.thrust = HOVER_ACC
            safety_cmd.angularVel.x = safety_cmd.angularVel.y = safety_cmd.angularVel.z = 0
            safety_cmd.mode = Command.ANGULAR_MODE
            safety_cmd.header.stamp = rospy.Time.now()
            self.ctbr_pub.publish(safety_cmd)


if __name__ == "__main__":
    rospy.init_node("real_end2end")
    rospy.sleep(1.0)
    agent = RealEnv(log_prefix="[DIFF LANDING START !!!!!]")
    control_rate = 30
    rate = rospy.Rate(control_rate)

    rospy.loginfo("Intitializing !!!!")
    timeout = time.time() + 5.0
    while not rospy.is_shutdown() and time.time() < timeout:
        if agent.position is not None and agent.orientation is not None:
            rospy.loginfo("Sensors Ready !")
            break
        rate.sleep()

    try:
        while not rospy.is_shutdown():
            start_time = time.time()
            agent.process()
            process_time = time.time() - start_time
            if process_time > 1.0 / control_rate:
                rospy.logwarn_throttle(1, f"处理时间 {process_time*1000:.1f}ms 超过周期")
            rate.sleep()
    except Exception as e:
        rospy.logerr(f"主循环异常: {e}")