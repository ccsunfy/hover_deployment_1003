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
MODE_SHIFT_VALUE = 0.25
SENSOR_TIMEOUT = 0.5
SUCCESS_WHITE_FRACTION = 0.90
SUCCESS_SPEED_LIMIT = 0.20
SUCCESS_HOLD_SECONDS = 2.0
th.set_grad_enabled(False)


class HoverSuccessTracker:
    def __init__(self):
        self.started_at = None
        self.last_check_at = None
        self.confirmed = False

    def reset(self):
        # Completion stays latched until the node restarts.
        self.started_at = None
        self.last_check_at = None

    def update(self, mask, velocity, now):
        if self.confirmed:
            return True
        if self.last_check_at is not None and now - self.last_check_at > SENSOR_TIMEOUT:
            self.started_at = None
        self.last_check_at = now
        white_fraction = float(np.count_nonzero(mask == 255)) / mask.size
        speed_ok = bool(velocity.norm() <= SUCCESS_SPEED_LIMIT)
        if white_fraction < SUCCESS_WHITE_FRACTION or not speed_ok:
            self.started_at = None
            return False
        if self.started_at is None:
            self.started_at = now
        self.confirmed = now - self.started_at >= SUCCESS_HOLD_SECONDS
        return self.confirmed


class RealEnv:
    def __init__(self, log_prefix="", env="demo1"):
        self.log_prefix = log_prefix
        # 确保此模型对应的输入 state 是 13 维 (3位pixel + 4位ori + 3位vel + 3位omega)
        self.model = ort.InferenceSession("/home/drone/sfy/diff_land/src/sim_to_real/searchLand_930.onnx")
        model_inputs = {item.name: item for item in self.model.get_inputs()}
        if set(model_inputs) != {"semantic", "state"}:
            raise RuntimeError(f"ONNX input mismatch: {list(model_inputs)}")
        if tuple(model_inputs["semantic"].shape[1:]) != (1, 48, 64) or tuple(model_inputs["state"].shape[1:]) != (13,):
            raise RuntimeError(f"ONNX shape mismatch: {[(name, item.shape) for name, item in model_inputs.items()]}")
        x1 = np.zeros((1, 1, 48, 64), dtype=np.float32)
        x2 = np.zeros((1, 13), dtype=np.float32)
        inputs = {'semantic': x1, 'state': x2}

        rospy.loginfo(f"{self.log_prefix} 模型预热中...")
        for _ in range(6):
            self.model.run(None, inputs)

        self.bridge = CvBridge()

        self.semantic_image = None
        self.semantic_image_lock = threading.Lock()  # 线程锁
        self.semantic_image_ready = False
        self.last_semantic_time = 0
        self.semantic_timeout = SENSOR_TIMEOUT

        self.device = th.device("cpu")
        self.state_lock = threading.Lock()
        self.position = None
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None
        self.last_odom_time = 0.0
        self.last_imu_time = 0.0
        self.mode_enable = False
        self.success_tracker = HoverSuccessTracker()
        self.num_envs = 1
        self.max_sense_radius = 10.0
        self.m = 0.5

        rospy.Subscriber("/bfctrl/local_odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        rospy.Subscriber('/mavros/imu/data', Imu, self.angular_callback, queue_size=10)
        rospy.Subscriber("/semantic/image", Image, self.semantic_callback, 
                        queue_size=1, buff_size=2**24, tcp_nodelay=True)

        self.ctbr_pub = rospy.Publisher('/bfctrl/cmd', Command, queue_size=10, tcp_nodelay=True)
        self.hover_success_pub = rospy.Publisher('/searchland/hover_success', UInt8, queue_size=1, latch=True)
        self.hover_success_pub.publish(UInt8(data=0))
        self.load("/home/drone/sfy/diff_land/src/sim_to_real/example.json")
        self.last_command = None
        self.command_history = deque(maxlen=5)
        self.processing_count = 0
        self.last_stat_time = time.monotonic()

        rospy.loginfo(f"{self.log_prefix} 初始化完成，等待传感器数据...")

    def get_pixel_obs(self, img):
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
            if cv_image.shape != (48, 64):
                rospy.logwarn_once(f"{self.log_prefix} 语义图像尺寸异常: {cv_image.shape}")
                cv_image = cv2.resize(cv_image, (64, 48), interpolation=cv2.INTER_NEAREST)
            if np.any((cv_image != 0) & (cv_image != 255)):
                rospy.logerr_throttle(1, f"{self.log_prefix} 语义图像不是 0/255 二值图，拒绝输入策略")
                return
            with self.semantic_image_lock:
                self.semantic_image = cv_image.copy()
                self.semantic_image_ready = True
                self.last_semantic_time = time.monotonic()
        except CvBridgeError as e:
            rospy.logerr_throttle(1, f"{self.log_prefix} CvBridge错误: {e}")
        except Exception as e:
            rospy.logerr_throttle(1, f"{self.log_prefix} 语义回调异常: {e}")

    def rc_callback(self, data: RCIn):
        if len(data.channels) <= MODE_CHANNEL:
            rospy.logerr_throttle(1, f"{self.log_prefix} RC 通道数量不足")
            return
        self.mode_enable = (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE

    def angular_callback(self, data):
        angular_velocity = th.tensor([
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z
        ], dtype=th.float32)
        orientation = th.tensor([
            data.orientation.w,
            data.orientation.x,
            data.orientation.y,
            data.orientation.z
        ], dtype=th.float32)
        with self.state_lock:
            self.angular_velocity = angular_velocity
            self.orientation = orientation
            self.last_imu_time = time.monotonic()
        
    def odom_callback(self, data):
        # orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        # self.orientation = th.tensor([orientation.w, orientation.x, orientation.y, orientation.z], dtype=th.float32)
        with self.state_lock:
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

    def landing_success(self, mask, velocity, now):
        return self.success_tracker.update(mask, velocity, now)

    def _publish_neutral(self):
        cmd = Command()
        cmd.thrust = HOVER_ACC
        cmd.angularVel.x = cmd.angularVel.y = cmd.angularVel.z = 0
        cmd.mode = Command.ANGULAR_MODE
        cmd.header.stamp = rospy.Time.now()
        self.ctbr_pub.publish(cmd)

    def _world_to_body_velocity(self, velocity, orientation):
        # 确保 orientation 和 velocity 是二维张量 [batch_size, dim]
        if orientation.dim() == 1:
            vel = velocity.unsqueeze(0).clone()  # 形状从 [3] 变为 [1, 3]
            ori = orientation.unsqueeze(0).clone()  # 形状从 [4] 变为 [1, 4]
        else:
            vel = velocity.clone()
            ori = orientation.clone()
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
        current_time = time.monotonic()
        self.hover_success_pub.publish(UInt8(data=int(self.success_tracker.confirmed)))
        with self.state_lock:
            position = self.position
            orientation = self.orientation
            velocity = self.velocity
            angular_velocity = self.angular_velocity
            last_odom_time = self.last_odom_time
            last_imu_time = self.last_imu_time

        if (position is None or orientation is None or velocity is None or angular_velocity is None
                or current_time - last_odom_time > SENSOR_TIMEOUT
                or current_time - last_imu_time > SENSOR_TIMEOUT):
            self.success_tracker.reset()
            self.last_command = None
            rospy.logerr_throttle(1, f"{self.log_prefix} 里程计或 IMU 缺失/过期，发送中性指令")
            self._publish_neutral()
            return
        with self.semantic_image_lock:
            image_time = self.last_semantic_time
            semantic_image_copy = self.semantic_image.copy() if self.semantic_image_ready else None
        if semantic_image_copy is None or current_time - image_time > self.semantic_timeout:
            self.success_tracker.reset()
            self.last_command = None
            rospy.logerr_throttle(1, f"{self.log_prefix} 语义图像缺失/过期，发送中性指令")
            self._publish_neutral()
            return
        if (not th.isfinite(position).all() or not th.isfinite(orientation).all()
                or not th.isfinite(velocity).all() or not th.isfinite(angular_velocity).all()
                or float(orientation.norm()) < 0.5):
            self.success_tracker.reset()
            self.last_command = None
            rospy.logerr_throttle(1, f"{self.log_prefix} 状态数据无效，发送中性指令")
            self._publish_neutral()
            return

        was_confirmed = self.success_tracker.confirmed
        if self.landing_success(semantic_image_copy, velocity, current_time):
            if not was_confirmed:
                rospy.loginfo(f"{self.log_prefix} 悬停成功：白色占比≥90%、速度≤0.2m/s 持续2秒")
                self.hover_success_pub.publish(UInt8(data=1))
            self._publish_neutral()
            return

        try:

            pixel_obs = self.get_pixel_obs(semantic_image_copy)
            body_velocity = self._world_to_body_velocity(velocity, orientation)
            orientation = orientation / orientation.norm()

            state = th.hstack([
                pixel_obs.unsqueeze(0),           # (1, 3)
                orientation.unsqueeze(0),    # (1, 4)
                body_velocity / 10,  # (1, 3)
                angular_velocity.unsqueeze(0) / 10, # (1, 3)
            ]).to(self.device).float()

            state = state.reshape(1, -1)

            if semantic_image_copy.dtype != np.float32:
                semantic_image_copy = semantic_image_copy.astype(np.float32)
            if len(semantic_image_copy.shape) == 2:
                semantic_image_copy = np.expand_dims(semantic_image_copy, axis=0)
            semantic_tensor = np.expand_dims(semantic_image_copy, axis=0)

            obs = {"semantic": semantic_tensor, "state": state.float().cpu().numpy()}

            start_inference = time.monotonic()
            action = self.model.run(None, obs)[0]
            action = np.clip(action, -1, 1)
            inference_time = (time.monotonic() - start_inference) * 1000
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
                    f"位置: [{position[0].item():.2f}, {position[1].item():.2f}, {position[2].item():.2f}] | "
                    f"白色占比: {np.count_nonzero(semantic_image_copy == 255) / semantic_image_copy.size:.2f}"
                )
                self.processing_count = 0
                self.last_stat_time = current_time
        except Exception as e:
            rospy.logerr(f"{self.log_prefix} 处理循环异常: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.success_tracker.reset()
            self.last_command = None
            self._publish_neutral()


if __name__ == "__main__":
    rospy.init_node("real_end2end")
    rospy.sleep(1.0)
    agent = RealEnv(log_prefix="[DIFF LANDING START !!!!!]")
    control_rate = 33
    rate = rospy.Rate(control_rate)

    rospy.loginfo("Intitializing !!!!")
    timeout = time.monotonic() + 5.0
    while not rospy.is_shutdown() and time.monotonic() < timeout:
        if (agent.position is not None and agent.orientation is not None
                and agent.angular_velocity is not None
                and agent.semantic_image_ready):
            rospy.loginfo("Sensors Ready !")
            break
        rate.sleep()

    try:
        while not rospy.is_shutdown():
            start_time = time.monotonic()
            agent.process()
            process_time = time.monotonic() - start_time
            if process_time > 1.0 / control_rate:
                rospy.logwarn_throttle(1, f"处理时间 {process_time*1000:.1f}ms 超过周期")
            rate.sleep()
    except Exception as e:
        rospy.logerr(f"主循环异常: {e}")