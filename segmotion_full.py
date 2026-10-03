#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segmotion_full.py —— 单色圆筒实时分割【完整版：独立窗口 + ROS 二合一，单文件自包含】

本文件由 merge_segmotion.py 从 segmotion.py（检测器 + 独立窗口版）与
segmotion_node.py（ROS 节点）机械合并生成，代码正文与两份原件逐字一致。
两份原文件保留不动；发文件给别人时只发这一个即可。

⚠️ 不要直接改这个文件：改阈值/逻辑请改 segmotion.py，然后跑 `python3 merge_segmotion.py` 重新生成。

两种运行方式（同一套检测器、同一套阈值）：

1) 独立模式（自带 OpenCV 窗口，不需要 ROS）：
       python segmotion_full.py --color red --source 0 --fourcc MJPG
       python segmotion_full.py --color blue --source udp://0.0.0.0:5600 --record out.mp4
       python segmotion_full.py --probe                 # 查设备号
   按键 q / ESC 退出。

2) ROS 模式（发布话题，需要 ROS1 + 能 import rospy/cv_bridge 的 python）：
       python segmotion_full.py --ros _color:=red
       python segmotion_full.py --ros _color:=blue _publish_overlay:=false
       python segmotion_full.py --ros --color red        # --color red 也可，会自动转成 ROS 参数
   发布：~mask / ~overlay / ~target / ~locked / ~set_color，以及 /semantic/image(48,64,mono8)
   查看：rqt_image_view 选 /segmotion_node/overlay
   换色：rostopic pub /segmotion_node/set_color std_msgs/String "data: 'blue'" -1

⚠️ 阈值 COLOR_HSV 是按目标数据集标定的，换相机/场地后需要重新标定。
"""

import argparse
import math
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# ROS 依赖为可选导入：没有 ROS 的环境仍可使用独立模式
try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, Float32MultiArray, String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# ---------------------------------------------------------------- 颜色阈值
# 与 segmentation.py 同一套标定值（HSV：H 0~179, S/V 0~255）
COLOR_HSV = {
    "red": [((0, 90, 70), (10, 255, 255)),
            ((168, 90, 70), (180, 255, 255))],
    "yellow": [((15, 150, 90), (38, 255, 255))],
    # 蓝色用两段区间（浅蓝板 + 亮面高光）：
    #   区间1 主体：饱和度下限放宽到 70（浅蓝板本身饱和度不高），亮度下限抬到 95
    #        （把实验场地那种偏暗的地面挡在外面）
    #   区间2 高光：亮面反光接近白色，饱和度可低到 40，但亮度很高（>=165）——
    #        接住这部分才能让 mask 连成整块，消除边缘碎裂/锯齿
    "blue": [((95, 70, 95), (130, 255, 255)),
             ((88, 40, 165), (132, 255, 255))],
}
COLOR_BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255), "blue": (255, 0, 0)}
COLOR_ID = {"red": 1, "yellow": 2, "blue": 3}   # 与 segmentation.py 标签图一致；ROS 版发话题也用

REF_WIDTH = 4000      # 核尺寸的标定参考宽度（segmentation.py 是在 4000px 宽的图上定的）
GAP = 8               # 左右两个画面之间的间隔


def scaled_kernel(base, width, minimum=3):
    """把按 REF_WIDTH 标定的核尺寸，缩放到当前处理宽度（保持奇数）。"""
    k = max(minimum, int(round(base * width / REF_WIDTH)))
    return k if k % 2 else k + 1


class ColorBucketDetector:
    """单色圆筒检测器：阈值 → 形态学 → 孔洞填充 → 连通域 → 形状筛选 → 时序平滑。"""

    def __init__(self, color, proc_width=640, min_area=100, min_solidity=0.70,
                 max_elongation=3.0, smooth=0.5, confirm=2, miss=5, temporal=True,
                 reacquire=0.25, keep_all=False,
                 max_ellipse_residual=0.075, ellipse_residual_k=0.75):
        if color not in COLOR_HSV:
            raise ValueError(f"不支持的颜色: {color}（可选 {list(COLOR_HSV)}）")
        self.color = color
        self.ranges = COLOR_HSV[color]          # 只保留选中颜色 → 其他颜色天然屏蔽
        self.proc_width = proc_width
        self.min_area = min_area
        self.min_solidity = min_solidity
        self.max_elongation = max_elongation
        # 形状判据：只保留圆/椭圆（挡掉同色的三角形、方块、星形等）
        # 残差 = 轮廓点变换到拟合椭圆的归一化坐标系后的半径标准差（完美圆/椭圆 ≈ 0）
        self.max_ellipse_residual = max_ellipse_residual
        self.ellipse_residual_k = ellipse_residual_k
        self.k_open = scaled_kernel(9, proc_width)
        self.k_close = scaled_kernel(25, proc_width)
        # 时序状态
        self.temporal = temporal
        self.smooth = smooth
        self.confirm_n, self.miss_max = confirm, miss
        self.reacquire = reacquire              # 与平滑位置偏离超过该比例×画面宽 → 重新捕获
        self.keep_all = keep_all                # True=mask 保留所有同色候选
        self.cx = self.cy = None
        self.hits = self.misses = 0

    def set_color(self, color):
        """运行期切换颜色（清空时序状态，重新捕获）。ROS 版通过话题调用。"""
        if color not in COLOR_HSV:
            raise ValueError(f"不支持的颜色: {color}（可选 {list(COLOR_HSV)}）")
        self.color = color
        self.ranges = COLOR_HSV[color]
        self.cx = self.cy = None
        self.hits = self.misses = 0

    def residual_limit(self, area):
        """椭圆残差上限（随面积自适应）。

        小目标像素化严重，轮廓阶梯会让残差天然偏大（实测直径 20px 的完美圆
        残差已有 0.064），所以小面积时按 k/sqrt(面积) 放宽，否则会误杀远处的桶。
        """
        return max(self.max_ellipse_residual,
                   self.ellipse_residual_k / math.sqrt(max(1.0, float(area))))

    @staticmethod
    def ellipse_residual(contour):
        """轮廓到拟合椭圆的归一化半径标准差：完美圆/椭圆≈0，方块≈0.10，三角≈0.19。

        做法：cv2.fitEllipse 拟合 → 把轮廓点旋转到椭圆主轴坐标系、按半轴归一化，
        完美椭圆上所有点半径恒为 1，残差即半径的离散程度。对长宽比不敏感
        （2:1、3:1 的斜视椭圆依然是"圆"）。
        """
        if len(contour) < 5:            # fitEllipse 至少需要 5 个点
            return None
        (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(contour)
        a, b = ax1 / 2.0, max(1e-6, ax2 / 2.0)
        th = math.radians(ang)
        pts = contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
        xr = pts[:, 0] * math.cos(th) + pts[:, 1] * math.sin(th)
        yr = -pts[:, 0] * math.sin(th) + pts[:, 1] * math.cos(th)
        r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        return float(r.std())

    def _best_blob(self, mask):
        """挑出最像圆筒的一块，返回 (质心x, 质心y, 面积, 包围框, 该块mask, 椭圆残差)。"""
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
            solidity = float(cv2.contourArea(c)) / hull_area
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            elongation = max(rw, rh) / max(1.0, min(rw, rh))
            if solidity < self.min_solidity or elongation > self.max_elongation:
                continue
            resid = self.ellipse_residual(c)
            if resid is not None and resid > self.residual_limit(area):
                continue                      # 不是圆/椭圆（方块、三角、星形…）
            if best is None or area > best[2]:
                x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
                best = (float(cents[i][0]), float(cents[i][1]), area, (x, y, w, h),
                        comp, resid)
        return best

    def detect(self, frame):
        """
        处理一帧，返回 (mask_small, result)。
        result 为 None 表示本帧（或平滑后）无有效目标；坐标已换算回原图尺度。
        """
        fh, fw = frame.shape[:2]
        scale = self.proc_width / fw
        pw, ph = self.proc_width, max(8, int(round(fh * scale)))
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = np.zeros((ph, pw), np.uint8)
        for lo, hi in self.ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.k_open, self.k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.k_close, self.k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.medianBlur(mask, 3)   # 去掉单像素毛刺：边界更平滑，消除锯齿

        # 外轮廓填充：补掉高光造成的内部空洞，得到实心目标
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 255, thickness=cv2.FILLED)

        blob = self._best_blob(mask)
        # 单目标模式：mask 只保留被追踪的那一块；--keep-all 时保留全部同色候选
        if self.keep_all:
            out_mask = mask
        elif blob is not None:
            out_mask = blob[4]
        else:
            out_mask = np.zeros_like(mask)
        return out_mask, self._track(blob, scale, pw)

    def _track(self, blob, scale, pw):
        """时序平滑：EMA 质心 + 连续确认 + 丢失保持 + 跳变重捕获。"""
        if not self.temporal:
            if blob is None:
                return None
            cx, cy, area, bbox, _, resid = blob
            return self._result(cx, cy, area, bbox, scale, locked=True,
                                raw=(cx, cy), resid=resid)

        if blob is not None:
            cx, cy, area, bbox, _, resid = blob
            # 与上一位置偏离过大 → 不是同一个目标（场景切换/目标更换），重新捕获而非硬平滑
            if self.cx is None or np.hypot(cx - self.cx, cy - self.cy) > self.reacquire * pw:
                self.cx, self.cy = cx, cy
                self.hits = 0
            else:
                a = self.smooth
                self.cx = a * self.cx + (1 - a) * cx
                self.cy = a * self.cy + (1 - a) * cy
            self.hits += 1
            self.misses = 0
            return self._result(self.cx, self.cy, area, bbox, scale,
                                locked=self.hits >= self.confirm_n, raw=(cx, cy),
                                resid=resid)

        # 本帧没检出：短暂保持上一位置，超过 miss 帧才判定丢失
        self.misses += 1
        self.hits = 0
        if self.cx is None or self.misses > self.miss_max:
            self.cx = self.cy = None
            return None
        return self._result(self.cx, self.cy, 0, None, scale, locked=False,
                            raw=None, resid=None)

    def _result(self, cx, cy, area, bbox, scale, locked, raw, resid=None):
        x, y, w, h = (bbox if bbox else (0, 0, 0, 0))
        res = {
            "color": self.color,
            "color_id": COLOR_ID[self.color],
            "locked": locked,
            "cx": round(cx / scale, 1),            # 平滑后位置（原图坐标）
            "cy": round(cy / scale, 1),
            "area": int(round(area / (scale ** 2))),
            "bbox": [int(x / scale), int(y / scale), int(w / scale), int(h / scale)],
        }
        if resid is not None:                      # 椭圆残差，越小越像圆/椭圆
            res["ellipse_residual"] = round(resid, 4)
        if raw is not None:                        # 本帧原始检出位置，控制回路可选更小滞后
            res["cx_raw"] = round(raw[0] / scale, 1)
            res["cy_raw"] = round(raw[1] / scale, 1)
        return res


URL_SCHEMES = ("rtsp", "rtmp", "http", "https", "udp", "rtp", "tcp", "srt")

# 网络流低延迟参数（FFMPEG 后端）。用分号分隔，可用环境变量
# OPENCV_FFMPEG_CAPTURE_OPTIONS 覆盖（本函数用 setdefault，不会覆盖用户已有设置）。
LOW_LATENCY_OPTS = ("fflags;nobuffer|flags;low_delay|max_delay;0|reorder_queue_size;0"
                    "|rtsp_transport;udp")


def is_url(source):
    return isinstance(source, str) and "://" in source and \
        source.split("://", 1)[0].lower() in URL_SCHEMES


def is_device_path(source):
    """识别 /dev/videoN 这类设备路径（等价于摄像头序号，走 V4L2 后端）。"""
    return isinstance(source, str) and source.startswith("/dev/video")


def fixup_url(url):
    """UDP 流加固：ffmpeg 默认 FIFO 很小，视频码率一冲就 overrun 断流，
    自动加大缓冲区并允许超限后继续（用户已显式指定时不覆盖）。"""
    if url.lower().startswith("udp://") and "fifo_size" not in url \
            and "overrun_nonfatal" not in url:
        return url + ("&" if "?" in url else "?") + \
            "fifo_size=10000000&overrun_nonfatal=1"
    return url


class VideoSource:
    """采集线程：只保留最新帧，避免处理速度跟不上时延迟堆积。

    支持的源：
      * 摄像头序号（int）—— V4L2 后端，USB 采集卡 / 机载相机走这条
      * 网络流 URL（rtsp/rtmp/http/udp/tcp/srt）—— FFMPEG 后端 + 低延迟参数
      * `gst:<管道>` —— GStreamer 后端（需 OpenCV 编译时启用 GStreamer）
      * 视频文件路径 —— 按原生帧率节流（调试用）
    """

    def __init__(self, source, width, height, fps, loop=False, fourcc=None):
        if isinstance(source, str) and source.startswith("gst:"):
            backend, target = cv2.CAP_GSTREAMER, source[4:]
        elif is_url(source):
            backend, target = cv2.CAP_FFMPEG, fixup_url(source)
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", LOW_LATENCY_OPTS)
        elif isinstance(source, int) or is_device_path(source):
            backend, target = cv2.CAP_V4L2, source
        else:
            backend, target = cv2.CAP_ANY, source
        self.backend, self.target = backend, target

        self.cap = cv2.VideoCapture(target, backend)
        self.source = source
        if not self.cap.isOpened():
            hint = ""
            if backend == cv2.CAP_GSTREAMER:
                hint = "（当前 OpenCV 未编译 GStreamer 支持，请改用 rtsp/udp 等 URL）"
            raise RuntimeError(f"无法打开视频源: {source} {hint}")

        self.is_camera = isinstance(source, int) or is_device_path(source)
        self.is_live = self.is_camera or is_url(source)   # 实时源：不节流，直接跟最新帧
        if self.is_camera:
            if fourcc:                       # 采集卡务必先设 MJPG 再设分辨率
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 关键：避免缓冲造成延迟堆积
        # 文件源：相机/网络流由驱动或发送端按帧率供帧，文件会被瞬间读完，必须节流，
        # 否则采集线程会跑在消费端前面、把整段视频丢光。
        video_fps = self.cap.get(cv2.CAP_PROP_FPS) if not self.is_live else 0.0
        self.frame_interval = 1.0 / video_fps if 1.0 < video_fps < 240 else 1.0 / 30.0
        self.loop = loop
        self.q = queue.Queue(maxsize=1)
        self.eof = False
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        t_next = time.perf_counter()
        fails = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if is_url(self.source):       # 网络流：丢包/断流后自动重连，不退出
                    fails += 1
                    if fails == 1:
                        print(f"[segmotion] 网络流中断，尝试重连... ({self.source})")
                    time.sleep(0.3)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.target, self.backend)
                    if fails > 200:           # 约 1 分钟仍连不上才放弃
                        self.eof = True
                        break
                    continue
                self.eof = True
                break
            fails = 0
            if not self.is_live:              # 文件源按原生帧率节流
                t_next += self.frame_interval
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()
            try:
                self.q.get_nowait()      # 丢掉旧帧
            except queue.Empty:
                pass
            self.q.put(frame)

    def read(self, timeout=2.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self):
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()


def compose(frame, mask_small, result, color, fps, latency_ms):
    """拼出可视化画面：左=原图+叠加，右=二值 mask，HUD 显示状态。"""
    fh, fw = frame.shape[:2]
    scale = fw / mask_small.shape[1]
    # 线性插值放大后再二值化：边界比最近邻放大平滑得多（消除块状锯齿）
    mask_full = (cv2.resize(mask_small, (fw, fh), interpolation=cv2.INTER_LINEAR) > 127)
    mask_full = mask_full.astype(np.uint8) * 255

    left = frame.copy()
    if result is not None:
        on = mask_full > 0
        left[on] = (0.55 * left[on] + 0.45 * np.array(COLOR_BGR[color])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cnts, -1, COLOR_BGR[color], 3)
        if result["bbox"][2] > 0:
            x, y, w, h = result["bbox"]
            cv2.rectangle(left, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # 实心十字 = 当帧原始质心（与 mask/包围框严格对齐）
        # 空心圆 = 时序平滑后的位置（检测滞后是否明显一看便知）；丢失保持时只有空心圆
        if "cx_raw" in result:
            cv2.drawMarker(left, (int(result["cx_raw"]), int(result["cy_raw"])),
                           (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
        sx, sy = int(result["cx"]), int(result["cy"])
        if "cx_raw" not in result or np.hypot(sx - result["cx_raw"], sy - result["cy_raw"]) > 4:
            cv2.circle(left, (sx, sy), 14, COLOR_BGR[color], 3)

    right = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((fh, fw * 2 + GAP, 3), np.uint8)
    canvas[:, :fw] = left
    canvas[:, fw + GAP:] = right

    # HUD（ASCII，OpenCV 内置字体不支持中文）
    if result is None:
        state = "NO TARGET"
    else:
        px = result.get("cx_raw", result["cx"])
        py = result.get("cy_raw", result["cy"])
        state = ("LOCKED " if result["locked"] else "HOLD   ") + \
                f"({px:.0f},{py:.0f}) A={result['area']}"
    lines = [
        f"COLOR : {color.upper()}   (other colors masked out)",
        f"STATE : {state}",
        f"FPS   : {fps:5.1f}    PROC: {latency_ms:5.1f} ms/frame",
    ]
    for i, t in enumerate(lines):
        cv2.putText(canvas, t, (16, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(canvas, t, (16, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def probe_devices():
    """列出本机可用视频设备与可读格式，用于确认采集卡是哪个 /dev/videoN。"""
    import glob
    print("本机视频设备：")
    names = {}
    for p in glob.glob("/sys/class/video4linux/*/name"):
        idx = int(p.split("/")[-2].replace("video", ""))
        names[idx] = open(p).read().strip()
    if not names:
        print("  （无 /dev/video* 设备：采集卡未插好，或缺 v4l2 驱动）")
        return
    for idx in sorted(names):
        line = f"  /dev/video{idx:<2}  {names[idx]:<28}"
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, f = cap.read()
            if ok:
                line += f"可读 {f.shape[1]}x{f.shape[0]} {cap.get(cv2.CAP_PROP_FPS):.0f}fps"
            else:
                line += "打开成功但读不到帧（可能是元数据节点，换下一个序号试）"
        else:
            line += "打不开"
        cap.release()
        print(line)
    print("\n提示：USB 采集卡通常占用一个 /dev/videoN，若插入后设备号变化，"
          "用 v4l2-ctl --list-devices 或本条命令按名称确认。")




def main_standalone():
    ap = argparse.ArgumentParser(description="单色圆筒实时分割（实时可视化）")
    ap.add_argument("--color", choices=list(COLOR_HSV),
                    help="抛掷物颜色：只分割该颜色的圆筒，其他颜色屏蔽")
    ap.add_argument("--source", default="0",
                    help="视频源：摄像头/采集卡序号（如 0）、"
                         "网络流 URL（rtsp:// udp:// http://）、gst:<管道>、或视频文件路径")
    ap.add_argument("--fourcc", default=None,
                    help="摄像头像素格式，如 MJPG（UVC 采集卡建议显式指定）")
    ap.add_argument("--probe", action="store_true",
                    help="列出本机视频设备与可读格式后退出（找采集卡用）")
    ap.add_argument("--width", type=int, default=1280, help="采集宽度（摄像头）")
    ap.add_argument("--height", type=int, default=720, help="采集高度（摄像头）")
    ap.add_argument("--fps", type=float, default=30.0, help="采集帧率（摄像头）")
    ap.add_argument("--proc-width", type=int, default=640,
                    help="处理分辨率宽度（越小越快，默认 640）")
    ap.add_argument("--min-area", type=int, default=100, help="最小面积（处理尺度像素）")
    ap.add_argument("--min-solidity", type=float, default=0.70, help="最小凸度")
    ap.add_argument("--max-elongation", type=float, default=3.0, help="最大伸长率")
    ap.add_argument("--max-ellipse-residual", type=float, default=0.075,
                    help="椭圆残差上限（越小越只认圆/椭圆；方块≈0.10、三角≈0.19）")
    ap.add_argument("--ellipse-residual-k", type=float, default=0.75,
                    help="小目标的残差放宽系数：上限 = max(上一参数, k/√面积)")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="质心 EMA 系数（0~1，越大越平滑越滞后）")
    ap.add_argument("--no-smooth", action="store_true", help="关闭时序平滑")
    ap.add_argument("--confirm", type=int, default=2, help="连续多少帧确认目标")
    ap.add_argument("--miss", type=int, default=5, help="连续多少帧未检出判定丢失")
    ap.add_argument("--reacquire", type=float, default=0.25,
                    help="偏离平滑位置超过该比例×画面宽时重新捕获（0~1）")
    ap.add_argument("--keep-all", action="store_true",
                    help="mask 保留全部同色候选（默认只保留被追踪的那一个）")
    ap.add_argument("--display-scale", type=float, default=1.0, help="显示/录制缩放")
    ap.add_argument("--record", default=None, help="录制可视化画面到视频文件")
    ap.add_argument("--loop", action="store_true", help="视频文件循环播放（调试用）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理满 N 帧后自动退出（0=不限，压力测试/比赛实跑可用）")
    ap.add_argument("--no-display", action="store_true", help="不弹窗（无显示器环境）")
    ap.add_argument("--quiet", action="store_true", help="不打印逐帧信息")
    args = ap.parse_args()

    if args.probe:
        probe_devices()
        return 0
    if not args.color:
        ap.error("必须指定 --color（或用 --probe 查看设备）")

    source = int(args.source) if args.source.isdigit() else args.source
    try:
        src = VideoSource(source, args.width, args.height, args.fps,
                          loop=args.loop, fourcc=args.fourcc)
    except RuntimeError as e:
        print(f"[错误] {e}")
        return 1

    det = ColorBucketDetector(
        args.color, proc_width=args.proc_width, min_area=args.min_area,
        min_solidity=args.min_solidity, max_elongation=args.max_elongation,
        smooth=args.smooth, confirm=args.confirm, miss=args.miss,
        temporal=not args.no_smooth, reacquire=args.reacquire, keep_all=args.keep_all,
        max_ellipse_residual=args.max_ellipse_residual,
        ellipse_residual_k=args.ellipse_residual_k)

    print(f"[segmotion] 颜色开关 = {args.color.upper()}（其他颜色已屏蔽）")
    print(f"[segmotion] 处理分辨率 {args.proc_width}px，"
          f"形态学核 open={det.k_open}/close={det.k_close}，"
          f"时序平滑={'关' if args.no_smooth else '开'}")
    print(f"[segmotion] 视频源 {source}；按 q 或 ESC 退出")

    gui = not args.no_display
    if gui:
        try:
            cv2.namedWindow("segmotion", cv2.WINDOW_NORMAL)
        except cv2.error:
            gui = False
    if not gui:
        print("[segmotion] 无窗口模式：只统计与录制（--record 可留证据）")

    writer = None
    intervals, latencies, frames, det_frames = deque(maxlen=60), deque(maxlen=200), 0, 0
    t_prev = time.perf_counter()

    try:
        while True:
            frame = src.read()
            if frame is None:
                if src.eof:
                    print("[segmotion] 视频结束")
                    break
                continue

            t0 = time.perf_counter()
            mask_small, result = det.detect(frame)
            latency = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            intervals.append(now - t_prev)
            t_prev = now
            latencies.append(latency)
            frames += 1
            det_frames += result is not None
            fps = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0

            canvas = compose(frame, mask_small, result, args.color, fps, latency)
            if args.display_scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=args.display_scale,
                                    fy=args.display_scale, interpolation=cv2.INTER_AREA)

            if args.record:
                if writer is None:
                    h, w = canvas.shape[:2]
                    w, h = w - (w % 2), h - (h % 2)      # 编码器要求偶数尺寸
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (w, h))
                    if not writer.isOpened():
                        print(f"[错误] 无法写入: {args.record}")
                        return 1
                writer.write(canvas[:canvas.shape[0] - canvas.shape[0] % 2,
                                    :canvas.shape[1] - canvas.shape[1] % 2])

            if gui:
                cv2.imshow("segmotion", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if not args.quiet and frames % 30 == 0:
                tag = "无目标" if result is None else \
                    f"目标({result.get('cx_raw', result['cx']):.0f}," \
                    f"{result.get('cy_raw', result['cy']):.0f}) 面积{result['area']}"
                print(f"  帧{frames:5d}  {fps:5.1f} fps  {latency:5.1f} ms  {tag}")
            if args.max_frames and frames >= args.max_frames:
                print(f"[segmotion] 已达 --max-frames {args.max_frames}，退出")
                break
    except KeyboardInterrupt:
        print("\n[segmotion] 已中断")
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if gui:
            cv2.destroyAllWindows()

    if latencies:
        lat = sorted(latencies)
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"\n[segmotion] 统计：{frames} 帧，检出 {det_frames} 帧 "
              f"({100.0 * det_frames / max(1, frames):.1f}%)")
        print(f"[segmotion] 单帧处理  p50={p50:.1f} ms  p95={p95:.1f} ms  "
              f"→ 理论上限 {1000.0 / max(p95, 1e-6):.1f} fps")
        if args.record:
            print(f"[segmotion] 已录制: {args.record}")
    return 0




class SegmotionNode(object):
    def __init__(self):
        rospy.init_node("segmotion_node")

        # ---------------- 参数
        self.color = rospy.get_param("~color", "red")
        if self.color not in COLOR_HSV:
            rospy.logerr("~color=%s 无效，可选 %s", self.color, list(COLOR_HSV))
            raise ValueError("invalid color")
        image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.publish_mask = bool(rospy.get_param("~publish_mask", True))
        self.publish_overlay = bool(rospy.get_param("~publish_overlay", True))
        # 语义图（喂给 diff_land/run.py 的格式，见文件头说明）
        self.publish_semantic = bool(rospy.get_param("~publish_semantic", True))
        self.semantic_topic = rospy.get_param("~semantic_topic", "/semantic/image")
        self.semantic_h = int(rospy.get_param("~semantic_h", 48))
        self.semantic_w = int(rospy.get_param("~semantic_w", 64))
        # 值尺度：255.0 → 0/255（默认，与 diff_land 的 semantic_color.py 格式一致）
        #          1.0   → 0/1（若下游脚本自己做了 /255，会把它再缩小，注意别重复归一化）
        self.semantic_scale = float(rospy.get_param("~semantic_scale", 255.0))

        self.detector = ColorBucketDetector(
            self.color,
            proc_width=int(rospy.get_param("~proc_width", 640)),
            min_area=int(rospy.get_param("~min_area", 100)),
            min_solidity=float(rospy.get_param("~min_solidity", 0.70)),
            max_elongation=float(rospy.get_param("~max_elongation", 3.0)),
            smooth=float(rospy.get_param("~smooth", 0.5)),
            confirm=int(rospy.get_param("~confirm_frames", 2)),
            miss=int(rospy.get_param("~miss_frames", 5)),
            reacquire=float(rospy.get_param("~reacquire", 0.25)),
            temporal=bool(rospy.get_param("~temporal", True)),
            max_ellipse_residual=float(rospy.get_param("~max_ellipse_residual", 0.075)),
            ellipse_residual_k=float(rospy.get_param("~ellipse_residual_k", 0.75)),
        )

        # ---------------- 通信
        self.bridge = CvBridge()
        self.pub_mask = rospy.Publisher("~mask", Image, queue_size=1)
        self.pub_overlay = rospy.Publisher("~overlay", Image, queue_size=1)
        self.pub_target = rospy.Publisher("~target", Float32MultiArray, queue_size=1)
        self.pub_locked = rospy.Publisher("~locked", Bool, queue_size=1)
        self.pub_semantic = rospy.Publisher(self.semantic_topic, Image, queue_size=1)
        # 图像话题：queue_size=1 + 大 buff_size，否则大图会被丢/积延迟
        rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber("~set_color", String, self.set_color_cb, queue_size=1)

        # ---------------- 统计
        self.frames = 0
        self.det_frames = 0
        self.t_prev = time.time()
        self.fps = 0.0
        self.proc_ms = 0.0

        rospy.loginfo("[segmotion] 颜色开关=%s（其他颜色屏蔽）  处理宽度=%dpx  形态学核 open=%d/close=%d",
                      self.color.upper(), self.detector.proc_width,
                      self.detector.k_open, self.detector.k_close)
        rospy.loginfo("[segmotion] 解释器 python %s (%s)",
                      sys.version.split()[0], sys.executable)
        rospy.loginfo("[segmotion] 订阅 %s → 发布 ~mask / ~overlay / ~target / ~locked",
                      image_topic)
        if self.publish_semantic:
            rospy.loginfo("[segmotion] 语义图 → %s，尺寸 (%d,%d)，mono8，值域 0/%.0f",
                          self.semantic_topic, self.semantic_h,
                          self.semantic_w, self.semantic_scale)

    # ------------------------------------------------------------ 回调
    def set_color_cb(self, msg):
        """运行期切换抛掷物颜色，无需重启节点。"""
        color = msg.data.strip().lower()
        try:
            self.detector.set_color(color)
        except ValueError as e:
            rospy.logwarn("[segmotion] 切换颜色失败: %s", e)
            return
        self.color = color
        rospy.loginfo("[segmotion] 颜色开关已切换为 %s（其他颜色屏蔽）", color.upper())

    def image_cb(self, msg):
        t0 = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(5.0, "[segmotion] cv_bridge 转换失败: %s", e)
            return

        mask, res = self.detector.detect(frame)

        h, w = frame.shape[:2]

        if self.publish_semantic:
            # 语义图：与 diff_land 的 semantic_color.py 输出完全一致 ——
            # mono8、shape (48,64)、值 0/255、header 透传，可直接被 run.py 订阅的
            # /semantic/image 消费（run.py 用 img>0 求质心，再做归一化 u/v）
            sem = cv2.resize(mask, (self.semantic_w, self.semantic_h),
                             interpolation=cv2.INTER_NEAREST)
            sem = (sem > 0).astype(np.float32) * self.semantic_scale
            sem = np.clip(sem, 0, 255).astype(np.uint8)
            s = self.bridge.cv2_to_imgmsg(sem, encoding="mono8")
            s.header = msg.header
            self.pub_semantic.publish(s)

        # 线性插值放大后再二值化：边界比最近邻放大平滑（消除块状锯齿）；语义图仍用最近邻
        mask_full = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 127)
        mask_full = mask_full.astype(np.uint8) * 255

        if self.publish_mask:
            m = self.bridge.cv2_to_imgmsg(mask_full, encoding="mono8")
            m.header = msg.header
            self.pub_mask.publish(m)

        if self.publish_overlay:
            o = self.bridge.cv2_to_imgmsg(self._draw(frame, mask_full, res), encoding="bgr8")
            o.header = msg.header
            self.pub_overlay.publish(o)

        self.pub_target.publish(self._target_msg(res))
        self.pub_locked.publish(Bool(data=bool(res and res["locked"])))

        # 统计与节流日志
        self.frames += 1
        self.det_frames += res is not None
        now = time.time()
        self.proc_ms = (now - t0) * 1000.0
        dt = now - self.t_prev
        self.t_prev = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        rospy.loginfo_throttle(
            2.0, "[segmotion] %.1f fps, %.1f ms/帧, 检出 %d/%d 帧, 目标=%s",
            self.fps, self.proc_ms, self.det_frames, self.frames,
            "无" if res is None else
            "(%.0f,%.0f) 面积%d %s" % (res.get("cx_raw", res["cx"]),
                                       res.get("cy_raw", res["cy"]), res["area"],
                                       "LOCKED" if res["locked"] else "HOLD"))

    # ------------------------------------------------------------ 工具
    def _target_msg(self, res):
        """[cx_raw, cy_raw, area, locked, cx, cy, color_id]；无目标用 -1 占位。"""
        if res is None:
            data = [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, float(COLOR_ID[self.color])]
        else:
            data = [float(res.get("cx_raw", res["cx"])), float(res.get("cy_raw", res["cy"])),
                    float(res["area"]), 1.0 if res["locked"] else 0.0,
                    float(res["cx"]), float(res["cy"]), float(res["color_id"])]
        return Float32MultiArray(data=data)

    def _draw(self, frame, mask_full, res):
        """叠加色块 + 轮廓 + 包围框 + 准星（原始位置）+ 空心圆（平滑位置）+ HUD。"""
        vis = frame.copy()
        color_bgr = COLOR_BGR[self.color]
        if res is not None:
            on = mask_full > 0
            vis[on] = (0.55 * vis[on] + 0.45 * np.array(color_bgr)).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color_bgr, 3)
            if res["bbox"][2] > 0:
                x, y, w, h = res["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
            if "cx_raw" in res:
                cv2.drawMarker(vis, (int(res["cx_raw"]), int(res["cy_raw"])),
                               (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
            sx, sy = int(res["cx"]), int(res["cy"])
            if "cx_raw" not in res or np.hypot(sx - res["cx_raw"], sy - res["cy_raw"]) > 4:
                cv2.circle(vis, (sx, sy), 14, color_bgr, 3)
            state = ("LOCKED " if res["locked"] else "HOLD   ") + \
                    "(%.0f,%.0f) A=%d" % (res.get("cx_raw", res["cx"]),
                                          res.get("cy_raw", res["cy"]), res["area"])
        else:
            state = "NO TARGET"

        lines = ["COLOR : %s   (other colors masked out)" % self.color.upper(),
                 "STATE : %s" % state,
                 "FPS   : %5.1f    PROC: %5.1f ms/frame" % (self.fps, self.proc_ms)]
        for i, text in enumerate(lines):
            cv2.putText(vis, text, (16, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(vis, text, (16, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return vis



# ==================================================================== 入口分发

def _cli_to_ros_params(argv):
    """把 --color red 这类写法转成 ROS 私参 _color:=red，兼容两种命令行风格。"""
    aliases = {
        '--color': 'color', '--image-topic': 'image_topic', '--proc-width': 'proc_width',
        '--min-area': 'min_area', '--min-solidity': 'min_solidity',
        '--max-elongation': 'max_elongation', '--smooth': 'smooth',
        '--semantic-topic': 'semantic_topic', '--semantic-scale': 'semantic_scale',
        '--max-ellipse-residual': 'max_ellipse_residual',
        '--ellipse-residual-k': 'ellipse_residual_k',
    }
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in aliases and i + 1 < len(argv):
            out.append('_%s:=%s' % (aliases[a], argv[i + 1]))   # 前导下划线 = ROS 私有参数
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def main_ros():
    """ROS 模式：起节点、发话题。"""
    if not ROS_AVAILABLE:
        print("[segmotion_full] ROS 模式不可用：当前 python 无法 import rospy/cv_bridge.\n"
              "  请先激活装了 ROS 依赖的 python 环境，并 source /opt/ros/<distro>/setup.bash",
              file=sys.stderr)
        return 1
    sys.argv = _cli_to_ros_params([a for a in sys.argv if a != "--ros"])
    try:
        SegmotionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    return 0


def main():
    if "--probe" in sys.argv[:3]:    # 查设备号不需要 ROS 也不需要窗口
        probe_devices()
        return 0
    if "--ros" in sys.argv[:3]:      # --ros 通常紧跟脚本名
        return main_ros()
    return main_standalone()


if __name__ == "__main__":
    sys.exit(main() or 0)
