#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import platform

# ========== 平台检测 ==========
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# ========== 关键修复：必须在导入任何其他库之前设置 ==========
if IS_WINDOWS:
    os.environ['QT_QPA_PLATFORM'] = 'windows'
else:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
# ===========================================================

# ========== 根据平台配置串口和摄像头 ==========
if IS_WINDOWS:
    # Windows 配置
    WEIGHT_PORT = 'COM19'  # 称重串口
    VOICE_PORT = 'COM10'   # 语音串口
    CAMERA_ID = 0          # Windows 默认摄像头
    print(f"[系统] 运行平台: Windows")
    print(f"[配置] 称重串口: {WEIGHT_PORT}")
    print(f"[配置] 语音串口: {VOICE_PORT}")
    print(f"[配置] 摄像头ID: {CAMERA_ID}")
else:
    # Linux (Muse Pi Pro) 配置
    WEIGHT_PORT = '/dev/ttyS10'    # 称重串口（接收重量+语音控制命令）
    VOICE_PORT = '/dev/ttyUSB0'    # 语音串口 - 通过 USB 转 TTL 模块连接（9600）
    CAMERA_ID = 20                 # Muse Pi Pro USB 摄像头
    print(f"[系统] 运行平台: Linux (Muse Pi Pro)")
    print(f"[配置] 称重串口: {WEIGHT_PORT}")
    print(f"[配置] 语音串口: {VOICE_PORT}")
    print(f"[配置] 摄像头ID: {CAMERA_ID}")
# ===========================================


# ========== 第1步：导入PyQt5 ==========
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QFrame, QLineEdit, QGridLayout, QSizePolicy,
                             QInputDialog, QMessageBox, QComboBox, QDialog, QDialogButtonBox,
                             QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter, QListWidget, QListWidgetItem, QCheckBox, QScrollArea,
                             QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QThread, QMutex
from PyQt5.QtGui import QPixmap, QFont, QImage, QColor
# =================================================================

# ========== 第2步：导入标准库 ==========
import sqlite3
import time
import tempfile
import subprocess
from datetime import datetime

# =====================================

# ========== 第3步：导入串口库 ==========
try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("提示: 安装pyserial: pip install pyserial")

# ========== 第4步：导入摄像头和AI库 ==========
try:
    import cv2
    import numpy as np

    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("提示: 安装opencv-python: pip install opencv-python")

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("提示: 安装onnxruntime: pip install onnxruntime")

# 尝试导入二维码库
try:
    from pyzbar import pyzbar

    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("提示: 安装pyzbar以支持二维码扫描: pip install pyzbar")
# =========================================

# ========== ONNX模型配置 ==========
MODEL_PATH = './model/best6.onnx'
MODEL_INPUT_SIZE = (320, 320)
INFERENCE_INTERVAL = 3   # 摄像头仅 ~5fps，间隔过大会拖慢结算
# 推理前中心裁剪比例：去掉画面边缘的深色桌面/杂物
# 实测（25 张实拍）准确率 60% -> 100%，见技术文档
CENTER_CROP = 0.60

# 只允许这 5 类参与判定：屏蔽掉柠檬/番茄等易混淆类别，避免梨被判成柠檬
# 对应 INGREDIENTS 下标：0 Bitter melon, 1 Carrot, 2 Green apple, 3 Lemon,
# 4 Pomegranate, 5 Tomato, 6 apple, 7 eggplant, 8 orange, 9 peach,
# 10 pear, 11 pumpkin, 12 watermelon
ALLOWED_IDS = (6, 9, 10, 11, 12)

# 食材名称（13个类别，顺序必须与训练时 sorted() 一致：大写在前）
INGREDIENTS = [
    "Bitter melon", "Carrot", "Green apple",
    "Lemon", "Pomegranate", "Tomato",
    "apple", "eggplant", "orange",
    "peach", "pear", "pumpkin",
    "watermelon"
]

INGREDIENTS_CN = [
    "苦瓜", "胡萝卜", "青苹果",
    "柠檬", "石榴", "番茄",
    "苹果", "茄子", "橙子",
    "桃子", "梨", "南瓜",
    "西瓜"
]

ID_TO_NAME_EN = {i: INGREDIENTS[i] for i in range(len(INGREDIENTS))}
ID_TO_NAME_CN = {i: INGREDIENTS_CN[i] for i in range(len(INGREDIENTS_CN))}


# ------------------------------
# TTS语音播报线程
# ------------------------------
class TTSThread(QThread):
    """语音播报线程"""

    def __init__(self, port=None, baudrate=9600, enabled=True):
        super().__init__()
        self.port = port if port else VOICE_PORT
        self.baudrate = baudrate
        self.enabled = enabled
        self._last_spoken = ""
        self.serial_port = None
        self.running = True
        self.speak_queue = []
        self.mutex = QMutex()
        self.reconnect_delay = 2000
        self._cooldown_ms = 600       # 两条播报之间的最小间隔

    def _connect(self):
        """打开串口（仅用指定波特率，不切换）"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                self.serial_port = None

            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"[语音] 串口 {self.port} 打开成功（波特率{self.baudrate}），语音功能已启用")

            # 初始化TTS模块
            try:
                self.serial_port.write(b'<V>5\r\n')
                time.sleep(0.2)
                self.serial_port.write(b'<S>5\r\n')
                time.sleep(0.2)
            except:
                pass
            return True
        except Exception as e:
            print(f"[语音] 串口 {self.port} 打开失败: {e}")
            return False
            return False

    def _close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        self.serial_port = None

    def run(self):
        """TTS线程主循环"""
        if not self.enabled:
            return

        if not SERIAL_AVAILABLE:
            print("[语音] pyserial未安装，语音功能不可用")
            return

        while self.running:
            if not self.serial_port or not self.serial_port.is_open:
                if not self._connect():
                    self.msleep(self.reconnect_delay)
                    continue

            self.mutex.lock()
            if self.speak_queue:
                text = self.speak_queue.pop(0)
                self.mutex.unlock()
                self._speak(text)
                time.sleep(self._cooldown_ms / 1000.0)  # 播报间最小间隔
            else:
                self.mutex.unlock()
            self.msleep(100)

    def _speak(self, text):
        """实际发送语音命令"""
        if not self.enabled or not self.serial_port:
            return

        try:
            # 将文本转换为GBK编码（大多数语音模块使用GBK）
            try:
                gbk_text = text.encode('gbk')
            except:
                gbk_text = text.encode('utf-8')

            # 发送数据，以\r\n结尾
            self.serial_port.write(gbk_text)
            self.serial_port.write(b'\r\n')
            self.serial_port.flush()
            print(f"[语音] 播报: {text}")

            # 等待播报完成（根据文字长度估算：中文约 0.3s/字）
            est_ms = max(500, len(text) * 300)
            time.sleep(est_ms / 1000.0)
        except Exception as e:
            print(f"[语音] 播报失败: {e}")

    def speak(self, text):
        """添加到播报队列（自动去重合并，防止重叠播报）"""
        if not self.enabled:
            return
        self._last_spoken = text
        self.mutex.lock()
        # 合并：若新文本与队尾内容高度重叠，替换队尾为更丰富的一条
        if self.speak_queue:
            last = self.speak_queue[-1]
            # 简单规则：队尾含于新文本 或 新文本含于队尾 → 保留更长的
            if last in text:
                self.speak_queue[-1] = text
                self.mutex.unlock()
                return
            if text in last:
                self.mutex.unlock()
                return
            # 队尾和新文本为同款"商品单价+总价"类型 → 用新文本替换
            if last.startswith(text[:2]) and ('单价' in text and '单价' in last):
                self.speak_queue[-1] = text
                self.mutex.unlock()
                return
        # 队列上限 3 条，超出删除最早
        while len(self.speak_queue) >= 3:
            self.speak_queue.pop(0)
        self.speak_queue.append(text)
        self.mutex.unlock()

    def repeat_last(self):
        """重复上一条播报"""
        if not self.enabled:
            return
        text = getattr(self, '_last_spoken', '')
        if text:
            self.speak(text)

    def set_enabled(self, enabled):
        """设置是否启用语音"""
        self.enabled = enabled
        if not enabled:
            self.mutex.lock()
            self.speak_queue.clear()
            self.mutex.unlock()


    def stop(self):
        self.running = False
        self._close()
        self.wait()


# ------------------------------
# 推理线程（异步ONNX推理）
# ------------------------------
class InferenceThread(QThread):
    result_ready = pyqtSignal(int, float, list, object)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.running = True
        self.mutex = QMutex()
        self.frame_queue = []
        self.max_queue_size = 1
        # 多帧平滑：对最近若干帧的概率取平均，抑制单帧抖动
        self.prob_history = []
        self.history_len = 3    # 只保留最近 3 帧概率，压低等待时间

    def init_session(self):
        if ONNX_AVAILABLE and os.path.exists(self.model_path):
            try:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # K1 有 8 核：4 线程做算子内并行，1 线程做算子间（实测最优）
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.session = ort.InferenceSession(
                    self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                print(f"[AI] ONNX模型加载成功 (优化已启用)")
                return True
            except Exception as e:
                print(f"[AI] ONNX模型加载失败: {e}")
                return False
        return False

    @staticmethod
    def _crop_to_plate(image, margin=0.06):
        """检测白色托板并裁剪到它的外接框；找不到返回 None。

        托板 = 画面中最大的低饱和高亮区域。裁到板上后，板外的桌面、
        其他商品都会被排除，实测置信度可达 99%+。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        m = ((s < 70) & (v > 110)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        H, W = image.shape[:2]
        if w * h < 0.04 * W * H or w * h > 0.98 * W * H:
            return None

        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        return image[y0:y1, x0:x1]

    @staticmethod
    def _crop_to_object(image, margin=0.10, min_ratio=0.03):
        """在给定图内找最大的高饱和目标（商品本身）并紧贴裁剪。

        托板裁剪后，板边缘可能还带着旁边水果；再按颜色定位主体，
        只把商品本身送进模型。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        m = ((s > 100) & (v > 120)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        H, W = image.shape[:2]
        if cv2.contourArea(c) < min_ratio * W * H:
            return None

        x, y, w, h = cv2.boundingRect(c)
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        if (x1 - x0) < 32 or (y1 - y0) < 32:
            return None
        return image[y0:y1, x0:x1]

    def preprocess_image(self, image):
        # 第一步：裁到白色托板，去掉板外的桌面
        plate = self._crop_to_plate(image)
        if plate is not None and plate.size > 0:
            image = plate

        # 第二步：在板内定位商品本体并紧贴裁剪，去掉板边带进来的其他水果
        obj = self._crop_to_object(image)
        if obj is not None and obj.size > 0:
            image = obj
        elif plate is None:
            # 兜底：中心裁剪
            h, w = image.shape[:2]
            ch, cw = int(h * CENTER_CROP), int(w * CENTER_CROP)
            if ch >= 32 and cw >= 32:
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                image = image[y0:y0 + ch, x0:x0 + cw]

        blob = cv2.dnn.blobFromImage(image, 1.0/255.0, MODEL_INPUT_SIZE, swapRB=True)
        return blob

    def add_frame(self, frame):
        if self.session is None:
            return
        self.mutex.lock()
        if len(self.frame_queue) >= self.max_queue_size:
            self.frame_queue.pop(0)
        self.frame_queue.append(frame.copy())
        self.mutex.unlock()

    def run(self):
        if not self.init_session():
            print("[AI] 推理线程初始化失败")
            return

        while self.running:
            frame = None
            self.mutex.lock()
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
            self.mutex.unlock()

            if frame is not None:
                try:
                    input_tensor = self.preprocess_image(frame)
                    result = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

                    if len(result.shape) == 2:
                        probs = result[0]
                    else:
                        probs = result.flatten()

                    # 多帧平滑：单帧判断会抖，取最近几帧的平均概率
                    self.prob_history.append(probs.astype(np.float32))
                    if len(self.prob_history) > self.history_len:
                        self.prob_history.pop(0)
                    smoothed = np.mean(np.stack(self.prob_history), axis=0)

                    # 只保留白名单类别，其余置零（屏蔽易混淆类）
                    if ALLOWED_IDS:
                        mask = np.zeros_like(smoothed)
                        for i in ALLOWED_IDS:
                            if 0 <= i < len(smoothed):
                                mask[i] = smoothed[i]
                        total = float(mask.sum())
                        smoothed = mask / total if total > 1e-6 else mask

                    top5_idx = np.argsort(smoothed)[-5:][::-1]
                    top1_idx = top5_idx[0]
                    top1_conf = float(smoothed[top1_idx])

                    self.result_ready.emit(top1_idx, top1_conf, top5_idx.tolist(), smoothed)
                except Exception as e:
                    print(f"[AI] 推理错误: {e}")

            self.msleep(10)

    def reset_history(self):
        """清空多帧历史。

        抓拍瞬间调用：避免把放商品之前的空托盘画面平均进来。
        """
        self.prob_history = []

    def stop(self):
        self.running = False
        self.wait()


# ------------------------------
# 摄像头线程
# ------------------------------
class CameraThread(QThread):
    image_update = pyqtSignal(QImage, object)
    capture_finished = pyqtSignal(str, list, list)
    recog_settled = pyqtSignal()  # 抓拍后推理已追上，可以取结果了

    def __init__(self, camera_id=None):
        super().__init__()
        self.camera_id = camera_id if camera_id is not None else CAMERA_ID
        self.running = False
        self.cap = None
        self.current_frame = None
        self.mutex = QMutex()
        self.capture_request = False
        self.capture_path = ""
        self.recognize_on_capture = True
        self.frame_count = 0
        self.inference_thread = None
        self.current_recognition = None
        self.current_confidence = 0.0
        self.current_top5 = []
        self._settle_count = 0
        self._settle_target = 3
        # 稳定判定：top1 连续一致 + 置信度够 才结算
        self._last_top1 = None
        self._stable_count = 0
        self._frames_since_capture = 0
        self._max_wait_frames = 4
        self._settled = False
        self._fast_mode = False   # 抓拍后进入快速模式：每个新帧都送推理

    def init_inference(self):
        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            self.inference_thread = InferenceThread(MODEL_PATH)
            self.inference_thread.result_ready.connect(self.on_inference_result)
            self.inference_thread.start()
            return True
        return False

    def on_inference_result(self, top1_idx, confidence, top5_indices, all_probs):
        self.current_recognition = top1_idx
        self.current_confidence = confidence
        self.current_top5 = []
        for idx in top5_indices[:5]:
            self.current_top5.append({
                'id': idx,
                'name_en': ID_TO_NAME_EN.get(idx, "Unknown"),
                'name_cn': ID_TO_NAME_CN.get(idx, "未知"),
                'confidence': float(all_probs[idx])
            })

        # 稳定判定：连续 3 帧 top1 相同且置信度 >= 0.5，或等满 12 帧
        self._frames_since_capture += 1
        if top1_idx == self._last_top1:
            self._stable_count += 1
        else:
            self._last_top1 = top1_idx
            self._stable_count = 1

        if not self._settled:
            # 快速结算：连续 2 帧一致即出结果，最多等 3 帧
            if ((self._stable_count >= 2 and confidence >= 0.5)
                    or self._frames_since_capture >= 3):
                self._settled = True
                self._fast_mode = False
                self.recog_settled.emit()

    def get_current_frame(self):
        self.mutex.lock()
        frame = self.current_frame.copy() if self.current_frame is not None else None
        self.mutex.unlock()
        return frame

    def run(self):
        self.running = True
        if CAMERA_AVAILABLE:
            # 保持默认后端：之前强制 V4L2+MJPG 反而把帧率降到 2.5fps
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"[摄像头] 警告: 无法打开摄像头 {self.camera_id}")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[摄像头] 采集参数: {real_w}x{real_h} @ {real_fps:.0f}fps")

            for i in range(10):
                self.cap.read()
            print(f"[摄像头] 摄像头 {self.camera_id} 已启动")

            fps_counter = 0
            fps_t0 = time.time()

            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))

                    self.mutex.lock()
                    self.current_frame = frame.copy()
                    self.mutex.unlock()

                    if self.inference_thread and (
                            self._fast_mode
                            or self.frame_count % INFERENCE_INTERVAL == 0):
                        self.inference_thread.add_frame(frame)

                    self.frame_count += 1

                    display_frame = frame
                    if self.current_recognition is not None and self.current_top5:
                        overlay = display_frame.copy()
                        name_en = ID_TO_NAME_EN.get(self.current_recognition, "?")
                        line1 = f"{name_en}  {self.current_confidence:.1%}"

                        sub_lines = []
                        for item in self.current_top5[1:4]:
                            en = item['name_en']
                            sub_lines.append(f"{en}  {item['confidence']:.1%}")

                        bar_h = 30 + 22 * len(sub_lines)
                        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], bar_h), (0, 0, 0), -1)
                        display_frame = cv2.addWeighted(overlay, 0.55, display_frame, 0.45, 0)

                        cv2.putText(display_frame, line1, (10, 22),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)

                        y = 46
                        for line in sub_lines:
                            cv2.putText(display_frame, line, (14, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                            y += 22

                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    # QImage 不拷贝数据，必须 .copy() 后才安全地跨线程传递
                    qt_image = QImage(rgb_frame.data, w, h,
                                      bytes_per_line, QImage.Format_RGB888).copy()

                    self.image_update.emit(qt_image, self.current_frame)

                    if self.capture_request:
                        self.capture_request = False
                        filename = self.capture_path
                        cv2.imwrite(filename, self.current_frame)

                        recognized_names = []
                        confidence_list = []
                        if self.recognize_on_capture and self.current_top5:
                            for item in self.current_top5:
                                recognized_names.append(item['name_cn'])
                                confidence_list.append(item['confidence'])

                        # 重置推理计数，等后续几个推理结果到达后再通知识别已稳定
                        if self.recognize_on_capture:
                            self._settle_count = 0
                            self._frames_since_capture = 0
                            self._stable_count = 0
                            self._last_top1 = None
                            self._settled = False
                            self._fast_mode = True   # 抓拍后每个新帧都送推理
                            # 清空多帧历史：从抓拍这一刻重新累积，排除空盘画面
                            if getattr(self, 'inference_thread', None) is not None:
                                self.inference_thread.reset_history()

                        print(f"[摄像头] 抓拍完成，识别到 {len(recognized_names)} 个结果")
                        self.capture_finished.emit(filename, recognized_names, confidence_list)

                # 帧率统计（每 2 秒打印一次）
                fps_counter += 1
                now = time.time()
                if now - fps_t0 >= 2.0:
                    print(f"[摄像头] 实际帧率 {fps_counter / (now - fps_t0):.1f} fps")
                    fps_counter = 0
                    fps_t0 = now

                self.msleep(2)
        else:
            while self.running:
                self.msleep(500)

    def request_capture(self, filepath, recognize=True):
        self.mutex.lock()
        self.capture_path = filepath
        self.capture_request = True
        self.recognize_on_capture = recognize
        self.mutex.unlock()


    def get_top5_results(self):
        return self.current_top5

    def stop(self):
        self.running = False
        if self.inference_thread:
            self.inference_thread.stop()
        if self.cap:
            self.cap.release()
        self.wait()


# ------------------------------
# 串口通信线程（接收重量数据）- 优化版
# ------------------------------
class SerialThread(QThread):
    weight_updated = pyqtSignal(float)
    connection_status = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)

    # TTS 语音控制信号（内嵌于称重串口 AA xx FE 55 协议）
    tts_disable_signal = pyqtSignal()
    tts_enable_signal = pyqtSignal()
    pending_remove_last_signal = pyqtSignal()
    tare_signal = pyqtSignal()
    open_payment_signal = pyqtSignal()
    start_qr_scan_signal = pyqtSignal()
    calc_total_signal = pyqtSignal()
    repeat_last_signal = pyqtSignal()

    # LCD→PC 按钮事件信号
    lcd_button_signal = pyqtSignal(int)

    # 有效语音控制命令字节
    VALID_VOICE_CMDS = frozenset({0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A})

    _CMD_SIGNALS = None  # lazy init in __init__

    def __init__(self, port=None, baudrate=9600):
        super().__init__()
        self.port = port if port else WEIGHT_PORT
        self.baudrate = baudrate
        self.running = False
        self.serial_port = None
        self.buffer = b''
        self.current_weight = 0.0
        self.reconnect_delay = 2000  # 重连延迟（毫秒）
        self.read_timeout = 0.5  # 读取超时（秒）
        self.frame_timeout = 1.0  # 帧超时（秒）
        self.last_activity = time.time()
        self.consecutive_errors = 0
        self._last_logged_weight = -1
        self._last_error_time = 0

        # 语音控制命令 → 信号映射
        self._CMD_SIGNALS = {
            0x01: self.tts_disable_signal,
            0x04: self.tts_enable_signal,
            0x05: self.pending_remove_last_signal,
            0x06: self.tare_signal,
            0x07: self.open_payment_signal,
            0x08: self.start_qr_scan_signal,
            0x09: self.calc_total_signal,
            0x0A: self.repeat_last_signal,
        }

    def calculate_checksum(self, data):
        """计算累加和（低8位）"""
        return sum(data) & 0xFF

    def parse_frame(self, frame):
        """
        解析串口数据帧
        协议格式:
        帧头(2): 0xAA 0x55
        串口号(1): 1-255
        帧长度(1): 从串口号到校验的字节数 = N + 3 (范围 3-253)
        数据域(N): N = 帧长度 - 3
        校验(1): 累加和低8位 (校验范围: 串口号+帧长度+数据域)
        帧尾(2): 0x0D 0x0A

        整帧实际长度 = N + 7 = 帧长度 + 4
        """
        frame_len = len(frame)

        # 最小帧长度: 头(2)+串口(1)+长度(1)+校验(1)+尾(2) = 7字节
        if frame_len < 7:
            return False, None, f"帧长度不足: {frame_len}"

        # 检查帧头
        if frame[0] != 0xAA or frame[1] != 0x55:
            return False, None, "帧头错误"

        # 检查帧尾
        if frame[-2] != 0x0D or frame[-1] != 0x0A:
            return False, None, "帧尾错误"

        # 获取串口号
        com_port = frame[2]
        if com_port < 1 or com_port > 255:
            return False, None, f"串口号无效: {com_port}"

        # 获取帧长度
        len_value = frame[3]
        if len_value < 3 or len_value > 253:
            return False, None, f"无效长度字段: {len_value}"

        n = len_value - 3
        expected_len = len_value + 4

        if frame_len != expected_len:
            return False, None, f"长度不匹配: 期望{expected_len}, 实际{frame_len}"

        data_field = frame[4:4 + n]
        received_checksum = frame[4 + n]
        checksum_data = frame[2:4 + n]
        calc_checksum = self.calculate_checksum(checksum_data)

        if calc_checksum != received_checksum:
            return False, None, f"校验和错误: 计算={calc_checksum:02X}, 接收={received_checksum:02X}"

        # 解析重量数据
        if n >= 2:
            weight_raw = (data_field[0] << 8) | data_field[1]

            if weight_raw > 50000:
                return False, None, f"忽略超范围重量: {weight_raw}g"
            if weight_raw < 0:
                return False, None, f"重量为负数: {weight_raw}g"

            weight_kg = weight_raw / 1000.0
            return True, weight_kg, f"重量={weight_kg:.3f}kg"
        elif n == 0:
            return True, None, "心跳包"
        elif n == 1 and com_port == 1:
            # LCD→PC 按钮事件: data[0] = command byte
            return True, None, f"LCD按钮:0x{data_field[0]:02X}", data_field[0]
        else:
            return True, None, f"无重量数据"

    def read_frame(self, voice_callback=None):
        """
        从串口读取一帧完整数据（阻塞直到收到帧尾或超时）
        同时检测内嵌的语音控制命令 AA CMD FE 55。

        参数:
            voice_callback: callable(cmd_byte) 当检测到语音命令时回调
        返回: bytes 完整帧数据（已剔除语音命令字节），超时返回 None
              若仅检测到语音命令（无待解析数据），返回 b'' 以加速循环
        """
        start_time = time.time()
        frame_data = b''

        while (time.time() - start_time) < self.frame_timeout:
            if not self.serial_port or not self.serial_port.is_open:
                return None

            try:
                byte = self.serial_port.read(1)
                if byte:
                    frame_data += byte
                    self.last_activity = time.time()

                    # 检测内嵌语音控制命令 AA CMD FE 55（4 字节）
                    if len(frame_data) >= 4:
                        tail = frame_data[-4:]
                        if (tail[0] == 0xAA and tail[2] == 0xFE and
                                tail[3] == 0x55 and tail[1] in self.VALID_VOICE_CMDS):
                            cmd = tail[1]
                            print(f"[串口] 检测到语音控制命令: 0x{cmd:02X}")
                            if voice_callback:
                                voice_callback(cmd)
                            # 从缓冲区剔除这 4 个命令字节
                            frame_data = frame_data[:-4]
                            # 剔除后若缓冲区已空且串口无新数据，立即返回避免阻塞
                            if len(frame_data) == 0 and self.serial_port.in_waiting == 0:
                                return b''
                            # 剔除后若缓冲区已有完整帧（\r\n），直接返回
                            if len(frame_data) >= 2 and frame_data[-2:] == b'\r\n':
                                return frame_data
                            continue

                    # 检查是否收到帧尾 \r\n
                    if len(frame_data) >= 2 and frame_data[-2:] == b'\r\n':
                        return frame_data
                else:
                    self.msleep(5)
            except serial.SerialException as e:
                error_msg = str(e)
                if "device reports readiness" not in error_msg:
                    print(f"[串口] 读取异常: {e}")
                return None
            except Exception as e:
                print(f"[串口] 读取错误: {e}")
                return None

        # 超时
        return None

    def _connect_serial(self):
        """尝试打开串口"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.read_timeout,
                write_timeout=1
            )
            self.connection_status.emit(True, f"{self.port} ({self.baudrate}bps)")
            print(f"[串口] {self.port} 打开成功")
            self.consecutive_errors = 0
            self.last_activity = time.time()
            return True
        except Exception as e:
            error_msg = str(e)
            self.connection_status.emit(False, f"失败: {error_msg}")
            print(f"[串口] 打开失败: {error_msg}")
            return False

    def _close_serial(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        self.serial_port = None

    def _on_voice_cmd(self, cmd):
        """语音控制命令回调：路由到对应信号"""
        signal = self._CMD_SIGNALS.get(cmd)
        if signal:
            signal.emit()

    def run(self):
        self.running = True

        if not SERIAL_AVAILABLE:
            self.error_occurred.emit("请安装pyserial: pip install pyserial")
            return

        # 尝试连接串口
        if not self._connect_serial():
            # 连接失败，继续尝试重连
            pass

        while self.running:
            # 检查串口是否可用
            if not self.serial_port or not self.serial_port.is_open:
                # 尝试重连
                self.error_occurred.emit("串口未连接，尝试重连...")
                self._close_serial()
                self.msleep(self.reconnect_delay)
                if self._connect_serial():
                    self.buffer = b''  # 清空缓冲区
                continue

            try:
                # 读取一帧完整数据（同步检测语音控制命令）
                frame = self.read_frame(voice_callback=self._on_voice_cmd)

                if frame:
                    # 收到数据，解析
                    result = self.parse_frame(frame)
                    success, weight, msg = result[0], result[1], result[2]
                    lcd_cmd = result[3] if len(result) > 3 else None

                    if success:
                        if lcd_cmd is not None:
                            # LCD 按钮事件 → 信号通知 UI 层
                            print(f"[串口] {msg}")
                            self.lcd_button_signal.emit(lcd_cmd)
                        elif weight is not None:
                            self.current_weight = weight
                            self.weight_updated.emit(weight)
                            # 只在重量变化时打印（避免刷屏）
                            if abs(weight - self._last_logged_weight) > 0.01:
                                print(f"[串口] {msg}")
                                self._last_logged_weight = weight
                                # 心跳包不处理
                    else:
                        # 解析失败，可能是数据错位，打印调试信息（限制频率）
                        if time.time() - self._last_error_time > 5:
                            print(f"[串口] 解析失败: {msg}")
                            if len(frame) <= 64:
                                print(f"[串口] 原始帧: {frame.hex()}")
                            else:
                                print(f"[串口] 原始帧: {frame[:32].hex()}...{frame[-32:].hex()}")
                            self._last_error_time = time.time()
                else:
                    # 超时没有数据，检查连接是否正常
                    if time.time() - self.last_activity > 10:
                        # 超过10秒没有活动，可能是设备断开
                        print("[串口] 长时间无数据，检查连接...")
                        self._close_serial()
                        self.msleep(500)
                        self._connect_serial()

            except serial.SerialException as e:
                error_msg = str(e)
                # 忽略常见的误报错误
                if "device reports readiness" in error_msg:
                    # 这是常见的误报，静默处理
                    pass
                elif "Permission denied" in error_msg:
                    print(f"[串口] 权限错误: {error_msg}")
                    self.error_occurred.emit(f"权限错误: {error_msg}")
                    self._close_serial()
                    self.msleep(5000)  # 权限错误等待5秒
                    self._connect_serial()
                else:
                    print(f"[串口] 异常: {error_msg}")
                    self._close_serial()
                    self.msleep(self.reconnect_delay)
                    self._connect_serial()
            except Exception as e:
                print(f"[串口] 未知错误: {e}")
                self.msleep(100)

        # 线程结束，关闭串口
        self._close_serial()

    def stop(self):
        self.running = False
        self._close_serial()
        self.wait()


    def _write_frame(self, data_bytes):
        """直接发送原始字节帧到 F407"""
        if not self.serial_port or not self.serial_port.is_open:
            print("[串口] 发送失败: 串口未打开")
            return False
        try:
            self.serial_port.write(data_bytes)
            self.serial_port.flush()
            return True
        except Exception as e:
            print(f"[串口] 发送失败: {e}")
            return False

    def send_pending_list(self, pending_items):
        """构建并发送待付款列表帧 (cmd=0x30) 到 F407 LCD"""
        payload = bytearray()

        # n_items (1B)
        n = min(len(pending_items), 10)
        payload.append(n)

        # total (2B, 元×100)
        total = int(sum(item.get('total', 0) for item in pending_items) * 100 + 0.5)
        payload.append((total >> 8) & 0xFF)
        payload.append(total & 0xFF)

        # 每条项目记录
        for i, item in enumerate(pending_items[:n]):
            payload.append(i)  # index

            # 品名：优先取 name_en，否则将中文名转拼音/截断
            name = item.get('name_en', '')
            if not name:
                name = item.get('product_name', '?')[:16]
            name_bytes = name.encode('ascii', errors='replace')[:16]
            payload.append(len(name_bytes))
            payload.extend(name_bytes)

            # 重量(g), uint16
            w = int(item.get('weight', 0) * 1000 + 0.5)
            if w > 65535:
                w = 65535
            payload.append((w >> 8) & 0xFF)
            payload.append(w & 0xFF)

            # 单价(元×100), uint16
            p = int(item.get('price', 0) * 100 + 0.5)
            if p > 65535:
                p = 65535
            payload.append((p >> 8) & 0xFF)
            payload.append(p & 0xFF)

            # 小计(元×100), uint16
            t = int(item.get('total', 0) * 100 + 0.5)
            if t > 65535:
                t = 65535
            payload.append((t >> 8) & 0xFF)
            payload.append(t & 0xFF)

        # 构建完整帧: AA 55 02 len cmd payload cs 0D 0A
        # len = cmd(1) + payload(N) + cs(1)，不含 port 和 len 自身
        frame_len = 2 + len(payload)
        frame = bytearray([0xAA, 0x55, 0x02, frame_len, 0x30])
        frame.extend(payload)
        cs = (0x02 + frame_len + 0x30 + sum(payload)) & 0xFF
        frame.append(cs)
        frame.extend([0x0D, 0x0A])

        print(f"[串口] 发送待付款列表: {n}项, 总额{total/100:.2f}, 帧长{len(frame)}")
        return self._write_frame(bytes(frame))

    def send_clear_list(self):
        """发送清空列表命令 (cmd=0x31) 到 F407 LCD"""
        # AA 55 02 02 31 cs 0D 0A (len=2: cmd+cs)
        cs = (0x02 + 2 + 0x31) & 0xFF
        frame = bytes([0xAA, 0x55, 0x02, 0x02, 0x31, cs, 0x0D, 0x0A])
        print("[串口] 发送清空列表命令")
        return self._write_frame(frame)

    def send_lcd_page_up(self):
        """发送下一页命令 (cmd=0x32) 到 F407 LCD"""
        cs = (0x02 + 2 + 0x32) & 0xFF
        frame = bytes([0xAA, 0x55, 0x02, 0x02, 0x32, cs, 0x0D, 0x0A])
        print("[串口] 发送LCD下一页命令")
        return self._write_frame(frame)

    def send_lcd_page_down(self):
        """发送上一页命令 (cmd=0x33) 到 F407 LCD"""
        cs = (0x02 + 2 + 0x33) & 0xFF
        frame = bytes([0xAA, 0x55, 0x02, 0x02, 0x33, cs, 0x0D, 0x0A])
        print("[串口] 发送LCD上一页命令")
        return self._write_frame(frame)

    def send_tare(self, weight_kg):
        """发送去皮命令 (cmd=0x34) 到 F407，payload为当前重量(克)"""
        weight_grams = int(weight_kg * 1000 + 0.5)
        payload = bytearray()
        payload.append((weight_grams >> 8) & 0xFF)
        payload.append(weight_grams & 0xFF)
        # len = cmd(1) + payload(2) + cs(1) = 4
        cs = (0x02 + 4 + 0x34 + sum(payload)) & 0xFF
        frame = bytearray([0xAA, 0x55, 0x02, 0x04, 0x34])
        frame.extend(payload)
        frame.append(cs)
        frame.extend([0x0D, 0x0A])
        print(f"[串口] 发送去皮命令, 当前重量: {weight_grams}g")
        return self._write_frame(bytes(frame))

    def send_zero(self):
        """发送清零/取消去皮命令 (cmd=0x35) 到 F407 LCD"""
        cs = (0x02 + 2 + 0x35) & 0xFF
        frame = bytes([0xAA, 0x55, 0x02, 0x02, 0x35, cs, 0x0D, 0x0A])
        print("[串口] 发送清零命令")
        return self._write_frame(frame)

    def send_lcd_payment_amount(self, amount_yuan):
        """设置 LCD Screen3 支付金额 (cmd=0x36)"""
        amt = int(amount_yuan * 100 + 0.5)
        if amt > 65535:
            amt = 65535
        payload = bytearray()
        payload.append((amt >> 8) & 0xFF)
        payload.append(amt & 0xFF)
        cs = (0x02 + 4 + 0x36 + sum(payload)) & 0xFF
        frame = bytearray([0xAA, 0x55, 0x02, 0x04, 0x36])
        frame.extend(payload)
        frame.append(cs)
        frame.extend([0x0D, 0x0A])
        print(f"[串口] 发送LCD支付金额: ¥{amount_yuan:.2f}")
        return self._write_frame(bytes(frame))

    def send_lcd_payment_result(self, success):
        """发送支付结果到 LCD Screen3 (cmd=0x37成功/0x38失败)"""
        cmd = 0x37 if success else 0x38
        cs = (0x02 + 2 + cmd) & 0xFF
        frame = bytes([0xAA, 0x55, 0x02, 0x02, cmd, cs, 0x0D, 0x0A])
        label = "成功" if success else "失败"
        print(f"[串口] 发送LCD支付结果: {label}")
        return self._write_frame(bytes(frame))

    def send_lcd_switch_screen(self, screen_num):
        """远程切换 LCD 页面 (cmd=0x39, payload=1B 屏号)"""
        payload = bytearray([screen_num & 0xFF])
        cs = (0x02 + 3 + 0x39 + sum(payload)) & 0xFF
        frame = bytearray([0xAA, 0x55, 0x02, 0x03, 0x39])
        frame.extend(payload)
        frame.append(cs)
        frame.extend([0x0D, 0x0A])
        print(f"[串口] 发送LCD切屏: Screen{screen_num}")
        return self._write_frame(bytes(frame))




# ------------------------------
# 数据库管理类（保持不变）
# ------------------------------
class DatabaseManager:
    def __init__(self, db_name="fruit_vegetables.db"):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name_en TEXT NOT NULL,
                name_cn TEXT NOT NULL,
                price REAL NOT NULL,
                unit TEXT DEFAULT 'kg',
                create_time TEXT,
                update_time TEXT
            )
        ''')

        cursor.execute("PRAGMA table_info(products)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'unit' not in columns:
            cursor.execute("ALTER TABLE products ADD COLUMN unit TEXT DEFAULT 'kg'")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                product_name TEXT,
                weight REAL,
                price REAL,
                total REAL,
                transaction_time TEXT,
                image_path TEXT
            )
        ''')

        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0:
            default_products = [
                ("0001", "apple", "苹果", 8.00),
                ("0002", "Bitter melon", "苦瓜", 8.00),
                ("0003", "Carrot", "胡萝卜", 5.00),
                ("0004", "eggplant", "茄子", 6.00),
                ("0005", "Green apple", "青苹果", 8.00),
                ("0006", "Lemon", "柠檬", 10.00),
                ("0007", "orange", "橙子", 8.00),
                ("0008", "peach", "桃子", 10.00),
                ("0009", "pear", "梨", 8.00),
                ("0010", "Pomegranate", "石榴", 12.00),
                ("0011", "pumpkin", "南瓜", 4.00),
                ("0012", "Tomato", "番茄", 5.00),
                ("0013", "watermelon", "西瓜", 3.00),
            ]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for p in default_products:
                cursor.execute('''
                    INSERT INTO products (id, name_en, name_cn, price, unit, create_time, update_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (p[0], p[1], p[2], p[3], "kg", now, ""))

        conn.commit()
        conn.close()
        print("[数据库] 初始化完成")

    def get_all_products(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name_en, name_cn, price FROM products ORDER BY id")
        products = cursor.fetchall()
        conn.close()
        return products

    def get_product_by_id(self, pid):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (pid,))
        product = cursor.fetchone()
        conn.close()
        return product

    def search_products_by_name(self, name):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        search_name = name.lower().strip()

        name_mapping = {
            'apple': '苹果', 'bitter melon': '苦瓜', 'carrot': '胡萝卜',
            'eggplant': '茄子', 'green apple': '青苹果', 'lemon': '柠檬',
            'orange': '橙子', 'peach': '桃子', 'pear': '梨',
            'pomegranate': '石榴', 'pumpkin': '南瓜', 'tomato': '番茄',
            'watermelon': '西瓜',
        }

        chinese_name = name_mapping.get(search_name, search_name)

        cursor.execute('''
            SELECT * FROM products 
            WHERE LOWER(name_en) LIKE ? 
            OR name_cn LIKE ? 
            OR LOWER(name_en) LIKE ?
            OR name_cn LIKE ?
            OR ? LIKE '%' || LOWER(name_en) || '%'
            OR ? LIKE '%' || name_cn || '%'
        ''', (
            f'%{search_name}%', f'%{search_name}%',
            f'%{chinese_name}%', f'%{chinese_name}%',
            search_name, search_name
        ))
        products = cursor.fetchall()
        conn.close()
        return products

    def smart_search(self, keyword):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name_en, name_cn, price FROM products 
            WHERE id LIKE ? OR name_en LIKE ? OR name_cn LIKE ?
            ORDER BY id LIMIT 20
        ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
        products = cursor.fetchall()
        conn.close()
        return products

    def add_product(self, pid, name_en, name_cn, price, unit="kg"):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO products (id, name_en, name_cn, price, unit, create_time, update_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pid, name_en.lower(), name_cn, price, unit, now, ""))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def update_product(self, pid, **kwargs):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for key, value in kwargs.items():
            if value is not None and key in ['name_en', 'name_cn', 'price', 'unit']:
                cursor.execute(f"UPDATE products SET {key} = ?, update_time = ? WHERE id = ?", (value, now, pid))
        conn.commit()
        conn.close()

    def delete_product(self, pid):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM products WHERE id = ?", (pid,))
        conn.commit()
        conn.close()

    def save_transaction(self, product_id, product_name, weight, price, total, image_path=""):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO transactions (product_id, product_name, weight, price, total, transaction_time, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (product_id, product_name, weight, price, total, now, image_path))
        conn.commit()
        conn.close()

    def get_all_transactions(self, limit=200):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, product_name, weight, price, total, transaction_time "
            "FROM transactions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_daily_total(self, date_str=None):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(total), 0) FROM transactions "
            "WHERE transaction_time LIKE ?", (date_str + '%',))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row[0] else 0.0

    def get_transaction_count(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_transaction_by_id(self, tid):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE id = ?", (tid,))
        conn.commit()
        conn.close()

    def delete_all_transactions(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions")
        conn.commit()
        conn.close()


# ------------------------------
# 商品管理对话框
# ------------------------------
class ProductManageDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("商品管理")
        self.setModal(True)
        self.resize(900, 600)
        self.init_ui()
        self.load_data()

    def init_ui(self):
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("➕ 新增")
        self.btn_edit = QPushButton("✏️ 修改")
        self.btn_delete = QPushButton("🗑️ 删除")
        self.btn_refresh = QPushButton("🔄 刷新")
        for btn in [self.btn_add, self.btn_edit, self.btn_delete, self.btn_refresh]:
            btn.setMinimumHeight(35)
            toolbar.addWidget(btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("编号/名称...")
        self.search_edit.textChanged.connect(self.search_products)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["编号", "英文名", "中文名", "价格(元/kg)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        self.btn_add.clicked.connect(self.add_product)
        self.btn_edit.clicked.connect(self.edit_product)
        self.btn_delete.clicked.connect(self.delete_product)
        self.btn_refresh.clicked.connect(self.load_data)

    def load_data(self):
        self.table.setRowCount(0)
        try:
            products = self.db.get_all_products()
            for row, p in enumerate(products):
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(p[0]))
                self.table.setItem(row, 1, QTableWidgetItem(p[1]))
                self.table.setItem(row, 2, QTableWidgetItem(p[2]))
                self.table.setItem(row, 3, QTableWidgetItem(f"{p[3]:.2f}"))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载数据失败: {str(e)}")

    def search_products(self, keyword):
        if not keyword:
            self.load_data()
            return
        self.table.setRowCount(0)
        try:
            products = self.db.smart_search(keyword)
            for row, p in enumerate(products):
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(p[0]))
                self.table.setItem(row, 1, QTableWidgetItem(p[1]))
                self.table.setItem(row, 2, QTableWidgetItem(p[2]))
                self.table.setItem(row, 3, QTableWidgetItem(f"{p[3]:.2f}"))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"搜索失败: {str(e)}")

    def get_current_product(self):
        row = self.table.currentRow()
        if row >= 0:
            pid = self.table.item(row, 0).text()
            return self.db.get_product_by_id(pid)
        return None

    def add_product(self):
        try:
            pid, ok = QInputDialog.getText(self, "新增", "商品编号:")
            if not ok or not pid:
                return
            if self.db.get_product_by_id(pid):
                QMessageBox.warning(self, "错误", f"编号 {pid} 已存在")
                return
            name_en, ok = QInputDialog.getText(self, "新增", "英文名称:")
            if not ok or not name_en:
                return
            name_cn, ok = QInputDialog.getText(self, "新增", "中文名称:")
            if not ok or not name_cn:
                return
            price, ok = QInputDialog.getDouble(self, "新增", "价格(元/kg):", 0, 0, 1000, 2)
            if not ok:
                return
            if self.db.add_product(pid, name_en, name_cn, price):
                QMessageBox.information(self, "成功", f"{name_cn} 已添加")
                self.load_data()
            else:
                QMessageBox.warning(self, "错误", "添加失败")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"添加失败: {str(e)}")

    def edit_product(self):
        product = self.get_current_product()
        if not product:
            QMessageBox.warning(self, "警告", "请选择要修改的商品")
            return
        items = ["修改价格", "修改英文名称", "修改中文名称"]
        item, ok = QInputDialog.getItem(self, "修改", "选择要修改的项目:", items, 0, False)
        if not ok:
            return
        pid = product[0]
        try:
            if item == "修改价格":
                new_price, ok = QInputDialog.getDouble(self, "修改价格", f"新价格 (当前: {product[3]:.2f}):",
                                                       product[3], 0, 1000, 2)
                if ok:
                    self.db.update_product(pid, price=new_price)
                    QMessageBox.information(self, "成功", "价格已修改")
            elif item == "修改英文名称":
                new_name, ok = QInputDialog.getText(self, "修改英文名称", f"新英文名称 (当前: {product[1]}):",
                                                    text=product[1])
                if ok and new_name:
                    self.db.update_product(pid, name_en=new_name.lower())
                    QMessageBox.information(self, "成功", "英文名称已修改")
            elif item == "修改中文名称":
                new_name, ok = QInputDialog.getText(self, "修改中文名称", f"新中文名称 (当前: {product[2]}):",
                                                    text=product[2])
                if ok and new_name:
                    self.db.update_product(pid, name_cn=new_name)
                    QMessageBox.information(self, "成功", "中文名称已修改")
            self.load_data()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"修改失败: {str(e)}")

    def delete_product(self):
        product = self.get_current_product()
        if not product:
            QMessageBox.warning(self, "警告", "请选择要删除的商品")
            return
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定删除商品 {product[2]} ({product[1]})?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.db.delete_product(product[0])
                self.load_data()
                QMessageBox.information(self, "成功", "商品已删除")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {str(e)}")


# ------------------------------
# 串口配置对话框
# ------------------------------
class SerialConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("串口配置")
        self.setModal(True)
        self.resize(400, 400)
        self.init_ui()
        self.refresh_ports()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 称重串口配置
        weight_group = QGroupBox("称重串口")
        weight_layout = QGridLayout(weight_group)

        weight_layout.addWidget(QLabel("串口号:"), 0, 0)
        self.weight_port_combo = QComboBox()
        self.weight_port_combo.setEditable(True)
        weight_layout.addWidget(self.weight_port_combo, 0, 1)

        self.weight_refresh_btn = QPushButton("刷新")
        self.weight_refresh_btn.clicked.connect(self.refresh_ports)
        weight_layout.addWidget(self.weight_refresh_btn, 0, 2)

        weight_layout.addWidget(QLabel("波特率:"), 1, 0)
        self.weight_baud_combo = QComboBox()
        self.weight_baud_combo.addItems(["9600", "115200", "4800", "19200", "38400", "57600"])
        self.weight_baud_combo.setCurrentText("9600")
        weight_layout.addWidget(self.weight_baud_combo, 1, 1)

        layout.addWidget(weight_group)

        # 语音串口配置
        voice_group = QGroupBox("语音播报串口")
        voice_layout = QGridLayout(voice_group)

        voice_layout.addWidget(QLabel("串口号:"), 0, 0)
        self.voice_port_combo = QComboBox()
        self.voice_port_combo.setEditable(True)
        voice_layout.addWidget(self.voice_port_combo, 0, 1)

        self.voice_refresh_btn = QPushButton("刷新")
        self.voice_refresh_btn.clicked.connect(self.refresh_ports)
        voice_layout.addWidget(self.voice_refresh_btn, 0, 2)

        voice_layout.addWidget(QLabel("波特率:"), 1, 0)
        self.voice_baud_combo = QComboBox()
        self.voice_baud_combo.addItems(["9600", "115200", "4800", "19200", "38400", "57600"])
        self.voice_baud_combo.setCurrentText("9600")
        voice_layout.addWidget(self.voice_baud_combo, 1, 1)

        voice_layout.addWidget(QLabel("语音开关:"), 2, 0)
        self.voice_enabled_check = QCheckBox("启用语音播报")
        self.voice_enabled_check.setChecked(True)
        voice_layout.addWidget(self.voice_enabled_check, 2, 1)

        layout.addWidget(voice_group)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def refresh_ports(self):
        self.weight_port_combo.clear()
        self.voice_port_combo.clear()
        if SERIAL_AVAILABLE:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.weight_port_combo.addItem(f"{port.device} - {port.description}")
                self.voice_port_combo.addItem(f"{port.device} - {port.description}")
            if self.weight_port_combo.count() == 0:
                if IS_WINDOWS:
                    self.weight_port_combo.addItem("COM19")
                    self.voice_port_combo.addItem("COM10")
                else:
                    self.weight_port_combo.addItem("/dev/ttyS10")
                    self.voice_port_combo.addItem("/dev/ttyUSB0")
        else:
            if IS_WINDOWS:
                self.weight_port_combo.addItem("COM19")
                self.voice_port_combo.addItem("COM10")
            else:
                self.weight_port_combo.addItem("/dev/ttyS10")
                self.voice_port_combo.addItem("/dev/ttyUSB0")

    def get_config(self):
        weight_port = self.weight_port_combo.currentText().split(' - ')[
            0] if ' - ' in self.weight_port_combo.currentText() else self.weight_port_combo.currentText()
        weight_baud = int(self.weight_baud_combo.currentText())
        voice_port = self.voice_port_combo.currentText().split(' - ')[
            0] if ' - ' in self.voice_port_combo.currentText() else self.voice_port_combo.currentText()
        voice_baud = int(self.voice_baud_combo.currentText())
        voice_enabled = self.voice_enabled_check.isChecked()
        return weight_port, weight_baud, voice_port, voice_baud, voice_enabled


# ------------------------------
# 自定义搜索控件
# ------------------------------
class SearchWidget(QWidget):
    product_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_products = []
        self.current_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("搜索商品编号或名称...")
        self.line_edit.setMinimumHeight(46)
        self.line_edit.setStyleSheet("""
            QLineEdit {
                border: 2px solid #DADCE0;
                border-radius: 10px;
                padding: 12px 16px;
                font-size: 17px;
                background: white;
            }
            QLineEdit:focus {
                border: 2px solid #1a73e8;
                background: #F8FBFF;
            }
        """)
        self.line_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.line_edit)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(200)
        self.list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #DADCE0;
                border-radius: 10px;
                background: white;
                font-size: 15px;
                padding: 4px;
            }
            QListWidget::item { padding: 8px 10px; border-radius: 6px; }
            QListWidget::item:hover { background: #E8F0FE; }
            QListWidget::item:selected { background: #1a73e8; color: white; }
        """)
        self.list_widget.itemClicked.connect(self.on_item_selected)
        self.list_widget.hide()
        layout.addWidget(self.list_widget)

        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.do_search)

    def set_all_products(self, products):
        self.all_products = products

    def setText(self, text):
        """设置输入框文本"""
        self.line_edit.setText(text)

    def text(self):
        """获取输入框文本"""
        return self.line_edit.text()

    def clear(self):
        """清空输入框"""
        self.line_edit.clear()
        self.list_widget.hide()

    def on_text_changed(self, text):
        self.current_text = text
        self.search_timer.start(200)

    def do_search(self):
        if not self.current_text:
            self.list_widget.hide()
            return
        matched = []
        for p in self.all_products:
            if (self.current_text.lower() in p[0].lower() or
                        self.current_text.lower() in p[1].lower() or
                        self.current_text.lower() in p[2].lower()):
                matched.append(p)
        if matched:
            self.list_widget.clear()
            for p in matched:
                item = QListWidgetItem(f"{p[0]}  {p[2]} ({p[1]})  ¥{p[3]:.2f}/kg")
                item.setData(Qt.UserRole, p)
                self.list_widget.addItem(item)
            self.list_widget.show()
        else:
            self.list_widget.hide()

    def on_item_selected(self, item):
        product = item.data(Qt.UserRole)
        if product:
            self.line_edit.setText(f"{product[0]}  {product[2]}  ¥{product[3]:.2f}/kg")
            self.list_widget.hide()
            self.product_selected.emit(product)

    def mouseDoubleClickEvent(self, event):
        if self.all_products:
            self.list_widget.clear()
            for p in self.all_products:
                item = QListWidgetItem(f"{p[0]}  {p[2]} ({p[1]})  ¥{p[3]:.2f}/kg")
                item.setData(Qt.UserRole, p)
                self.list_widget.addItem(item)
            self.list_widget.show()
        event.accept()

    def focusOutEvent(self, event):
        QTimer.singleShot(200, self.list_widget.hide)
        super().focusOutEvent(event)


# ------------------------------
# 主界面（使用全局配置）
# ------------------------------
class AIScaleSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("灵眸·智衡 —— 全自助视觉称重收费系统")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        self.current_weight = 0.00
        self.last_weight = 0.00
        self.last_spoken_weight = -1  # 上次播报的重量
        self.current_price = 0.00
        self.total_price = 0.00
        self.current_product = None
        self.tare_offset = 0.00
        self.last_captured_image = None
        self.capture_in_progress = False
        self.all_products = []
        self.serial_thread = None
        self.camera_thread = None
        self.tts_thread = None
        self.recognition_results = []
        self.is_zero_weight = True
        self.voice_enabled = True
        self.tts_volume = 5  # TTS 音量 (1-10)

        # 待付款列表相关
        self.pending_items = []  # 待付款商品列表
        self.pending_total = 0.00  # 待付款总金额

        # 支付相关变量
        self.current_remaining = 0.00  # 当前剩余待付金额
        self.payment_records = []  # 扣款记录
        self.qr_scanner_running = False
        self.qr_timer = None
        self.payment_panel = None  # 支付面板
        self._payment_finished = False  # 付款完成标志，阻止 on_payment_panel_closed 立即切屏
        self._last_payment_close_time = 0.0  # 上次关闭付款面板的时间戳，防抖
        self._last_zero_time = 0.0           # 上次归零时间戳，防误触发付款

        self.db = DatabaseManager("fruit_vegetables.db")

        # 重量稳定播报定时器（500ms 防抖）
        self.stable_timer = QTimer()
        self.stable_timer.setSingleShot(True)
        self.stable_timer.timeout.connect(self.on_weight_stable)
        self.last_spoken_total = -1.0   # 上次播报的总价，用于去重
        self._last_nonzero_weight = 0.0  # 最近一次非零净重，用于归零时记录准确重量
        self._stable_timer_weight = 0.0  # 触发 stable_timer 时的净重，避免每 100ms 重启
        self._stable_capture_needed = False  # 重量稳定后触发一次抓拍识别
        self._recap_retries = 0             # 重新抓拍次数（最多2次）
        self._recap_baseline = 0.0          # 上次稳定抓拍时的净重，用于检测 >30g 波动

        # 商品识别播报防抖定时器（500ms）
        self.product_stable_timer = QTimer()
        self.product_stable_timer.setSingleShot(True)
        self.product_stable_timer.timeout.connect(self.on_product_stable)
        self.pending_product_info = None   # 存储待播报的商品 (名称, 价格)

        # 重量归零稳定定时器（500ms 防抖），避免重量抖动导致重复播报"已添加"
        self.zero_stable_timer = QTimer()
        self.zero_stable_timer.setSingleShot(True)
        self.zero_stable_timer.timeout.connect(self.on_zero_stable)
        self.pending_zero_weight = 0.0

        # 播报去重：记录上次播报内容，避免相同内容重复播报
        self.last_spoken_product_info = None    # (name, price) tuple
        self.last_spoken_pending_added = None   # (name, weight_grams, total) tuple
        self.last_spoken_product_time = 0.0     # 上次播报商品的时间戳，用于冷却间隔
        self._user_selected_product = False     # 用户手动选择过商品，禁止自动覆盖

        self.init_ui()
        self.load_products()
        self.load_transaction_history()
        self.init_camera()
        self.init_serial()
        self.init_tts()

        # F407 启动同步标志：首次串口连通时发送清零命令
        self._f407_synced = False

    def init_serial(self):
        """初始化称重串口（使用全局配置，同时承载语音控制命令 AA xx FE 55）"""
        if SERIAL_AVAILABLE:
            self.serial_thread = SerialThread(port=WEIGHT_PORT, baudrate=9600)
            self.serial_thread.weight_updated.connect(self.on_weight_received)
            self.serial_thread.connection_status.connect(self.on_serial_status)
            self.serial_thread.error_occurred.connect(self.on_serial_error)
            # 语音控制信号（复用称重串口上的 AA xx FE 55 命令）
            self.serial_thread.tts_disable_signal.connect(self.on_tts_control_disable)
            self.serial_thread.tts_enable_signal.connect(self.on_tts_control_enable)
            self.serial_thread.pending_remove_last_signal.connect(self.on_tts_control_remove_last)
            self.serial_thread.tare_signal.connect(self.on_tts_control_tare)
            self.serial_thread.open_payment_signal.connect(self.on_tts_control_open_payment)
            self.serial_thread.start_qr_scan_signal.connect(self.on_tts_control_start_qr_scan)
            self.serial_thread.calc_total_signal.connect(self.on_tts_control_calc_total)
            self.serial_thread.repeat_last_signal.connect(self.on_tts_control_repeat_last)
            self.serial_thread.lcd_button_signal.connect(self.on_lcd_button)
            self.serial_thread.start()
            self.serial_status_label.setText("称重: 连接中...")
            self.serial_status_label.setStyleSheet("font-size: 16px; color: #FF6D00; font-weight: bold;")
        else:
            self.serial_status_label.setText("称重: 未安装")
            self.serial_status_label.setStyleSheet("font-size: 16px; color: #EA4335; font-weight: bold;")

    def init_tts(self):
        """初始化语音播报（使用全局配置）"""
        if SERIAL_AVAILABLE and self.voice_enabled:
            self.tts_thread = TTSThread(port=VOICE_PORT, baudrate=9600, enabled=True)
            self.tts_thread.start()
            self.voice_status_label.setText("语音: 连接中...")
            self.voice_status_label.setStyleSheet("font-size: 16px; color: #FF6D00; font-weight: bold;")
            # 延迟检查串口是否真正连接
            QTimer.singleShot(3000, self._check_tts_connected)
        else:
            self.voice_status_label.setText("语音: 未启用")
            self.voice_status_label.setStyleSheet("font-size: 16px; color: #80868B; font-weight: 600;")

    def _check_tts_connected(self):
        """检查 TTS 串口是否真正连接"""
        if self.tts_thread:
            if self.tts_thread.serial_port and self.tts_thread.serial_port.is_open:
                self.voice_status_label.setText(f"语音: 已启用 ({self.tts_thread.baudrate}bps)")
                self.voice_status_label.setStyleSheet("font-size: 16px; color: #0F9D58; font-weight: bold;")
            else:
                self.voice_status_label.setText(f"语音: 未连接 ({VOICE_PORT})")
                self.voice_status_label.setStyleSheet("font-size: 16px; color: #EA4335; font-weight: bold;")
                print(f"[语音] 严重: TTS串口 {VOICE_PORT} 所有波特率均连接失败")

    def on_tts_control_disable(self):
        """关闭TTS模块"""
        # 先播报再关闭，确保语音能发出去
        if self.tts_thread:
            self.tts_thread.speak("语音已关闭")
            QTimer.singleShot(800, self._do_disable_tts)
        else:
            self._do_disable_tts()

    def _do_disable_tts(self):
        self.voice_enabled = False
        if self.tts_thread:
            self.tts_thread.set_enabled(False)
        self.voice_status_label.setText("语音: 已关闭")
        self.voice_status_label.setStyleSheet("font-size: 16px; color: #EA4335; font-weight: bold;")
        print("[TTS控制] TTS模块已关闭")

    def on_tts_control_enable(self):
        """恢复TTS模块"""
        self.voice_enabled = True
        if self.tts_thread:
            self.tts_thread.set_enabled(True)
        else:
            self.init_tts()
        self.voice_status_label.setText("语音: 已启用")
        self.voice_status_label.setStyleSheet("font-size: 16px; color: #0F9D58; font-weight: bold;")
        if self.tts_thread:
            self.tts_thread.speak("语音已恢复")
        print("[TTS控制] TTS模块已恢复")

    def on_tts_control_remove_last(self):
        """删除最近一次待付款商品"""
        if not self.pending_items:
            if self.tts_thread:
                self.tts_thread.speak("待付款列表为空")
            return
        removed = self.pending_items.pop()
        self.update_pending_list()
        self.update_pending_total()
        print(f"[TTS控制] 已删除最近商品: {removed['product_name']}")
        self.statusBar().showMessage(f"已删除: {removed['product_name']}", 2000)
        if self.tts_thread:
            self.tts_thread.speak(f"已删除{removed['product_name']}")
        # 重置播报去重状态
        self.last_spoken_pending_added = None

    def on_tts_control_tare(self):
        """去皮"""
        self.tare_offset = self.current_weight
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_tare(self.current_weight)
        self.update_total()
        display_weight = max(0, self.current_weight - self.tare_offset)
        self.statusBar().showMessage(f"去皮已设置 (偏移: {self.tare_offset:.3f} kg)", 2000)
        print(f"[TTS控制] 去皮完成，偏移={self.tare_offset:.3f}kg")
        if self.tts_thread:
            self.tts_thread.speak(f"去皮完成，当前重量{display_weight:.3f}千克")

    def on_tts_control_open_payment(self):
        """进入扫码付款界面"""
        if self.pending_total <= 0:
            if self.tts_thread:
                self.tts_thread.speak("没有待付款商品")
            return
        if self.tts_thread:
            self.tts_thread.speak(f"已进入扫码付款，待付{self.pending_total:.2f}元")
        self.open_payment_dialog(from_user=True)  # 语音是明确指令，绕过 3 秒自动打开防抖

    def on_tts_control_start_qr_scan(self):
        """开始扫码"""
        if not hasattr(self, 'payment_panel') or self.payment_panel is None:
            # 付款面板未打开，先打开
            if self.pending_total <= 0:
                if self.tts_thread:
                    self.tts_thread.speak("没有待付款商品")
                return
            self.open_payment_dialog(from_user=True)  # 同上，语音触发绕过自动防抖
            # 延迟一下等面板创建完成再扫码
            QTimer.singleShot(500, self._do_start_qr_scan)
        else:
            self._do_start_qr_scan()

    def _do_start_qr_scan(self):
        if not hasattr(self, 'payment_panel') or self.payment_panel is None:
            return
        if self.tts_thread:
            self.tts_thread.speak("请将付款码对准摄像头")
        self.start_qr_scanner(self.payment_panel)

    def on_tts_control_calc_total(self):
        """播报待付款总额"""
        total = self.pending_total
        print(f"[TTS控制] 当前待付款总额: ¥{total:.2f}")
        if self.tts_thread:
            if total <= 0:
                self.tts_thread.speak("待付款列表为空")
            else:
                self.tts_thread.speak(f"待付款共{total:.2f}元")

    def on_tts_control_repeat_last(self):
        """重复上一条播报"""
        if self.tts_thread:
            self.tts_thread.repeat_last()
            print("[TTS控制] 重复上一条播报")

    def on_lcd_button(self, cmd):
        """处理 LCD 触摸按钮事件（0x3A=扫码付款, 0x3B=从付款页返回）"""
        if cmd == 0x3A:
            print("[LCD按钮] 扫码付款")
            self.open_payment_dialog(from_user=True)
        elif cmd == 0x3B:
            print("[LCD按钮] 从付款页返回")
            if hasattr(self, 'payment_panel') and self.payment_panel:
                try:
                    self.payment_panel.close()
                except:
                    pass

    def init_camera(self):
        """初始化摄像头（使用全局配置）"""
        if CAMERA_AVAILABLE:
            self.camera_thread = CameraThread(camera_id=CAMERA_ID)
            self.camera_thread.image_update.connect(self.update_camera_display)
            self.camera_thread.capture_finished.connect(self.on_capture_finished)
            self.camera_thread.recog_settled.connect(self._on_recog_settled)

            if ONNX_AVAILABLE:
                if self.camera_thread.init_inference():
                    print("[AI] ONNX识别已加载")

            self.camera_thread.start()
            self.statusBar().showMessage(f"摄像头 {CAMERA_ID} 已启动")
        else:
            self.statusBar().showMessage("OpenCV未安装")


    def speak_total_price(self, total_price):
        """播报总费用（仅金额）"""
        if not self.voice_enabled or not self.tts_thread:
            return
        try:
            text = f"{total_price:.2f}元"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报总价失败: {e}")

    def speak_product_price(self, product_name, price):
        """播报商品单价（去重：相同商品+价格不重复播报）"""
        if not self.voice_enabled or not self.tts_thread:
            return
        if self.last_spoken_product_info == (product_name, price):
            return
        self.last_spoken_product_info = (product_name, price)
        try:
            text = f"{product_name}单价为{price:.2f}元每千克"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报单价失败: {e}")


    def speak_pending_added(self, product_name, weight_grams, total_price):
        """播报添加商品到待付款列表（去重：相同内容不重复播报）"""
        if not self.voice_enabled or not self.tts_thread:
            return
        if self.last_spoken_pending_added == (product_name, weight_grams, total_price):
            return
        self.last_spoken_pending_added = (product_name, weight_grams, total_price)
        try:
            text = f"已添加{product_name}，重量{weight_grams}克，金额{total_price:.2f}元"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报添加失败: {e}")

    def speak_payment_amount(self, amount, remaining):
        """播报扣款金额"""
        if not self.voice_enabled or not self.tts_thread:
            return
        try:
            text = f"收款{amount:.2f}元，剩余{remaining:.2f}元"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报扣款失败: {e}")

    def speak_amount_modified(self, new_amount):
        """播报金额修改"""
        if not self.voice_enabled or not self.tts_thread:
            return
        try:
            text = f"金额已修改为{new_amount:.2f}元"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报金额修改失败: {e}")

    def speak_payment_exceed_warning(self, remaining):
        """播报金额超限警告"""
        if not self.voice_enabled or not self.tts_thread:
            return
        try:
            text = f"付款金额超过待付金额，待付金额为{remaining:.2f}元"
            self.tts_thread.speak(text)
        except Exception as e:
            print(f"[语音] 播报警告失败: {e}")

    def on_weight_received(self, weight):
        """接收重量数据 - 使用稳定定时器播报总价（500ms防抖）"""
        if weight < 0 or weight > 50:
            return

        net_weight = max(0, weight - self.tare_offset)
        self.current_weight = weight
        self.update_total()

        # 净重 < 5g 视为无物
        if net_weight <= 0.005:
            if not self.is_zero_weight:
                print("[称重] 净重归零，等待稳定后记录商品到待付款列表")
                # 优先用 _last_nonzero_weight，其次用 historical last_weight
                display_weight = self._last_nonzero_weight
                if display_weight <= 0.005:
                    display_weight = max(0, self.last_weight - self.tare_offset)
                if display_weight >= 0.005:
                    self.pending_zero_weight = display_weight
                    self.zero_stable_timer.start(500)
                else:
                    print("[称重] 重量太轻，不记录")
                self._last_zero_time = time.time()  # 仅在“刚归零”时记录，避免空秤时每帧刷新

            self.recognition_list.clear()
            self.recognition_results = []
            self.recognition_count_label.setText("共 0 个结果")
            self.is_zero_weight = True
            self.pending_product_info = None
            self.last_spoken_total = -1.0
            self._user_selected_product = False
            self.capture_in_progress = False
            self._stable_timer_weight = 0.0
            self._last_nonzero_weight = 0.0
            self._stable_capture_needed = False
            self._recap_retries = 0
            self._recap_baseline = 0.0
        else:
            # 有重物
            self.zero_stable_timer.stop()
            self.pending_zero_weight = 0.0

            was_zero = self.is_zero_weight
            self.is_zero_weight = False

            # 从零唤醒或重量变化>20g时标记需要抓拍，等重量稳定后再抓
            if was_zero or abs(weight - self.last_weight) > 0.02:
                if was_zero:
                    print(f"[称重] 从归零唤醒，等待稳定后抓拍识别")
                else:
                    print(f"[称重] 重量变化，等待稳定后抓拍识别")
                self._stable_capture_needed = True

            # 仅在净重变化 >3g 时重启防抖定时器，避免 100ms 周期无限重启
            if abs(net_weight - self._stable_timer_weight) > 0.003:
                self._stable_timer_weight = net_weight
                self.stable_timer.start(500)

            # 已抓拍过的情况下，净重波动 >30g 触发重新抓拍（用户调整物品角度，最多2次）
            if (self._recap_baseline > 0.005 and
                    self._recap_retries < 2 and
                    abs(net_weight - self._recap_baseline) > 0.03 and
                    not self._stable_capture_needed):
                print(f"[称重] 净重波动 {abs(net_weight - self._recap_baseline)*1000:.0f}g，触发重新抓拍 (第{self._recap_retries+1}次)")
                self._stable_capture_needed = True

        self.last_weight = weight

    def on_weight_stable(self):
        """重量稳定后触发抓拍、记录重量（不播报）"""
        net_weight = max(0, self.current_weight - self.tare_offset)
        if net_weight <= 0.005:
            return
        # 稳定后的重量才是真实重量，供归零路径使用
        self._last_nonzero_weight = net_weight

        # 重量稳定后抓拍（延迟到此时，手已离开，画面干净）
        if self._stable_capture_needed and not self.capture_in_progress:
            self._stable_capture_needed = False
            self._recap_baseline = net_weight
            if self._recap_retries > 0:
                print(f"[称重] 重新抓拍完成 (第{self._recap_retries}次，净重: {net_weight*1000:.0f}g)")
            else:
                print(f"[称重] 初始抓拍完成 (净重: {net_weight*1000:.0f}g)")
            self._recap_retries += 1
            self.capture_async(recognize=True)

    def on_zero_stable(self):
        """重量归零稳定后（500ms无变化），将商品添加到待付款列表并播报"""
        if self.pending_zero_weight >= 0.005:
            self.add_to_pending_list_with_weight(self.pending_zero_weight)
        self.pending_zero_weight = 0.0

    # ---------- 商品识别防抖 ----------
    def on_product_stable(self):
        """识别稳定后记录商品信息（不播报）"""
        if not self.pending_product_info:
            return
        if max(0, self.current_weight - self.tare_offset) <= 0.005:
            return
        self.pending_product_info = None


    def add_to_pending_list_with_weight(self, weight):
        """将当前商品添加到待付款列表（使用指定的重量）"""
        try:
            # 优先使用 current_product，如果没有则使用界面显示的商品信息
            product_id = None
            product_name = None
            price = None

            if self.current_product is not None:
                # 使用选中的商品
                product_id = self.current_product[0]
                product_name = self.current_product[2]
                price = self.current_price
                print(f"[待付款] 使用选中的商品: {product_name}")
            elif self.label_id.text() != "-":
                # 使用界面显示的商品信息
                product_id = self.label_id.text()
                product_name = self.label_name_cn.text()
                price_text = self.label_price.text().replace("元/kg", "").strip()
                try:
                    price = float(price_text)
                    print(f"[待付款] 使用界面显示的商品: {product_name}, 价格: {price}")
                except:
                    price = 0
            else:
                print("[待付款] 没有商品信息，无法添加")
                return

            if weight <= 0.005:
                print("[待付款] 重量太轻(<5g)，不添加")
                return

            if price <= 0:
                print("[待付款] 价格无效，不添加")
                QMessageBox.warning(self, "提示", "商品价格无效，请检查商品信息")
                return

            # 获取英文名（用于 LCD 显示）
            name_en = ''
            if self.current_product is not None and len(self.current_product) > 1:
                name_en = self.current_product[1]  # DB: [id, name_en, name_cn, price]
            elif product_id:
                db_product = self.db.get_product_by_id(product_id)
                if db_product and len(db_product) > 1:
                    name_en = db_product[1]

            # 创建待付款项
            item = {
                'product_id': product_id,
                'product_name': product_name,
                'name_en': name_en,
                'price': price,
                'weight': weight,
                'total': weight * price,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }

            self.pending_items.append(item)
            self.update_pending_list()
            self.update_pending_total()

            weight_grams = int(weight * 1000)
            print(
                f"[待付款] 已添加: {item['product_name']} {item['weight']:.3f}kg × ¥{item['price']:.2f} = ¥{item['total']:.2f}")
            self.statusBar().showMessage(f"已添加: {item['product_name']} {weight_grams}克 ¥{item['total']:.2f}", 3000)

            # 语音播报添加的商品（零重量已稳定，直接播报）
            self.speak_pending_added(product_name, weight_grams, item['total'])

            # 清空当前商品选择，准备下一个
            self.current_product = None
            self.current_price = 0
            self.label_id.setText("-")
            self.label_name_en.setText("-")
            self.label_name_cn.setText("-")
            self.label_price.setText("-")
            self.label_current_price.setText("0.00 元/kg")
            self.search_widget.clear()  # 使用 clear 方法清空
            self.update_total()
            # 重置播报状态，确保下一个商品（即使同品种）能正常播报
            self.last_spoken_product_info = None
            self.last_spoken_product_time = 0.0
            self.last_spoken_total = -1.0
            self._last_nonzero_weight = 0.0

        except Exception as e:
            print(f"[待付款] 添加失败: {e}")

    def update_pending_list(self):
        """更新待付款列表显示"""
        self.pending_list.clear()

        if not self.pending_items:
            self.pending_list.addItem("暂无待付款商品")
            return

        for item in self.pending_items:
            display_text = f"{item['product_name']}  {item['weight']:.3f}kg × ¥{item['price']:.2f} = ¥{item['total']:.2f}"
            list_item = QListWidgetItem(display_text)
            list_item.setData(Qt.UserRole, item)
            self.pending_list.addItem(list_item)

    def update_pending_total(self):
        """更新待付款总价显示"""
        self.pending_total = sum(item['total'] for item in self.pending_items)
        self.pending_total_label.setText(f"待付款总额: ¥{self.pending_total:.2f}")

        # 付款面板打开时不发 LCD 命令，避免屏幕乱切
        panel_open = hasattr(self, 'payment_panel') and self.payment_panel is not None
        if hasattr(self, 'serial_thread') and self.serial_thread and not panel_open:
            if self.pending_items:
                self.serial_thread.send_pending_list(self.pending_items)
            else:
                self.serial_thread.send_clear_list()
                self.serial_thread.send_lcd_switch_screen(1)

    def clear_pending_list(self):
        """清空待付款列表"""
        self.pending_items = []
        self.pending_total = 0.00
        self.current_remaining = 0.00  # 重置剩余金额
        self.payment_records = []  # 清空扣款记录
        self.update_pending_list()
        self.update_pending_total()
        # 重置播报去重状态
        self.last_spoken_pending_added = None
        self.last_spoken_product_info = None
        print("[待付款] 列表已清空")

    def lcd_page_up(self):
        """LCD 下一页"""
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_lcd_page_up()
            self.statusBar().showMessage("LCD 下一页", 1500)

    def lcd_page_down(self):
        """LCD 上一页"""
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_lcd_page_down()
            self.statusBar().showMessage("LCD 上一页", 1500)

    def remove_selected_pending_item(self):
        """删除选中的待付款项"""
        current_row = self.pending_list.currentRow()
        if current_row >= 0 and current_row < len(self.pending_items):
            removed = self.pending_items.pop(current_row)
            self.update_pending_list()
            self.update_pending_total()
            print(f"[待付款] 已删除: {removed['product_name']}")
            self.statusBar().showMessage(f"已删除: {removed['product_name']}", 2000)
        else:
            QMessageBox.warning(self, "提示", "请先选择要删除的商品")

    def start_qr_scanner(self, dialog):
        """启动二维码扫描（不遮挡视频）"""
        if not CAMERA_AVAILABLE:
            QMessageBox.warning(dialog, "提示", "摄像头不可用，请手动输入金额")
            return

        if not PYZBAR_AVAILABLE:
            QMessageBox.warning(dialog, "提示", "二维码库未安装，请安装pyzbar\npip install pyzbar")
            return

        # 重复调用时先停掉旧定时器，避免多个定时器同时解码
        if getattr(self, 'qr_timer', None):
            try:
                self.qr_timer.stop()
            except Exception:
                pass
            self.qr_timer = None

        self.qr_scanner_running = True

        # 使用定时器从摄像头读取帧
        self.qr_timer = QTimer()
        self.qr_timer.timeout.connect(self.scan_qr_frame)
        self.qr_timer.start(100)

        # 状态栏提示，不阻塞界面
        self.statusBar().showMessage("扫码中，请将付款码对准摄像头...", 3000)

    def scan_qr_frame(self):
        """扫描二维码帧"""
        if not self.qr_scanner_running or not self.camera_thread:
            return

        frame = self.camera_thread.get_current_frame()
        if frame is None:
            return

        try:
            # 使用pyzbar解码二维码
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)

            for barcode in barcodes:
                qr_data = barcode.data.decode('utf-8')
                print(f"[扫码] 识别到二维码: {qr_data}")

                # 停止扫描
                if self.qr_timer:
                    self.qr_timer.stop()
                self.qr_scanner_running = False

                # 状态栏提示
                self.statusBar().showMessage(f"识别到二维码: {qr_data[:20]}...", 2000)

                # 处理扫码结果
                self.process_qr_result(qr_data)
                break
        except Exception as e:
            print(f"[扫码] 错误: {e}")

    def process_qr_result(self, qr_data):
        """处理扫码结果（带金额上限检查）"""
        # 获取当前支付面板
        dialog = self.payment_panel if hasattr(self, 'payment_panel') and self.payment_panel else None
        if dialog is None:
            return

        try:
            # 尝试解析为金额
            payment_amount = float(qr_data)
        except ValueError:
            # 如果不是纯数字，可能是订单号，显示输入对话框（最大值限制为待付金额）
            amount, ok = QInputDialog.getDouble(dialog, "输入金额",
                                                f"扫码内容: {qr_data}\n请输入付款金额(元):\n(最大金额: ¥{self.current_remaining:.2f})",
                                                0, 0, self.current_remaining, 2)
            if not ok:
                return
            payment_amount = amount

        # 金额有效性检查
        if payment_amount <= 0:
            QMessageBox.warning(dialog, "错误", "付款金额必须大于0")
            return

        # ===== 微信/支付宝付款码识别（摄像头扫码路径）=====
        # 16~18 位纯数字串视为付款码，直接全额扣款并提示支付成功。
        if qr_data.isdigit() and len(qr_data) >= 12:
            payment_amount = self.current_remaining
            self.current_remaining = 0.0
            self.payment_records.append({
                'type': 'payment',
                'code': f"微信扫码:{qr_data}",
                'amount': payment_amount,
                'remaining': 0.0,
                'time': datetime.now().strftime("%H:%M:%S")
            })
            if hasattr(self, 'remain_amount_label'):
                self.remain_amount_label.setText("¥0.00")
            if hasattr(self, 'scan_code_edit'):
                self.scan_code_edit.setText(qr_data)
            self.update_payment_record_list()
            if hasattr(self, 'speak_payment_amount'):
                self.speak_payment_amount(payment_amount, 0.0)
            QMessageBox.information(dialog, "支付成功",
                                    f"微信扫码支付成功！\n已收款 ¥{payment_amount:.2f}")
            dialog.accept()
            self.complete_payment()
            return

        # 金额上限检查：不能超过待付金额（允许0.01容差）
        if payment_amount > self.current_remaining + 0.01:
            self.speak_payment_exceed_warning(self.current_remaining)
            QMessageBox.warning(dialog, "错误",
                                f"付款金额超过待付金额！\n\n待付金额: ¥{self.current_remaining:.2f}\n付款金额: ¥{payment_amount:.2f}\n\n请重新扫描或输入正确的金额")
            return

        # 记录扣款
        self.current_remaining -= payment_amount
        self.payment_records.append({
            'type': 'payment',
            'code': qr_data,
            'amount': payment_amount,
            'remaining': self.current_remaining,
            'time': datetime.now().strftime("%H:%M:%S")
        })

        # 更新显示
        if hasattr(self, 'remain_amount_label'):
            self.remain_amount_label.setText(f"¥{self.current_remaining:.2f}")
        if hasattr(self, 'scan_code_edit'):
            self.scan_code_edit.setText(qr_data)
        self.update_payment_record_list()

        # 语音播报
        self.speak_payment_amount(payment_amount, self.current_remaining)

        QMessageBox.information(dialog, "扣款成功",
                                f"已扣款 ¥{payment_amount:.2f}\n剩余 ¥{self.current_remaining:.2f}")

        # 检查是否付清
        if self.current_remaining <= 0.01:
            QMessageBox.information(dialog, "付款完成", "款项已全部付清！")
            dialog.accept()
            self.complete_payment()

    def open_payment_dialog(self, from_user=False):
        """打开扫码支付面板（嵌入式，不遮挡摄像头）
        from_user=True: 手动点击按钮，不设防抖限制
        from_user=False: 自动触发（语音/LCD/串口），需要防抖"""
        if self.pending_total <= 0:
            QMessageBox.warning(self, "提示", "没有待付款商品")
            return

        # 仅自动触发时应用防抖（防止串口噪声/误触）
        if not from_user:
            if time.time() - getattr(self, '_last_payment_close_time', 0) < 3.0:
                print("[付款] 距上次关闭不足3秒，忽略自动打开请求")
                return
            if time.time() - getattr(self, '_last_zero_time', 0) < 3.0:
                print("[付款] 刚归零，忽略自动打开请求")
                return

        # 先停掉可能还在运行的扫码定时器，避免旧面板的扫描结果串到新面板
        if getattr(self, 'qr_timer', None):
            try:
                self.qr_timer.stop()
            except Exception:
                pass
        self.qr_scanner_running = False

        # 初始化支付数据
        self.current_remaining = self.pending_total

        # 同步到 F407 LCD
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_lcd_payment_amount(self.pending_total)
            self.serial_thread.send_lcd_switch_screen(3)
        self.payment_records = []

        # 检查是否已存在支付面板
        if hasattr(self, 'payment_panel') and self.payment_panel is not None:
            # 断开旧面板的信号，避免 on_payment_panel_closed 发送冲突的切屏命令
            try:
                self.payment_panel.finished.disconnect(self.on_payment_panel_closed)
            except:
                pass
            try:
                self.payment_panel.close()
                self.payment_panel.deleteLater()
            except:
                pass

        # 创建支付面板（作为独立窗口，但不阻塞主窗口）
        self.payment_panel = QDialog(self)
        self.payment_panel.setWindowTitle("扫码支付")
        self.payment_panel.setModal(False)  # 非模态，不阻塞主窗口
        self.payment_panel.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)
        self.payment_panel.resize(550, 650)  # 加宽以容纳修改金额控件

        # 设置窗口位置（主窗口右侧，不遮挡摄像头）
        main_geometry = self.geometry()
        panel_x = main_geometry.x() + main_geometry.width() - 570
        panel_y = main_geometry.y() + 50
        self.payment_panel.move(panel_x, panel_y)

        self.payment_panel.setStyleSheet("""
            QDialog {
                background: #F0F4F8;
                border: 2px solid #1976D2;
                border-radius: 14px;
            }
            QLabel {
                font-size: 15px;
            }
            QLineEdit {
                font-size: 16px;
                padding: 10px;
                border: 1px solid #B0BEC5;
                border-radius: 8px;
                background: white;
            }
            QLineEdit:focus { border: 2px solid #1E88E5; }
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                color: white;
            }
            QPushButton:hover { background-color: #42A5F5; }
            QPushButton:pressed { background-color: #1976D2; }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #1565C0;
                border: 1px solid #D0D9E5;
                border-radius: 12px;
                margin-top: 14px;
                padding-top: 18px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
            }
        """)

        layout = QVBoxLayout(self.payment_panel)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("💳  扫码支付")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #0D47A1;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 待付款总额显示
        total_frame = QFrame()
        total_frame.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFF3E0, stop:1 #FFE0B2); border-radius: 10px; padding: 12px;")
        total_layout = QHBoxLayout(total_frame)
        total_layout.addWidget(QLabel("待付款总额:"))
        self.payment_total_label = QLabel(f"¥{self.pending_total:.2f}")
        self.payment_total_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #F44336;")
        total_layout.addWidget(self.payment_total_label)
        total_layout.addStretch()
        layout.addWidget(total_frame)

        # 修改金额区域
        modify_group = QGroupBox("修改金额")
        modify_layout = QHBoxLayout(modify_group)

        self.modify_amount_edit = QLineEdit()
        self.modify_amount_edit.setPlaceholderText("输入新金额，留空不修改")
        self.modify_amount_edit.setMinimumHeight(35)
        modify_layout.addWidget(self.modify_amount_edit)

        btn_apply_modify = QPushButton("应用修改")
        btn_apply_modify.setStyleSheet("background: #FB8C00; font-size: 16px; min-width: 100px;")
        btn_apply_modify.clicked.connect(lambda: self.apply_amount_modify(self.payment_panel))
        modify_layout.addWidget(btn_apply_modify)

        layout.addWidget(modify_group)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 扫码输入区域
        scan_group = QGroupBox("扫码付款")
        scan_layout = QVBoxLayout(scan_group)

        # 扫码按钮和输入框
        scan_btn_layout = QHBoxLayout()
        self.scan_code_edit = QLineEdit()
        self.scan_code_edit.setPlaceholderText("扫描二维码或手动输入金额")
        self.scan_code_edit.setMinimumHeight(40)
        self.scan_code_edit.returnPressed.connect(lambda: self.process_manual_payment(self.payment_panel))
        scan_btn_layout.addWidget(self.scan_code_edit)

        btn_scan = QPushButton("📷 扫码")
        btn_scan.setStyleSheet("background: #1E88E5; font-size: 15px; min-width: 80px;")
        btn_scan.clicked.connect(lambda: self.start_qr_scanner(self.payment_panel))
        scan_btn_layout.addWidget(btn_scan)

        scan_layout.addLayout(scan_btn_layout)

        btn_confirm = QPushButton("✅ 确认扣款")
        btn_confirm.setMinimumHeight(45)
        btn_confirm.setStyleSheet("background: #2E7D32; font-size: 16px;")
        btn_confirm.clicked.connect(lambda: self.process_manual_payment(self.payment_panel))
        scan_layout.addWidget(btn_confirm)

        layout.addWidget(scan_group)

        # 扣款记录区域
        record_group = QGroupBox("扣款记录")
        record_layout = QVBoxLayout(record_group)
        self.payment_record_list = QListWidget()
        self.payment_record_list.setMaximumHeight(150)
        self.payment_record_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
        """)
        record_layout.addWidget(self.payment_record_list)
        layout.addWidget(record_group)

        # 剩余待付金额
        remain_frame = QFrame()
        remain_frame.setStyleSheet("background: #E8F5E9; border-radius: 8px; padding: 10px;")
        remain_layout = QHBoxLayout(remain_frame)
        remain_layout.addWidget(QLabel("剩余待付:"))
        self.remain_amount_label = QLabel(f"¥{self.pending_total:.2f}")
        self.remain_amount_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2196F3;")
        remain_layout.addWidget(self.remain_amount_label)
        remain_layout.addStretch()
        layout.addWidget(remain_frame)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_close = QPushButton("关闭")
        btn_close.setStyleSheet("background: #78909C; font-size: 14px; min-width: 80px;")
        btn_close.clicked.connect(self.payment_panel.close)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        # 更新扣款记录显示
        self.update_payment_record_list()

        # 窗口关闭时清理
        self.payment_panel.finished.connect(self.on_payment_panel_closed)

        self.payment_panel.show()

        # 打开后自动开始摄像头扫码，并把焦点放到扫码输入框（兼容扫码枪输入）
        self.scan_code_edit.setFocus()
        if CAMERA_AVAILABLE and PYZBAR_AVAILABLE and getattr(self, 'camera_thread', None):
            QTimer.singleShot(300, self._auto_start_qr)

    def _auto_start_qr(self):
        """付款面板打开后自动启动扫码，避免每次都要手动点“扫码”"""
        if not hasattr(self, 'payment_panel') or self.payment_panel is None:
            return
        if self.qr_scanner_running:
            return
        self.start_qr_scanner(self.payment_panel)

    def on_payment_panel_closed(self):
        """支付面板关闭时的回调"""
        # 停止二维码扫描
        if hasattr(self, 'qr_timer') and self.qr_timer:
            self.qr_timer.stop()
        self.qr_scanner_running = False
        self._last_payment_close_time = time.time()
        # 清理引用
        if hasattr(self, 'payment_panel'):
            self.payment_panel = None
        # 如果是付款完成触发的关闭，不立即切屏（由 _finish_payment 延迟处理）
        if getattr(self, '_payment_finished', False):
            return
        # 手动关闭支付面板时 LCD 切回 Screen1
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_lcd_switch_screen(1)

    def apply_amount_modify(self, dialog):
        """应用金额修改"""
        try:
            modify_text = self.modify_amount_edit.text().strip()
            if modify_text:
                new_total = float(modify_text)

                # 金额必须大于0
                if new_total <= 0:
                    QMessageBox.warning(dialog, "错误", "金额必须大于0")
                    return

                # 记录修改
                old_total = self.current_remaining
                self.current_remaining = new_total

                # 同时更新待付款总额
                self.pending_total = new_total

                self.payment_records.append({
                    'type': 'modify',
                    'amount': new_total - old_total,
                    'new_total': new_total,
                    'old_total': old_total,
                    'time': datetime.now().strftime("%H:%M:%S")
                })

                # 更新显示
                if hasattr(self, 'payment_total_label'):
                    self.payment_total_label.setText(f"¥{new_total:.2f}")
                if hasattr(self, 'remain_amount_label'):
                    self.remain_amount_label.setText(f"¥{new_total:.2f}")

                # 同时更新主界面的待付款总额显示
                self.pending_total_label.setText(f"待付款总额: ¥{new_total:.2f}")

                self.modify_amount_edit.clear()
                self.update_payment_record_list()

                # 同步修改后的金额到 F407 LCD Screen3
                if hasattr(self, 'serial_thread') and self.serial_thread:
                    self.serial_thread.send_lcd_payment_amount(new_total)

                QMessageBox.information(dialog, "成功", f"金额已修改为 ¥{new_total:.2f}")

                # 语音播报金额修改
                self.speak_amount_modified(new_total)
        except ValueError:
            QMessageBox.warning(dialog, "错误", "请输入有效的金额")

    def process_manual_payment(self, dialog):
        """处理手动输入扣款（带金额上限检查）"""
        if not hasattr(self, 'scan_code_edit'):
            return

        scan_code = self.scan_code_edit.text().strip()
        if not scan_code:
            QMessageBox.warning(dialog, "提示", "请输入金额或扫描二维码")
            return

        try:
            payment_amount = float(scan_code)
        except ValueError:
            QMessageBox.warning(dialog, "错误", "请输入有效的金额")
            return

        if payment_amount <= 0:
            QMessageBox.warning(dialog, "错误", "付款金额必须大于0")
            return

        # ===== 微信/支付宝付款码识别 =====
        # 真实付款码是 16~18 位纯数字（如 130861823629471120），不是金额。
        # 扫码枪扫到后会被 float() 解析成天文数字，这里识别出来直接当作"扫码支付成功"，
        # 按当前待付金额全额扣款，不再弹"金额超过待付"错误框。
        if scan_code.isdigit() and len(scan_code) >= 12:
            payment_amount = self.current_remaining
            self.current_remaining = 0.0
            self.payment_records.append({
                'type': 'payment',
                'code': f"微信扫码:{scan_code}",
                'amount': payment_amount,
                'remaining': 0.0,
                'time': datetime.now().strftime("%H:%M:%S")
            })
            if hasattr(self, 'remain_amount_label'):
                self.remain_amount_label.setText("¥0.00")
            self.scan_code_edit.clear()
            self.update_payment_record_list()
            if hasattr(self, 'speak_payment_amount'):
                self.speak_payment_amount(payment_amount, 0.0)
            QMessageBox.information(dialog, "支付成功",
                                    f"微信扫码支付成功！\n已收款 ¥{payment_amount:.2f}")
            dialog.accept()
            self.complete_payment()
            return

        # 金额上限检查：不能超过待付金额（允许0.01容差）
        if payment_amount > self.current_remaining + 0.01:
            self.speak_payment_exceed_warning(self.current_remaining)
            QMessageBox.warning(dialog, "错误",
                                f"付款金额超过待付金额！\n\n待付金额: ¥{self.current_remaining:.2f}\n付款金额: ¥{payment_amount:.2f}\n\n请重新输入正确的金额")
            self.scan_code_edit.clear()
            self.scan_code_edit.setFocus()
            return

        # 记录扣款
        self.current_remaining -= payment_amount
        self.payment_records.append({
            'type': 'payment',
            'code': f"手动:{payment_amount}",
            'amount': payment_amount,
            'remaining': self.current_remaining,
            'time': datetime.now().strftime("%H:%M:%S")
        })

        # 更新显示
        self.remain_amount_label.setText(f"¥{self.current_remaining:.2f}")
        self.scan_code_edit.clear()
        self.update_payment_record_list()

        # 语音播报
        self.speak_payment_amount(payment_amount, self.current_remaining)

        QMessageBox.information(dialog, "扣款成功",
                                f"已扣款 ¥{payment_amount:.2f}\n剩余 ¥{self.current_remaining:.2f}")

        # 检查是否付清
        if self.current_remaining <= 0.01:
            QMessageBox.information(dialog, "付款完成", "款项已全部付清！")
            dialog.accept()
            self.complete_payment()

    def update_payment_record_list(self):
        """更新扣款记录列表"""
        if not hasattr(self, 'payment_record_list'):
            return

        self.payment_record_list.clear()

        if not self.payment_records:
            self.payment_record_list.addItem("暂无扣款记录")
            return

        for record in self.payment_records:
            if record['type'] == 'modify':
                display_text = f"[修改金额] {record['time']} 金额调整: {record['amount']:+.2f}元 → ¥{record['new_total']:.2f}"
            else:
                display_text = f"[扣款] {record['time']} 扣款 ¥{record['amount']:.2f} 剩余 ¥{record['remaining']:.2f}"

            item = QListWidgetItem(display_text)
            self.payment_record_list.addItem(item)

    def print_payment_receipt(self, items, total_amount, payment_records):
        """打印付款清单"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            receipt_lines = []
            receipt_lines.append("=" * 48)
            receipt_lines.append("           AI智能称重系统 - 购物清单")
            receipt_lines.append("=" * 48)
            receipt_lines.append(f"时间: {timestamp}")
            receipt_lines.append("-" * 48)
            receipt_lines.append(f"{'商品名称':<12} {'重量(kg)':<10} {'单价(元/kg)':<12} {'金额(元)':<10}")
            receipt_lines.append("-" * 48)

            total_weight = 0
            for item in items:
                product_name = item['product_name']
                weight = item['weight']
                price = item['price']
                total = item['total']
                total_weight += weight
                receipt_lines.append(f"{product_name:<12} {weight:<10.3f} {price:<12.2f} {total:<10.2f}")

            receipt_lines.append("-" * 48)
            receipt_lines.append(f"{'合计':<12} {total_weight:<10.3f} {'':<12} {total_amount:<10.2f}")
            receipt_lines.append("=" * 48)

            # 添加付款记录
            if payment_records:
                receipt_lines.append("")
                receipt_lines.append("付款记录:")
                receipt_lines.append("-" * 48)
                for record in payment_records:
                    if record['type'] == 'payment':
                        receipt_lines.append(
                            f"  {record['time']}  收款 ¥{record['amount']:.2f}  剩余 ¥{record['remaining']:.2f}")
                    elif record['type'] == 'modify':
                        receipt_lines.append(f"  {record['time']}  金额调整: {record['amount']:+.2f}元")
                receipt_lines.append("-" * 48)

            receipt_lines.append("")
            receipt_lines.append("        感谢您的光临，欢迎再次惠顾！")
            receipt_lines.append("=" * 48)

            receipt_text = "\n".join(receipt_lines)

            # 弹出打印选项对话框
            reply = QMessageBox.question(self, "打印清单",
                                         f"{receipt_text}\n\n是否打印此清单？",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                # 保存到文件
                filename = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(receipt_text)

                # 尝试发送到打印机
                if self.print_to_printer(receipt_text):
                    QMessageBox.information(self, "打印成功", f"购物清单已打印\n并保存至: {filename}")
                else:
                    QMessageBox.information(self, "保存成功", f"购物清单已保存至: {filename}\n请手动打印")

                self.statusBar().showMessage(f"购物清单已保存: {filename}", 3000)

                # 语音播报打印完成
                if self.voice_enabled and self.tts_thread:
                    self.tts_thread.speak("购物清单已打印")
            else:
                # 仅保存不打印
                filename = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(receipt_text)
                self.statusBar().showMessage(f"购物清单已保存: {filename}", 3000)

        except Exception as e:
            print(f"[打印] 打印清单失败: {e}")
            QMessageBox.warning(self, "错误", f"打印清单失败: {str(e)}")

    def print_to_printer(self, text):
        """发送到打印机。

        Linux（DL-58 热敏机 + CUPS raw 队列）：发送 ESC/POS + GBK，
        避免 CUPS 把 UTF-8 中文原样发给打印机导致乱码。
        Windows：保持原来的 notepad /p 方式。
        """
        try:
            import tempfile
            import subprocess

            if sys.platform == 'win32':
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                 suffix='.txt', delete=False) as f:
                    f.write(text)
                    temp_file = f.name
                subprocess.run(['notepad', '/p', temp_file], shell=True, timeout=5)
                QTimer.singleShot(3000, lambda: os.unlink(temp_file) if os.path.exists(temp_file) else None)
                return True

            # ---- Linux: ESC/POS + GBK ----
            text = text.replace('¥', '￥')  # 半角 ¥ 转全角，GBK 才能编码
            payload = bytearray()
            payload += b'\x1b@'      # ESC @ 初始化
            payload += b'\x1c&'      # FS & 进入汉字模式
            payload += text.encode('gbk', errors='replace')
            payload += b'\n\n\n'     # 走纸
            payload += b'\x1dV\x00'  # GS V 0 切纸（没有切刀的机器会忽略）

            with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
                f.write(payload)
                temp_file = f.name

            result = subprocess.run(['lp', '-o', 'raw', temp_file],
                                    capture_output=True, timeout=5)
            os.unlink(temp_file)
            if result.returncode != 0:
                print('[打印] lp 失败: %s' % result.stderr.decode('utf-8', 'replace'))
            return result.returncode == 0
        except Exception as e:
            print(f"[打印] 打印机错误: {e}")
            return False

    def complete_payment(self):
        """完成付款 - 打印购物清单"""
        if self.current_remaining <= 0.01:
            # 保存所有交易记录
            for item in self.pending_items:
                self.db.save_transaction(
                    item['product_id'],
                    item['product_name'],
                    item['weight'],
                    item['price'],
                    item['total'],
                    ""
                )

            # 保存付款记录和待付款项到临时变量
            items_to_print = self.pending_items.copy()
            total_to_print = self.pending_total
            records_to_print = self.payment_records.copy()

            # 标记付款已完成，阻止 on_payment_panel_closed 立即切屏
            self._payment_finished = True

            # 同步支付成功到 F407 LCD Screen3（显示成功面板）
            if hasattr(self, 'serial_thread') and self.serial_thread:
                self.serial_thread.send_lcd_payment_result(True)

            # 关闭支付面板（on_payment_panel_closed 检测到 _payment_finished 会跳过切屏）
            if hasattr(self, 'payment_panel') and self.payment_panel:
                try:
                    self.payment_panel.close()
                except:
                    pass

            # 延迟 2 秒后再切回 Screen1 + 清空列表（让用户看清付款成功提示）
            QTimer.singleShot(2000, self._finish_payment)

            # 存储打印数据供 _finish_payment 使用
            self._receipt_data = (items_to_print, total_to_print, records_to_print)
        else:
            # 更新主界面的待付款总额
            self.pending_total = self.current_remaining
            self.pending_total_label.setText(f"待付款总额: ¥{self.pending_total:.2f}")

            QMessageBox.information(self, "部分付款",
                                    f"已收款 ¥{self.pending_total - self.current_remaining:.2f}\n剩余待付: ¥{self.current_remaining:.2f}")
            self.statusBar().showMessage(
                f"已收款 ¥{self.pending_total - self.current_remaining:.2f}，剩余 ¥{self.current_remaining:.2f}", 3000)

    def _finish_payment(self):
        """付款完成后的延迟回调：切回 Screen1 + 清空列表 + 打印"""
        # 切回 Screen1 并清空待付款列表
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_lcd_switch_screen(1)
        self.clear_pending_list()

        # 刷新交易记录
        self.load_transaction_history()

        self._payment_finished = False

        if hasattr(self, '_receipt_data'):
            items_to_print, total_to_print, records_to_print = self._receipt_data
            del self._receipt_data

            # 弹出打印清单对话框
            reply = QMessageBox.question(self, "付款完成",
                                         f"付款成功！总计 ¥{total_to_print:.2f}\n\n是否打印购物清单？",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                self.print_payment_receipt(items_to_print, total_to_print, records_to_print)

        QMessageBox.information(self, "付款成功", "所有款项已付清！")
        self.statusBar().showMessage("付款成功", 3000)

        # 语音播报付款完成
        if self.voice_enabled and self.tts_thread:
            self.tts_thread.speak("付款成功，感谢您的光临")

    def pay(self):
        """付款 - 打开扫码支付面板（手动点击）"""
        if self.pending_total <= 0:
            QMessageBox.warning(self, "提示", "没有待付款商品")
            return

        self.open_payment_dialog(from_user=True)

    def clear_recognition_list(self):
        """清空识别结果列表"""
        self.recognition_list.clear()
        self.recognition_results = []
        self.recognition_count_label.setText("共 0 个结果")
        print("[识别] 识别结果列表已清空")

    def on_serial_status(self, connected, message):
        if connected:
            self.serial_status_label.setText(f"称重: {message}")
            self.serial_status_label.setStyleSheet("font-size: 16px; color: #0F9D58; font-weight: bold;")
            # 首次串口连通时发送清零命令，同步 F407 去皮状态
            if not self._f407_synced:
                self._f407_synced = True
                self.serial_thread.send_zero()
                print("[同步] 启动时已清零 F407 去皮状态")
        else:
            self.serial_status_label.setText(f"称重: {message}")
            self.serial_status_label.setStyleSheet("font-size: 16px; color: #EA4335; font-weight: bold;")

    def on_serial_error(self, error):
        self.serial_status_label.setText("称重: 错误")
        self.serial_status_label.setStyleSheet("font-size: 16px; color: #EA4335; font-weight: bold;")
        print(f"[串口错误] {error}")

    def update_camera_display(self, qimage, raw_frame):
        # QPixmap 必须在 GUI 线程创建
        pixmap = QPixmap.fromImage(qimage)
        # 低帧率场景用快速缩放，省掉平滑插值的开销
        scaled = pixmap.scaled(self.camera_label.size(),
                               Qt.KeepAspectRatio, Qt.FastTransformation)
        self.camera_label.setPixmap(scaled)

    def on_capture_finished(self, filepath, recognized_names, confidence_list):
        """抓拍完成回调 - 识别处理由 recog_settled 信号驱动，此处仅更新状态"""
        self.capture_in_progress = False
        self.last_captured_image = filepath

        if self.is_zero_weight or max(0, self.current_weight - self.tare_offset) <= 0.01:
            print("[识别] 当前重量为0，忽略识别结果")
            self.ai_info_label.setText("等待物品放置...")
            return

        print(f"[识别] 抓拍完成，等待推理追上...")

    def _on_recog_settled(self):
        """推理稳定后，从摄像头线程取最新 top5 重建识别结果"""
        if self.is_zero_weight or max(0, self.current_weight - self.tare_offset) <= 0.01:
            return

        top5 = self.camera_thread.get_top5_results() if self.camera_thread else []
        if not top5:
            print("[识别] 无推理结果")
            return

        print(f"[识别] 稳定后结果: {[t['name_cn'] for t in top5]}")

        self.recognition_list.clear()
        self.recognition_results = []

        for item in top5:
            name = item['name_cn']
            confidence_percent = item['confidence'] * 100

            matched_products = self.db.search_products_by_name(name)

            if matched_products:
                for product in matched_products:
                    product_name_cn = product[2]
                    product_name_en = product[1]
                    search_name_lower = name.lower()

                    if product_name_cn == name or product_name_en == search_name_lower:
                        match_score = min(100, confidence_percent + 20)
                    elif search_name_lower in product_name_en.lower() or name in product_name_cn:
                        match_score = min(100, confidence_percent + 10)
                    elif product_name_cn in name or product_name_en in search_name_lower:
                        match_score = min(100, confidence_percent + 5)
                    else:
                        match_score = confidence_percent

                    item_data = {
                        'product_id': product[0],
                        'name_en': product[1],
                        'name_cn': product[2],
                        'price': product[3],
                        'confidence': match_score,
                        'recognized_name': name
                    }

                    exists = any(r['product_id'] == product[0] for r in self.recognition_results)
                    if not exists:
                        self.recognition_results.append(item_data)
            else:
                item_data = {
                    'product_id': None,
                    'name_en': '',
                    'name_cn': f"{name} (未找到匹配)",
                    'price': 0,
                    'confidence': confidence_percent,
                    'recognized_name': name
                }
                self.recognition_results.append(item_data)

        self.recognition_results.sort(key=lambda x: x['confidence'], reverse=True)

        unique_results = []
        seen_ids = set()
        for item in self.recognition_results:
            if item['product_id']:
                if item['product_id'] not in seen_ids:
                    seen_ids.add(item['product_id'])
                    unique_results.append(item)
            else:
                unique_results.append(item)
        self.recognition_results = unique_results

        self.update_recognition_list()

        matched = [r for r in self.recognition_results if r['product_id']]
        if matched:
            self.ai_info_label.setText(f"识别到 {matched[0]['name_cn']}")
        else:
            self.ai_info_label.setText("未识别到商品")

        if self.recognition_results and self.recognition_results[0]['product_id'] is not None:
            if self.recognition_results[0]['confidence'] >= 30 and not self._user_selected_product:
                self.select_recognition_item(0)

    def update_recognition_list(self):
        self.recognition_list.clear()

        for item in self.recognition_results:
            if item['product_id']:
                if item['confidence'] >= 80:
                    icon = "✅"
                elif item['confidence'] >= 50:
                    icon = "🟢"
                else:
                    icon = "🟡"
                display_text = f"{icon} {item['name_cn']} ({item['name_en']}) - ¥{item['price']:.2f}/kg"
                pass  # 概率已通过颜色和图标表达，不重复显示
            else:
                display_text = f"❓ {item['name_cn']}"

            list_item = QListWidgetItem(display_text)
            list_item.setData(Qt.UserRole, item)

            if item['product_id']:
                if item['confidence'] >= 80:
                    list_item.setBackground(QColor(220, 255, 220))
                elif item['confidence'] >= 50:
                    list_item.setBackground(QColor(255, 255, 200))
                else:
                    list_item.setBackground(QColor(255, 245, 245))
            else:
                list_item.setBackground(QColor(255, 240, 240))

            self.recognition_list.addItem(list_item)

        matched_count = len([r for r in self.recognition_results if r['product_id']])
        total_count = len(self.recognition_results)
        self.recognition_count_label.setText(f"共 {total_count} 个结果 (匹配 {matched_count} 个)")

    def select_recognition_item(self, index, from_user=False):
        if index < 0 or index >= len(self.recognition_results):
            return

        item_data = self.recognition_results[index]

        if item_data['product_id']:
            product = self.db.get_product_by_id(item_data['product_id'])
            if product:
                self.current_product = product
                self.current_price = product[3]
                self.label_id.setText(product[0])
                self.label_name_en.setText(product[1])
                self.label_name_cn.setText(product[2])
                self.label_price.setText(f"{product[3]:.2f} 元/kg")
                self.label_current_price.setText(f"{product[3]:.2f} 元/kg")
                self.update_total()
                self.statusBar().showMessage(f"已选择: {product[2]}  ¥{product[3]:.2f}/kg", 2000)
                self.ai_info_label.setText(f"识别到 {product[2]}")

                if from_user:
                    self._user_selected_product = True

                # 通知 pending_product_info 供重量稳定后的总价播报使用
                self.pending_product_info = (product[2], product[3])
                self.product_stable_timer.start(500)

    def on_recognition_item_clicked(self, item):
        item_data = item.data(Qt.UserRole)
        if item_data:
            index = self.recognition_results.index(item_data)
            self.select_recognition_item(index, from_user=True)

    def capture_async(self, recognize=True):
        if self.capture_in_progress or not self.camera_thread:
            return None
        self.capture_in_progress = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp}.jpg"
        self.camera_thread.request_capture(filename, recognize)
        return filename

    # ========== 全局样式表 ==========
    _GLOBAL_STYLE = """
        * { font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif; }

        QMainWindow { background: #F0F2F5; }

        QGroupBox {
            font-size: 17px;
            font-weight: bold;
            color: #1a73e8;
            border: 1px solid #DADCE0;
            border-radius: 12px;
            margin-top: 10px;
            padding-top: 18px;
            background: #FFFFFF;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px 0 8px;
            color: #1a73e8;
        }

        QPushButton {
            font-size: 17px;
            font-weight: 600;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            border: none;
        }
        QPushButton:hover { opacity: 0.9; }
        QPushButton:pressed { opacity: 0.8; }

        QDialog QPushButton {
            color: #202124;
        }

        QLineEdit {
            font-size: 18px;
            padding: 12px 16px;
            border: 2px solid #DADCE0;
            border-radius: 8px;
            background: white;
            min-height: 24px;
        }
        QLineEdit:focus { border: 2px solid #1a73e8; background: #F8FBFF; }

        QListWidget {
            font-size: 17px;
            border: 1px solid #DADCE0;
            border-radius: 10px;
            background: white;
            padding: 6px;
            outline: none;
        }
        QListWidget::item {
            padding: 10px 12px;
            border-radius: 6px;
            margin: 2px 0px;
        }
        QListWidget::item:hover { background: #E8F0FE; }
        QListWidget::item:selected { background: #1a73e8; color: white; }

        QLabel { font-size: 17px; color: #3C4043; }

        QStatusBar {
            font-size: 15px;
            color: #5F6368;
            background: #FFFFFF;
            border-top: 1px solid #DADCE0;
            padding: 2px 12px;
        }

        QScrollBar:vertical {
            border: none;
            background: #F0F2F5;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #C4C7CC;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #9AA0A6; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

        QTableWidget {
            font-size: 16px;
            border: 1px solid #DADCE0;
            border-radius: 10px;
            background: white;
            gridline-color: #E8EAED;
        }
        QTableWidget::item { padding: 8px; }
        QTableWidget::item:selected { background: #1a73e8; color: white; }
        QHeaderView::section {
            background: #F8F9FA;
            border: none;
            border-bottom: 2px solid #DADCE0;
            padding: 10px;
            font-weight: bold;
            font-size: 16px;
        }
    """

    def init_ui(self):
        # 应用全局样式
        self.setStyleSheet(self._GLOBAL_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 12, 16, 12)

        # --- 标题栏 ---
        title_bar = QFrame()
        title_bar.setFixedHeight(68)
        title_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0D47A1, stop:0.4 #1565C0, stop:0.7 #1976D2, stop:1 #1a73e8);
                border-radius: 12px;
            }
        """)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(20, 0, 20, 0)

        title_icon = QLabel("⚖️")
        title_icon.setStyleSheet("font-size: 34px; border: none;")
        title_bar_layout.addWidget(title_icon)

        title = QLabel("灵眸 · 智衡")
        title.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #FFFFFF; letter-spacing: 3px;")
        title_bar_layout.addWidget(title)

        title_bar_layout.addStretch()

        subtitle = QLabel("全自助视觉称重收费系统")
        subtitle.setStyleSheet("font-size: 17px; color: #BBDEFB; font-weight: 500;")
        title_bar_layout.addWidget(subtitle)

        main_layout.addWidget(title_bar)

        # --- 主体三列布局 ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # ==================== 左侧：商品区域 (stretch=2) ====================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 搜索框
        search_label = QLabel("商品搜索")
        search_label.setStyleSheet(
            "font-size: 17px; font-weight: 700; color: #5F6368; "
            "text-transform: uppercase; letter-spacing: 1px; padding: 2px 4px;")
        left_layout.addWidget(search_label)

        self.search_widget = SearchWidget()
        self.search_widget.product_selected.connect(self.on_product_selected)
        self.search_widget.line_edit.setStyleSheet("""
            QLineEdit {
                border: 2px solid #DADCE0;
                border-radius: 10px;
                padding: 14px 16px;
                font-size: 18px;
                background: white;
                min-height: 26px;
            }
            QLineEdit:focus { border: 2px solid #1a73e8; background: #F8FBFF; }
        """)
        left_layout.addWidget(self.search_widget)

        # 商品信息卡片
        info_group = QGroupBox("商品信息")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(7)
        info_layout.setContentsMargins(14, 18, 14, 12)

        for i, (k, v) in enumerate([
            ("编号", "label_id"), ("英文名", "label_name_en"),
            ("中文名", "label_name_cn"), ("单价", "label_price")
        ]):
            lbl = QLabel(k)
            lbl.setStyleSheet("font-size: 16px; color: #80868B; font-weight: normal;")
            info_layout.addWidget(lbl, i, 0)

        self.label_id = QLabel("-")
        self.label_id.setStyleSheet("font-size: 19px; font-weight: bold; color: #1a73e8;")
        info_layout.addWidget(self.label_id, 0, 1)
        self.label_name_en = QLabel("-")
        self.label_name_en.setStyleSheet("font-size: 16px; color: #5F6368;")
        info_layout.addWidget(self.label_name_en, 1, 1)
        self.label_name_cn = QLabel("-")
        self.label_name_cn.setStyleSheet("font-size: 22px; font-weight: bold; color: #0F9D58;")
        info_layout.addWidget(self.label_name_cn, 2, 1)
        self.label_price = QLabel("-")
        self.label_price.setStyleSheet("font-size: 20px; font-weight: bold; color: #FF6D00;")
        info_layout.addWidget(self.label_price, 3, 1)
        left_layout.addWidget(info_group)

        # 价格设置卡片
        price_group = QGroupBox("价格设置")
        price_layout = QHBoxLayout(price_group)
        price_layout.setContentsMargins(14, 12, 14, 12)
        price_layout.setSpacing(8)
        self.edit_manual_price = QLineEdit()
        self.edit_manual_price.setPlaceholderText("输入价格 (元/kg)")
        self.edit_manual_price.setMinimumHeight(42)
        btn_set_price = QPushButton("设置")
        btn_set_price.setMinimumHeight(42)
        btn_set_price.setMinimumWidth(72)
        btn_set_price.setStyleSheet(
            "background: #1a73e8; font-size: 17px; font-weight: 600;")
        btn_set_price.clicked.connect(self.manual_set_price)
        price_layout.addWidget(self.edit_manual_price, stretch=1)
        price_layout.addWidget(btn_set_price)
        left_layout.addWidget(price_group)

        # 商品管理按钮
        btn_manage = QPushButton("管理商品库")
        btn_manage.setMinimumHeight(48)
        btn_manage.setStyleSheet(
            "background: #1a73e8; font-size: 18px; font-weight: 600;")
        btn_manage.clicked.connect(self.open_product_manage)
        left_layout.addWidget(btn_manage)

        # LCD 翻页按钮（小号、低调，放在管理商品库下方空白区）
        lcd_page_layout = QHBoxLayout()
        lcd_page_layout.setSpacing(8)
        lcd_page_layout.setContentsMargins(4, 8, 4, 4)
        btn_lcd_prev = QPushButton("◀")
        btn_lcd_prev.setMinimumHeight(28)
        btn_lcd_prev.setMaximumWidth(60)
        btn_lcd_prev.setStyleSheet(
            "background: #BDC1C6; font-size: 13px; font-weight: 600;"
            "border-radius: 4px; padding: 2px 4px;")
        btn_lcd_prev.clicked.connect(self.lcd_page_down)
        lcd_page_layout.addWidget(btn_lcd_prev)

        btn_lcd_next = QPushButton("▶")
        btn_lcd_next.setMinimumHeight(28)
        btn_lcd_next.setMaximumWidth(60)
        btn_lcd_next.setStyleSheet(
            "background: #BDC1C6; font-size: 13px; font-weight: 600;"
            "border-radius: 4px; padding: 2px 4px;")
        btn_lcd_next.clicked.connect(self.lcd_page_up)
        lcd_page_layout.addWidget(btn_lcd_next)

        lcd_page_layout.addStretch()
        left_layout.addLayout(lcd_page_layout)

        left_layout.addStretch()

        # ==================== 中间：摄像头 + 识别 (stretch=5) ====================
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setSpacing(10)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # 摄像头卡片
        camera_group = QGroupBox("实时监控")
        camera_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setContentsMargins(10, 24, 10, 10)
        camera_layout.setSpacing(8)

        self.camera_label = QLabel("等待摄像头...")
        self.camera_label.setMinimumSize(560, 520)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet(
            "border: 3px solid #1a73e8; background: #0D1117; color: #8AB4F8; "
            "border-radius: 14px; qproperty-alignment: AlignCenter;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(False)
        camera_layout.addWidget(self.camera_label, stretch=1)

        # AI 状态栏
        ai_frame = QFrame()
        ai_frame.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #E8F5E9, stop:1 #C8E6C9); border-radius: 8px;")
        ai_frame_layout = QHBoxLayout(ai_frame)
        ai_frame_layout.setContentsMargins(12, 10, 12, 10)
        self.ai_info_label = QLabel("AI 等待识别...")
        self.ai_info_label.setStyleSheet(
            "font-size: 19px; color: #2E7D32; font-weight: bold; border: none; background: transparent;")
        self.ai_info_label.setAlignment(Qt.AlignCenter)
        ai_frame_layout.addWidget(self.ai_info_label)
        camera_layout.addWidget(ai_frame)

        btn_capture = QPushButton("手动抓拍识别")
        btn_capture.setMinimumHeight(42)
        btn_capture.setStyleSheet(
            "background: #1a73e8; font-size: 17px; font-weight: 600;")
        btn_capture.clicked.connect(self.manual_capture)
        camera_layout.addWidget(btn_capture)

        center_layout.addWidget(camera_group, stretch=4)

        # 识别结果卡片
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 12, 10, 8)
        result_layout.setSpacing(6)

        result_title_layout = QHBoxLayout()
        result_title = QLabel("识别到的食材")
        result_title.setStyleSheet("font-size: 16px; color: #80868B; font-weight: normal;")
        result_title_layout.addWidget(result_title)
        result_title_layout.addStretch()
        self.recognition_count_label = QLabel("共 0 个结果")
        self.recognition_count_label.setStyleSheet("font-size: 16px; color: #80868B;")
        result_title_layout.addWidget(self.recognition_count_label)
        result_layout.addLayout(result_title_layout)

        self.recognition_list = QListWidget()
        self.recognition_list.setMinimumHeight(110)
        self.recognition_list.setMaximumHeight(180)
        self.recognition_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                border: 1px solid #DADCE0;
                border-radius: 10px;
                background: white;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 4px 6px;
                border-radius: 4px;
                margin: 1px 0px;
            }
            QListWidget::item:hover { background: #E8F0FE; }
            QListWidget::item:selected { background: #1a73e8; color: white; }
        """)
        self.recognition_list.itemClicked.connect(self.on_recognition_item_clicked)
        result_layout.addWidget(self.recognition_list)

        center_layout.addWidget(result_group)

        # ==================== 右侧：称重 + 结算 (stretch=3) ====================
        right_widget = QWidget()
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 状态指示（紧凑横排）
        status_frame = QFrame()
        status_frame.setStyleSheet(
            "background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 10px;")
        status_frame_layout = QHBoxLayout(status_frame)
        status_frame_layout.setContentsMargins(12, 10, 12, 10)
        status_frame_layout.setSpacing(16)

        self.serial_status_label = QLabel("称重: 未连接")
        self.serial_status_label.setStyleSheet("font-size: 16px; color: #80868B; font-weight: 600;")
        self.serial_status_label.setMinimumWidth(130)
        status_frame_layout.addWidget(self.serial_status_label)

        self.voice_status_label = QLabel("语音: 未启用")
        self.voice_status_label.setStyleSheet("font-size: 16px; color: #80868B; font-weight: 600;")
        self.voice_status_label.setMinimumWidth(150)
        self.voice_status_label.setWordWrap(False)
        status_frame_layout.addWidget(self.voice_status_label)
        status_frame_layout.addStretch()

        btn_config = QPushButton("配置")
        btn_config.setMinimumHeight(32)
        btn_config.setMinimumWidth(60)
        btn_config.setStyleSheet(
            "background: #5F6368; font-size: 15px; font-weight: 600; padding: 4px 14px;")
        btn_config.clicked.connect(self.open_serial_config)
        status_frame_layout.addWidget(btn_config)

        right_layout.addWidget(status_frame)

        # ===== QTabWidget: 称重结算 | 交易记录 =====
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #DADCE0; border-radius: 8px; background: #F0F2F5; }
            QTabBar::tab { font-size: 14px; font-weight: bold; padding: 6px 10px; min-width: 80px;
                           border: 1px solid #DADCE0; border-radius: 8px 8px 0 0;
                           background: #E8EAED; color: #5F6368; margin-right: 2px; }
            QTabBar::tab:selected { background: #FFFFFF; color: #1a73e8;
                                    border-bottom: 2px solid #1a73e8; }
        """)

        # ----- Tab 0: 称重结算 -----
        settle_tab = QWidget()
        settle_tab_layout = QVBoxLayout(settle_tab)
        settle_tab_layout.setSpacing(6)
        settle_tab_layout.setContentsMargins(4, 6, 4, 4)

        # 称重显示卡片
        weight_group = QGroupBox("称重信息")
        weight_group.setMinimumHeight(195)
        weight_layout = QVBoxLayout(weight_group)
        weight_layout.setContentsMargins(12, 10, 12, 8)
        weight_layout.setSpacing(2)

        self.weight_display = QLabel("0.000")
        self.weight_display.setAlignment(Qt.AlignCenter)
        self.weight_display.setMinimumHeight(90)
        self.weight_display.setStyleSheet(
            "font-size: 78px; font-weight: bold; color: #1a73e8; "
            "background: #E8F0FE; border-radius: 12px; padding: 10px;")

        weight_unit = QLabel("千克 (kg)")
        weight_unit.setAlignment(Qt.AlignCenter)
        weight_unit.setStyleSheet("font-size: 16px; color: #80868B; padding: 0px;")

        weight_layout.addWidget(self.weight_display)
        weight_layout.addWidget(weight_unit)

        w_btn_layout = QHBoxLayout()
        w_btn_layout.setSpacing(8)
        self.btn_tare = QPushButton("去皮")
        self.btn_clear = QPushButton("清零")
        for btn in [self.btn_tare, self.btn_clear]:
            btn.setMinimumHeight(42)
            btn.setStyleSheet("background: #5F6368; font-size: 17px; font-weight: 600; padding: 8px 16px;")
        self.btn_tare.clicked.connect(self.tare_scale)
        self.btn_clear.clicked.connect(self.clear_weight)
        w_btn_layout.addWidget(self.btn_tare)
        w_btn_layout.addWidget(self.btn_clear)
        weight_layout.addLayout(w_btn_layout)
        settle_tab_layout.addWidget(weight_group)

        # 结算卡片
        settle_group = QGroupBox("实时结算")
        settle_group.setMinimumHeight(150)
        settle_layout = QGridLayout(settle_group)
        settle_layout.setSpacing(5)
        settle_layout.setContentsMargins(14, 12, 14, 10)

        for i, (k, label_attr, val) in enumerate([
            ("当前重量", "label_current_weight", "0.000 kg"),
            ("商品单价", "label_current_price", "0.00 元/kg"),
        ]):
            lbl = QLabel(k)
            lbl.setStyleSheet("font-size: 16px; color: #80868B; font-weight: normal;")
            settle_layout.addWidget(lbl, i, 0)
            setattr(self, label_attr, QLabel(val))
            getattr(self, label_attr).setStyleSheet(
                "font-size: 18px; font-weight: 600; color: #1a73e8;")
            settle_layout.addWidget(getattr(self, label_attr), i, 1)

        settle_layout.addWidget(QLabel("结算金额"), 2, 0)
        self.label_total_amount = QLabel("0.00 元")
        self.label_total_amount.setStyleSheet(
            "font-size: 38px; font-weight: bold; color: #EA4335; "
            "background: #FCE8E6; border-radius: 10px; padding: 10px 14px;")
        settle_layout.addWidget(self.label_total_amount, 2, 1)
        settle_tab_layout.addWidget(settle_group)

        # 待付款卡片
        pending_group = QGroupBox("待付款列表")
        pending_group.setMinimumHeight(120)
        pending_layout = QVBoxLayout(pending_group)
        pending_layout.setContentsMargins(10, 8, 10, 6)
        pending_layout.setSpacing(4)

        self.pending_list = QListWidget()
        self.pending_list.setMinimumHeight(60)
        self.pending_list.setMaximumHeight(135)
        self.pending_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                border: 1px solid #DADCE0;
                border-radius: 10px;
                background: white;
                padding: 2px;
                outline: none;
            }
            QListWidget::item {
                padding: 3px 8px;
                border-radius: 4px;
                margin: 1px 0px;
            }
            QListWidget::item:hover { background: #E8F0FE; }
            QListWidget::item:selected { background: #1a73e8; color: white; }
        """)
        pending_layout.addWidget(self.pending_list)

        # 待付款总额
        pending_total_frame = QFrame()
        pending_total_frame.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #FFF3E0, stop:1 #FFE0B2); border-radius: 10px;")
        pending_total_layout = QHBoxLayout(pending_total_frame)
        pending_total_layout.setContentsMargins(12, 8, 12, 8)
        self.pending_total_label = QLabel("总额: ¥0.00")
        self.pending_total_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #FF6D00;")
        pending_total_layout.addWidget(self.pending_total_label)
        pending_total_layout.addStretch()
        pending_layout.addWidget(pending_total_frame)

        # 待付款按钮行
        pending_btn_layout = QHBoxLayout()
        pending_btn_layout.setSpacing(8)
        btn_remove_pending = QPushButton("删除")
        btn_remove_pending.setMinimumHeight(42)
        btn_remove_pending.setStyleSheet(
            "background: #EA4335; font-size: 17px; font-weight: 600;")
        btn_remove_pending.clicked.connect(self.remove_selected_pending_item)
        pending_btn_layout.addWidget(btn_remove_pending)

        btn_pay = QPushButton("扫码付款")
        btn_pay.setMinimumHeight(48)
        btn_pay.setStyleSheet(
            "background: #0F9D58; font-size: 20px; font-weight: bold;")
        btn_pay.clicked.connect(self.pay)
        pending_btn_layout.addWidget(btn_pay)

        pending_layout.addLayout(pending_btn_layout)
        settle_tab_layout.addWidget(pending_group)

        # 语音测试
        btn_test_voice = QPushButton("测试语音播报")
        btn_test_voice.setMinimumHeight(42)
        btn_test_voice.setStyleSheet(
            "background: #FF6D00; font-size: 17px; font-weight: 600;")
        btn_test_voice.clicked.connect(self.test_voice)
        settle_tab_layout.addWidget(btn_test_voice)

        settle_tab_layout.addStretch()
        self.right_tabs.addTab(settle_tab, "💳 称重结算")

        # ----- Tab 1: 交易记录 -----
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setSpacing(6)
        history_layout.setContentsMargins(4, 6, 4, 4)

        # 今日总额
        history_top = QHBoxLayout()
        history_top.addWidget(QLabel("今日总额:"))
        self.label_daily_total = QLabel("¥0.00")
        self.label_daily_total.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #0F9D58;")
        history_top.addWidget(self.label_daily_total)
        history_top.addStretch()
        btn_refresh_history = QPushButton("刷新")
        btn_refresh_history.setMinimumHeight(32)
        btn_refresh_history.setStyleSheet(
            "background: #1a73e8; font-size: 14px; font-weight: 600; padding: 4px 12px;")
        btn_refresh_history.clicked.connect(self.load_transaction_history)
        history_top.addWidget(btn_refresh_history)
        history_layout.addLayout(history_top)

        # 交易记录表格
        self.transaction_table = QTableWidget()
        self.transaction_table.setColumnCount(6)
        self.transaction_table.setHorizontalHeaderLabels(
            ["ID", "时间", "商品", "重量(kg)", "单价", "金额(元)"])
        self.transaction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transaction_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transaction_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transaction_table.setMinimumHeight(200)
        self.transaction_table.verticalHeader().setDefaultSectionSize(28)  # 行高
        self.transaction_table.verticalHeader().setVisible(False)
        self.transaction_table.setColumnWidth(0, 40)   # ID列收窄
        history_layout.addWidget(self.transaction_table, stretch=1)

        # 操作按钮行
        history_btn_layout = QHBoxLayout()
        self.btn_show_receipt = QPushButton("查看清单")
        self.btn_show_receipt.setMinimumHeight(36)
        self.btn_show_receipt.setStyleSheet(
            "background: #FF6D00; font-size: 15px; font-weight: 600;")
        self.btn_show_receipt.clicked.connect(self.show_receipt_for_transaction)
        history_btn_layout.addWidget(self.btn_show_receipt)

        btn_del_one = QPushButton("删除选中")
        btn_del_one.setMinimumHeight(36)
        btn_del_one.setStyleSheet(
            "background: #EA4335; font-size: 15px; font-weight: 600;")
        btn_del_one.clicked.connect(self.delete_selected_transaction)
        history_btn_layout.addWidget(btn_del_one)

        btn_del_all = QPushButton("全部删除")
        btn_del_all.setMinimumHeight(36)
        btn_del_all.setStyleSheet(
            "background: #C62828; font-size: 15px; font-weight: 600;")
        btn_del_all.clicked.connect(self.delete_all_transactions)
        history_btn_layout.addWidget(btn_del_all)

        history_btn_layout.addStretch()
        history_layout.addLayout(history_btn_layout)

        self.receipt_preview = QLabel("选中一条记录后点击「查看清单」")
        self.receipt_preview.setMinimumHeight(80)
        self.receipt_preview.setWordWrap(True)
        self.receipt_preview.setStyleSheet(
            "font-size: 13px; color: #3C4043; background: #F8F9FA; "
            "border: 1px solid #DADCE0; border-radius: 6px; padding: 8px;")
        history_layout.addWidget(self.receipt_preview)

        self.right_tabs.addTab(history_tab, "📋 交易记录")

        # 切换 tab 时自动刷新
        self.right_tabs.currentChanged.connect(self._on_tab_changed)

        right_layout.addWidget(self.right_tabs)

        # --- 右侧滚动区域 ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 8px; background: transparent; }"
            "QScrollBar::handle:vertical { background: #C4C7CC; border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )

        # --- 三列占比：左:中:右 = 2:5:3 ---
        content_layout.addWidget(left_widget, stretch=2)
        content_layout.addWidget(center_widget, stretch=5)
        content_layout.addWidget(right_scroll, stretch=3)
        main_layout.addLayout(content_layout, stretch=1)

        # 状态栏
        platform_info = (
            f"平台: {'Windows' if IS_WINDOWS else 'Linux (Muse Pi Pro)'}  |  "
            f"称重: {WEIGHT_PORT}  |  TTS: {VOICE_PORT}  |  摄像头: {CAMERA_ID}"
        )
        self.statusBar().showMessage(platform_info)

    def test_voice(self):
        """测试语音播报"""
        if self.voice_enabled and self.tts_thread:
            self.tts_thread.speak("语音播报测试，系统工作正常")
            self.statusBar().showMessage("语音测试已发送", 2000)
        else:
            QMessageBox.warning(self, "提示", "语音功能未启用，请检查串口配置")

    def open_serial_config(self):
        dialog = SerialConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            weight_port, weight_baud, voice_port, voice_baud, voice_enabled = dialog.get_config()

            # 重启称重串口
            if self.serial_thread:
                self.serial_thread.stop()
            self.serial_thread = SerialThread(port=weight_port, baudrate=weight_baud)
            self.serial_thread.weight_updated.connect(self.on_weight_received)
            self.serial_thread.connection_status.connect(self.on_serial_status)
            self.serial_thread.error_occurred.connect(self.on_serial_error)
            self.serial_thread.start()

            # 重启语音串口
            self.voice_enabled = voice_enabled
            if self.tts_thread:
                self.tts_thread.stop()
            if voice_enabled:
                self.tts_thread = TTSThread(port=voice_port, baudrate=voice_baud, enabled=True)
                self.tts_thread.start()
                self.voice_status_label.setText("语音: 连接中...")
                self.voice_status_label.setStyleSheet("font-size: 16px; color: #FF6D00; font-weight: bold;")
                QTimer.singleShot(3000, self._check_tts_connected)
            else:
                self.voice_status_label.setText("语音: 已禁用")
                self.voice_status_label.setStyleSheet("font-size: 16px; color: #80868B; font-weight: 600;")

    def load_products(self):
        try:
            self.all_products = self.db.get_all_products()
            self.search_widget.set_all_products(self.all_products)
        except Exception as e:
            print(f"[错误] 加载商品失败: {e}")
            self.all_products = []

    def _on_tab_changed(self, index):
        if index == 1:
            self.load_transaction_history()

    def load_transaction_history(self):
        try:
            transactions = self.db.get_all_transactions()
            daily_total = self.db.get_daily_total()
            self.label_daily_total.setText(f"¥{daily_total:.2f}")
            self.label_daily_total.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #0F9D58;")

            self.transaction_table.setRowCount(0)
            for row, t in enumerate(transactions):
                self.transaction_table.insertRow(row)
                tid = str(t[0]) if len(t) > 0 else ""
                items = [
                    (tid,),
                    (t[5] if len(t) > 5 else "",),
                    (t[1] if len(t) > 1 else "",),
                    (f"{t[2]:.3f}" if len(t) > 2 else "",),
                    (f"{t[3]:.2f}" if len(t) > 3 else "",),
                    (f"¥{t[4]:.2f}" if len(t) > 4 else "",),
                ]
                for col, (text,) in enumerate(items):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    if col == 5:
                        item.setForeground(QColor("#EA4335"))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    self.transaction_table.setItem(row, col, item)

            count = self.db.get_transaction_count()
            self.receipt_preview.setText(f"共 {count} 条交易记录 | 选中一条后点击「查看清单」")
        except Exception as e:
            print(f"[交易记录] 加载失败: {e}")

    def show_receipt_for_transaction(self):
        current_row = self.transaction_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "提示", "请先在表格中选中一条交易记录")
            return

        tid = self.transaction_table.item(current_row, 0).text()
        trans_time = self.transaction_table.item(current_row, 1).text()
        product = self.transaction_table.item(current_row, 2).text()
        weight = self.transaction_table.item(current_row, 3).text()
        price = self.transaction_table.item(current_row, 4).text()
        total = self.transaction_table.item(current_row, 5).text()

        receipt = (
            "═══════════════════════════════════\n"
            "      AI智能称重系统 - 购物清单\n"
            "═══════════════════════════════════\n"
            f"编号: {tid}\n"
            f"时间: {trans_time}\n"
            "───────────────────────────────────\n"
            f"商品: {product}\n"
            f"重量: {weight} kg\n"
            f"单价: {price} 元/kg\n"
            f"金额: {total}\n"
            "───────────────────────────────────\n"
            "     感谢您的光临，欢迎再次惠顾！\n"
            "═══════════════════════════════════"
        )
        self.receipt_preview.setText(receipt)
        self.statusBar().showMessage(f"已加载清单: {product}", 3000)

    def delete_selected_transaction(self):
        current_row = self.transaction_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "提示", "请先在表格中选中一条交易记录")
            return
        tid = int(self.transaction_table.item(current_row, 0).text())
        reply = QMessageBox.question(self, "确认删除",
            f"确定删除编号为 {tid} 的交易记录吗？",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_transaction_by_id(tid)
            self.load_transaction_history()
            self.statusBar().showMessage(f"已删除交易 #{tid}", 3000)

    def delete_all_transactions(self):
        count = self.db.get_transaction_count()
        if count == 0:
            QMessageBox.information(self, "提示", "没有交易记录可删除")
            return
        reply = QMessageBox.question(self, "确认全部删除",
            f"确定删除全部 {count} 条交易记录吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_all_transactions()
            self.load_transaction_history()
            self.statusBar().showMessage("全部交易记录已删除", 3000)

    def on_product_selected(self, product):
        self.current_product = product
        self.current_price = product[3]
        self.label_id.setText(product[0])
        self.label_name_en.setText(product[1])
        self.label_name_cn.setText(product[2])
        self.label_price.setText(f"{product[3]:.2f} 元/kg")
        self.label_current_price.setText(f"{product[3]:.2f} 元/kg")
        self.ai_info_label.setText(f"已选择: {product[2]}  ¥{product[3]:.2f}/kg")
        self.update_total()
        self.statusBar().showMessage(f"已选择: {product[2]}  ¥{product[3]:.2f}/kg", 2000)

        # 使用防抖播报：等待稳定后再播报，避免重复
        self.pending_product_info = (product[2], product[3])
        self.product_stable_timer.start(500)

    def manual_set_price(self):
        try:
            price = float(self.edit_manual_price.text().strip())
            if price <= 0:
                QMessageBox.warning(self, "错误", "价格必须大于0")
                return
            self.current_price = price
            self.label_price.setText(f"{price:.2f} 元/kg")
            self.label_current_price.setText(f"{price:.2f} 元/kg")
            self.update_total()
            self.edit_manual_price.clear()
            self.statusBar().showMessage(f"单价已设置为: ¥{price:.2f}/kg", 2000)

            # 播报新设置的价格
            if self.current_product:
                self.speak_product_price(self.current_product[2], price)
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的价格")

    def manual_capture(self):
        self.capture_async(recognize=True)
        self.statusBar().showMessage("正在抓拍识别...", 2000)

    def tare_scale(self):
        self.tare_offset = self.current_weight
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_tare(self.current_weight)
        self.update_total()
        self.statusBar().showMessage(f"去皮已设置 (偏移: {self.tare_offset:.3f} kg)", 2000)

    def clear_weight(self):
        self.tare_offset = 0
        self.current_weight = 0
        self.last_weight = 0
        if hasattr(self, 'serial_thread') and self.serial_thread:
            self.serial_thread.send_zero()
        self.update_total()
        self.clear_recognition_list()
        self.statusBar().showMessage("重量已清零", 2000)
        self.stable_timer.stop()
        self.zero_stable_timer.stop()
        self.pending_zero_weight = 0.0
        self._last_nonzero_weight = 0.0
        self._stable_timer_weight = 0.0
        self._stable_capture_needed = False
        self._recap_retries = 0
        self._recap_baseline = 0.0
        self.last_spoken_total = -1.0
        self.last_spoken_product_info = None

    def update_total(self):
        display_weight = max(0, self.current_weight - self.tare_offset)
        # 净重 < 3g 当作 0 显示，过滤传感器微小噪声
        if display_weight < 0.003:
            display_weight = 0.0
        self.total_price = display_weight * self.current_price
        display_grams = display_weight * 1000

        self.weight_display.setText(f"{display_weight:.3f}")
        self.label_current_weight.setText(f"{display_weight:.3f} kg ({display_grams:.0f} g)")
        self.label_total_amount.setText(f"{self.total_price:.2f} 元")


    def open_product_manage(self):
        dialog = ProductManageDialog(self.db, self)
        dialog.exec_()
        QTimer.singleShot(100, self.load_products)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.serial_thread:
            self.serial_thread.stop()
        if self.tts_thread:
            self.tts_thread.stop()
        if hasattr(self, 'qr_timer') and self.qr_timer:
            self.qr_timer.stop()
        if hasattr(self, 'payment_panel') and self.payment_panel:
            try:
                self.payment_panel.close()
            except:
                pass
        self.stable_timer.stop()
        self.product_stable_timer.stop()
        self.zero_stable_timer.stop()
        event.accept()


if __name__ == "__main__":
    print("=" * 50)
    print(f"AI智能称重系统启动")
    print(f"平台: {'Windows' if IS_WINDOWS else 'Linux (Muse Pi Pro)'}")
    print(f"称重串口: {WEIGHT_PORT}")
    print(f"语音串口: {VOICE_PORT}")
    print(f"摄像头ID: {CAMERA_ID}")
    print(f"语音控制: 复用称重串口 {WEIGHT_PORT} (AA xx FE 55)")
    print("=" * 50)

    app = QApplication(sys.argv)
    window = AIScaleSystem()
    window.show()
    sys.exit(app.exec_())







