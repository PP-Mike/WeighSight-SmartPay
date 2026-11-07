"""摄像头采集配置对比：测各种后端/格式/分辨率下的实际帧率。

用法：
    python3 camera_bench.py            # 默认 /dev/video20
    python3 camera_bench.py 20         # 指定摄像头 ID
"""

import sys
import time

import cv2

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N = 60          # 每种配置采样帧数


def fourcc_str(v):
    try:
        v = int(v)
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "?"


def run_case(name, open_fn, settings):
    cap = open_fn()
    if not cap or not cap.isOpened():
        print(f"{name:28s} 打开失败")
        return
    for prop, value in settings:
        cap.set(prop, value)
    # 预热
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_req = cap.get(cv2.CAP_PROP_FPS)
    cc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))

    t0 = time.time()
    got = 0
    for _ in range(N):
        ok, _f = cap.read()
        if ok:
            got += 1
    dt = time.time() - t0
    cap.release()

    fps = got / dt if dt > 0 else 0
    print(f"{name:28s} {w}x{h} {cc:5s} 请求{fps_req:5.0f}fps | "
          f"实际 {fps:5.1f} fps | {got}/{N} 帧成功")


CASES = [
    ("默认后端 640x480", lambda: cv2.VideoCapture(CAM),
     [(cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480)]),

    ("V4L2 640x480", lambda: cv2.VideoCapture(CAM, cv2.CAP_V4L2),
     [(cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480)]),

    ("V4L2+MJPG 640x480@30", lambda: cv2.VideoCapture(CAM, cv2.CAP_V4L2),
     [(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
      (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
      (cv2.CAP_PROP_FPS, 30)]),

    ("V4L2+YUYV 640x480@30", lambda: cv2.VideoCapture(CAM, cv2.CAP_V4L2),
     [(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
      (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
      (cv2.CAP_PROP_FPS, 30)]),

    ("默认后端 320x240", lambda: cv2.VideoCapture(CAM),
     [(cv2.CAP_PROP_FRAME_WIDTH, 320), (cv2.CAP_PROP_FRAME_HEIGHT, 240)]),

    ("默认后端 不设分辨率", lambda: cv2.VideoCapture(CAM), []),
]


def main():
    print(f"摄像头 /dev/video{CAM}，每种配置采 {N} 帧\n")
    print(f"{'配置':28s} {'实际参数':18s} {'结果'}")
    print("-" * 78)
    for name, opener, settings in CASES:
        try:
            run_case(name, opener, settings)
        except Exception as e:
            print(f"{name:28s} 异常: {e}")


if __name__ == "__main__":
    main()
