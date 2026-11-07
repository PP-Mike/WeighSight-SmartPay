"""ONNX 分类模型性能与准确率测试（对应 best6.onnx, 320x320, 13 类）。

用法：
    # 1) 测单帧推理耗时（用摄像头实时抽帧）
    python3 bench_onnx.py --model ./model/best6.onnx --latency

    # 2) 测准确率（文件夹按类别命名）
    python3 bench_onnx.py --model ./model/best6.onnx --acc ./dataset

    # 3) 两个都测
    python3 bench_onnx.py --model ./model/best6.onnx --latency --acc ./dataset

数据集目录结构（准确率模式）：
    dataset/
    ├── apple/        *.jpg
    ├── peach/        *.jpg
    ├── pear/         *.jpg
    └── ...
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("未安装 onnxruntime：pip3 install onnxruntime")

# 与主程序保持一致
INGREDIENTS = [
    "Bitter melon", "Carrot", "Green apple",
    "Lemon", "Pomegranate", "Tomato",
    "apple", "eggplant", "orange",
    "peach", "pear", "pumpkin",
    "watermelon",
]
INPUT_SIZE = (320, 320)


def build_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4          # K1 有 8 核，4 线程实测最优
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def preprocess(img):
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, INPUT_SIZE, swapRB=True)
    return blob


def infer(sess, in_name, out_name, img):
    out = sess.run([out_name], {in_name: preprocess(img)})[0]
    probs = out[0] if out.ndim == 2 else out.flatten()
    return probs


def test_latency(sess, in_name, out_name, cam_id=20, runs=60):
    print("\n== 单帧推理延迟（摄像头抽帧）==")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头 {cam_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = []
    for _ in range(120):
        ok, f = cap.read()
        if ok:
            frames.append(f)
        if len(frames) >= runs + 10:
            break
    cap.release()

    if len(frames) < 10:
        raise SystemExit("取帧失败")

    # 预热 10 次
    for f in frames[:10]:
        infer(sess, in_name, out_name, f)

    times = []
    for f in frames[10:10 + runs]:
        t0 = time.perf_counter()
        infer(sess, in_name, out_name, f)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  样本数   : {len(times)}")
    print(f"  平均     : {times.mean():.1f} ms")
    print(f"  中位数   : {np.median(times):.1f} ms")
    print(f"  P95      : {np.percentile(times, 95):.1f} ms")
    print(f"  最小/最大: {times.min():.1f} / {times.max():.1f} ms")
    print(f"  理论吞吐 : {1000 / times.mean():.1f} FPS")
    return times.mean()


def test_accuracy(sess, in_name, out_name, dataset_dir):
    print(f"\n== 准确率（{dataset_dir}）==")
    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    if not classes:
        raise SystemExit("数据集为空，应包含按类别命名的子目录")

    name_to_idx = {n.lower(): i for i, n in enumerate(INGREDIENTS)}
    total = correct = 0
    top2 = 0
    per_class = defaultdict(lambda: [0, 0])
    confusions = defaultdict(int)

    t_start = time.time()
    for cname in classes:
        idx = name_to_idx.get(cname.lower())
        if idx is None:
            print(f"  跳过未知类别目录: {cname}")
            continue
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(dataset_dir, cname, ext))
        if not files:
            continue

        for f in sorted(files):
            img = cv2.imread(f)
            if img is None:
                continue
            probs = infer(sess, in_name, out_name, img)
            order = np.argsort(probs)[::-1]
            pred, conf = int(order[0]), float(probs[order[0]])
            total += 1
            per_class[cname][1] += 1
            if pred == idx:
                correct += 1
                per_class[cname][0] += 1
            else:
                confusions[f"{cname} -> {INGREDIENTS[pred]}"] += 1
            if idx in order[:2]:
                top2 += 1

    dt = time.time() - t_start
    if not total:
        raise SystemExit("没有可用图片")

    print(f"  类别数   : {len(classes)}")
    print(f"  图片数   : {total}")
    print(f"  Top-1    : {correct}/{total} = {correct / total:.1%}")
    print(f"  Top-2    : {top2}/{total} = {top2 / total:.1%}")
    print(f"  总耗时   : {dt:.1f} s（{dt / total * 1000:.0f} ms/张）")

    print("\n  各类准确率：")
    for c in classes:
        ok, n = per_class[c]
        if n:
            print(f"    {c:14s} {ok:3d}/{n:3d}  {ok / n:6.1%}")

    if confusions:
        print("\n  主要混淆：")
        for k, v in sorted(confusions.items(), key=lambda x: -x[1])[:8]:
            print(f"    {k}: {v} 次")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./model/best6.onnx")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--acc", metavar="DIR")
    ap.add_argument("--cam", type=int, default=20)
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"模型不存在: {args.model}")

    print(f"模型: {args.model}")
    sess, in_name, out_name = build_session(args.model)
    print(f"输入: {sess.get_inputs()[0].shape} | 输出: {sess.get_outputs()[0].shape}")

    if args.latency:
        test_latency(sess, in_name, out_name, args.cam, args.runs)
    if args.acc:
        test_accuracy(sess, in_name, out_name, args.acc)
    if not (args.latency or args.acc):
        print("\n请加 --latency 或 --acc <目录>")


if __name__ == "__main__":
    main()
