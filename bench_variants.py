"""对比不同预处理方式对分类准确率的影响。

用于判断：模型准确率低，是因为"看不清"，还是因为"画面构成与训练时不一致"。

用法：
    python3 bench_variants.py --model ./model/best6.onnx --data ./dataset
"""

import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = (320, 320)
INGREDIENTS = [
    "Bitter melon", "Carrot", "Green apple", "Lemon", "Pomegranate", "Tomato",
    "apple", "eggplant", "orange", "peach", "pear", "pumpkin", "watermelon",
]


def make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


# ---------------- 各种预处理方案 ----------------

def v_raw(img):
    """原图直接缩放（主程序当前做法）"""
    return img


def v_center(img, ratio=0.7):
    """中心裁剪 70%"""
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def v_letterbox(img, scale=0.5, color=(255, 255, 255)):
    """把图缩小后贴在白底中央（模拟"白底 + 小物体"的训练分布）"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    y0 = (h - small.shape[0]) // 2
    x0 = (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def v_letterbox_gray(img, scale=0.5):
    """缩小 + 灰底"""
    return v_letterbox(img, scale, (128, 128, 128))


def v_wb(img):
    """白平衡校正（以浅色区为参考）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    board = (s < 40) & (v > 150)
    if board.mean() < 0.03:
        return img
    means = [float(img[:, :, c][board].mean()) for c in range(3)]
    target = sum(means) / 3.0
    out = img.astype(np.float32)
    for c in range(3):
        if means[c] > 1:
            out[:, :, c] *= target / means[c]
    return np.clip(out, 0, 255).astype(np.uint8)


VARIANTS = {
    "原图": v_raw,
    "裁剪0.9": lambda im: v_center(im, 0.9),
    "裁剪0.8": lambda im: v_center(im, 0.8),
    "裁剪0.7": lambda im: v_center(im, 0.7),
    "裁剪0.6": lambda im: v_center(im, 0.6),
    "裁剪0.5": lambda im: v_center(im, 0.5),
    "裁剪0.4": lambda im: v_center(im, 0.4),
    "白底缩小70%": lambda im: v_letterbox(im, 0.7),
}


def predict(sess, img):
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, INPUT_SIZE, swapRB=True)
    out = sess.run([out_name], {in_name: blob})[0]
    p = out[0] if out.ndim == 2 else out.flatten()
    return int(np.argmax(p)), float(np.max(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./model/best6.onnx")
    ap.add_argument("--data", default="./dataset")
    args = ap.parse_args()

    sess = make_session(args.model)
    name_to_idx = {n.lower(): i for i, n in enumerate(INGREDIENTS)}

    samples = []
    for cname in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, cname)
        if not os.path.isdir(d):
            continue
        idx = name_to_idx.get(cname.lower())
        if idx is None:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in glob.glob(os.path.join(d, ext)):
                img = cv2.imread(f)
                if img is not None:
                    samples.append((img, idx, cname))

    if not samples:
        raise SystemExit("没有样本")
    print(f"样本 {len(samples)} 张，类别 {len({s[2] for s in samples})} 个\n")

    print(f"{'预处理':16s} {'Top-1':>12s}  各类正确数")
    print("-" * 60)
    best = None
    for vname, fn in VARIANTS.items():
        ok = 0
        per = {}
        for img, idx, cname in samples:
            try:
                proc = fn(img)
            except Exception:
                proc = img
            pred, _ = predict(sess, proc)
            hit = (pred == idx)
            ok += hit
            per[cname] = per.get(cname, [0, 0])
            per[cname][1] += 1
            per[cname][0] += hit
        rate = ok / len(samples)
        detail = " ".join(f"{c}:{v[0]}/{v[1]}" for c, v in sorted(per.items()))
        print(f"{vname:16s} {ok:3d}/{len(samples)} {rate:6.1%}  {detail}")
        if best is None or rate > best[1]:
            best = (vname, rate)

    print(f"\n最佳预处理: {best[0]}（{best[1]:.1%}）")


if __name__ == "__main__":
    main()
