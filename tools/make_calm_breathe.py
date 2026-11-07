# -*- coding: utf-8 -*-
"""
v3 呼吸+眨眼生成器：AI 重绘基座 + 交叉淡化眨眼
================================================
素材：generated/calm_open.png（闭嘴睁眼）、generated/calm_closed.png（闭嘴闭眼）

处理链（全部是之前趟平的坑的复用）：
1. 涂掉左下角"AI生成"水印 -> 绿幕抠像 -> HSV 去青绿
2. 归一化到底对齐 213×240（和精灵表角色同尺寸）
3. 睁眼/闭眼两张图做亚像素自动对齐（MSE 搜索）
4. 眨眼 = 两张图交叉淡化（眼皮真正合上，不再是压缩）
5. 呼吸 = 亚像素仿射浮沉（同 v2）
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC_OPEN = ROOT / "generated" / "calm_open.png"
SRC_CLOSED = ROOT / "generated" / "calm_closed.png"
OUT = ROOT / "assets" / "vid_breathe"

W, H = 213, 240
CONTENT_H = 223
FPS, LOOP_SECONDS = 24, 4
N_FRAMES = FPS * LOOP_SECONDS

BOB_AMPLITUDE = 2.5
SCALE_AMPLITUDE = 0.006
BLINK_AT = 53
# 眨眼序列：("wipe", 覆盖率) = 眼皮从上扫到 cover 处；("full",) = 全闭
# 24fps 下共 4 帧 ≈ 167ms：快闭 -> 保持 -> 稍慢睁开
BLINK_SEQ = [("wipe", 0.62), ("full",), ("full",), ("wipe", 0.38)]


def key_and_clean(path: Path) -> np.ndarray:
    """去水印 -> 抠绿 -> HSV去青绿 -> 边缘羽化，返回 float32 RGBA。"""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    im = Image.open(path).convert("RGB")
    a = np.array(im).astype(np.float32)
    h, w = a.shape[:2]
    # 1) 水印区涂纯绿（左下角，覆盖 1024 宽图上的 "AI生成"）
    a[h - 70:, :260] = [0, 255, 0]
    # 2) 绿幕抠像：到纯绿的距离 -> alpha（带平滑过渡）
    dist = np.sqrt(((a - np.array([0, 255, 0])) ** 2).sum(axis=2))
    alpha = np.clip((dist - 90) / 60.0, 0, 1)
    out = np.dstack([a, alpha * 255.0]).astype(np.float32)
    # 3) HSV 去青绿 spill
    rgb = out[:, :, :3] / 255.0
    hsv = rgb_to_hsv(rgb)
    hue, sat = hsv[:, :, 0] * 360, hsv[:, :, 1]
    hit = (hue > 75) & (hue < 200) & (sat > 0.05) & (alpha > 0)
    hsv[:, :, 0] = np.where(hit, 215.0 / 360.0, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(hit, sat * 0.75, hsv[:, :, 1])
    out[:, :, :3] = hsv_to_rgb(hsv) * 255.0
    return out


def normalize(arr: np.ndarray) -> np.ndarray:
    """裁内容 -> 预乘缩放到内容高 223 -> 底对齐放 213×240。"""
    ys, xs = (arr[:, :, 3] > 10).nonzero()
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    content = arr[y0:y1, x0:x1]
    bh, bw = content.shape[:2]
    scale = min(CONTENT_H / bh, (W - 16) / bw)
    nw, nh = max(1, round(bw * scale)), max(1, round(bh * scale))
    # 预乘缩放
    a = content[:, :, 3:4] / 255.0
    pre = content.copy()
    pre[:, :, :3] *= a
    img = Image.fromarray(pre.astype(np.uint8), "RGBA").resize((nw, nh), Image.LANCZOS)
    o = np.array(img).astype(np.float32)
    a2 = o[:, :, 3:4] / 255.0
    nz = a2[:, :, 0] > 1e-3
    o[:, :, :3][nz] /= a2[nz]
    o[:, :, :3] = np.clip(o[:, :, :3], 0, 255)
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    px, py = (W - nw) // 2, H - nh
    canvas[py:py + nh, px:px + nw] = o
    return canvas


def align(base: np.ndarray, other: np.ndarray, max_shift=8):
    """用 MSE 搜索把 other 平移对齐到 base，返回最佳 (dx, dy)。"""
    mask = base[:, :, 3] > 128
    best, best_off = float("inf"), (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(other, dy, axis=0), dx, axis=1)
            diff = (shifted[:, :, :3] - base[:, :, :3])[mask]
            mse = float((diff ** 2).mean())
            if mse < best:
                best, best_off = mse, (dx, dy)
    return best_off


def shift(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


def build_eye_mask(open_arr: np.ndarray, closed_arr: np.ndarray):
    """找两张图差异最大的区域（眼睛），返回羽化后的矩形蒙版与包围盒。

    只在上半身的横带里搜，避开头发丝的重绘噪声。
    """
    diff = np.abs(open_arr[:, :, :3] - closed_arr[:, :, :3]).sum(axis=2)
    solid = np.minimum(open_arr[:, :, 3], closed_arr[:, :, 3]) > 128
    # 只在眼睛高度的横带里搜（避开刘海和两侧头发的重绘噪声）
    band = np.zeros((H, W), bool)
    band[int(H * 0.25):int(H * 0.56)] = True
    strong = (diff > 400) & solid & band
    # 4-邻域腐蚀：眼睛是大块实心区能活下来，1-2px 的头发丝噪点被削掉
    strong = (strong
              & np.roll(strong, -1, 0) & np.roll(strong, 1, 0)
              & np.roll(strong, -1, 1) & np.roll(strong, 1, 1))
    ys, xs = strong.nonzero()
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1.0
    mi = Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(4))
    mask = np.array(mi).astype(np.float32) / 255.0
    return mask, (x0, y0, x1, y1)


def lid_profile(box, cover: float) -> np.ndarray:
    """眼皮扫掠权重：眼睛包围盒内，顶部 cover 比例已闭合，带 0.25 的过渡斜坡。"""
    x0, y0, x1, y1 = box
    prof = np.zeros((H, W), np.float32)
    yy = (np.arange(y0, y1, dtype=np.float32) - y0) / max(1, y1 - y0 - 1)
    w = np.clip((cover - yy) / 0.25 + 0.5, 0, 1)
    prof[y0:y1, x0:x1] = w[:, None]
    return prof


def breathe_transform(arr: np.ndarray, ty: float, sy: float) -> np.ndarray:
    """亚像素仿射浮沉（预乘 + 双三次）。"""
    x = arr.copy()
    a = x[:, :, 3:4] / 255.0
    x[:, :, :3] *= a
    pre = Image.fromarray(x.astype(np.uint8), "RGBA")
    e = 1.0 / sy
    f = H - H * e - ty
    out = pre.transform((W, H), Image.AFFINE, (1, 0, 0, 0, e, f),
                        resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    o = np.array(out).astype(np.float32)
    a2 = o[:, :, 3:4] / 255.0
    nz = a2[:, :, 0] > 1e-3
    o[:, :, :3][nz] /= a2[nz]
    o[:, :, :3] = np.clip(o[:, :, :3], 0, 255)
    return o


def main():
    print(">> 抠像与净化…")
    open_arr = normalize(key_and_clean(SRC_OPEN))
    closed_arr = normalize(key_and_clean(SRC_CLOSED))
    print(">> 自动对齐闭眼帧…")
    dx, dy = align(open_arr, closed_arr)
    print(f"   对齐偏移: dx={dx}, dy={dy}")
    closed_arr = shift(closed_arr, dx, dy)

    print(">> 构建眨眼蒙版…")
    eye_mask, eye_box = build_eye_mask(open_arr, closed_arr)
    print(f"   眼睛区域: {eye_box}")
    # 调试图：蒙版范围叠到睁眼图上
    dbg = open_arr[:, :, :3].copy()
    dbg[:, :, 0] = np.clip(dbg[:, :, 0] + eye_mask * 90, 0, 255)
    dbg_a = np.dstack([dbg, open_arr[:, :, 3]]).astype(np.uint8)
    Image.fromarray(dbg_a, "RGBA").save(ROOT / "check_v3_mask.png")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("frame_*.png"):
        old.unlink()

    blink_map = {BLINK_AT + k: v for k, v in enumerate(BLINK_SEQ)}
    for i in range(N_FRAMES):
        t = i / N_FRAMES
        phase = np.sin(2 * np.pi * t)
        base = open_arr
        if i in blink_map:  # 蒙版内眨眼：脸和头发纹丝不动
            spec = blink_map[i]
            if spec[0] == "full":
                b = eye_mask
            else:
                b = eye_mask * lid_profile(eye_box, spec[1])
            b = b[:, :, None]
            base = open_arr * (1 - b) + closed_arr * b
        frame = breathe_transform(base, BOB_AMPLITUDE * phase,
                                  1.0 + SCALE_AMPLITUDE * phase)
        Image.fromarray(frame.astype(np.uint8), "RGBA").save(
            OUT / f"frame_{i + 1:04d}.png")

    (OUT / "meta.json").write_text(
        json.dumps({"fps": FPS, "loop": True}, ensure_ascii=False),
        encoding="utf-8")
    # 同时导出基座图供以后复用
    Image.fromarray(open_arr.astype(np.uint8), "RGBA").save(ROOT / "assets" / "idle_calm.png")
    print(f">> 完成：{N_FRAMES} 帧 -> {OUT}")


if __name__ == "__main__":
    main()
