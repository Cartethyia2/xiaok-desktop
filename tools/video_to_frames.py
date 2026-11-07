# -*- coding: utf-8 -*-
"""
视频转帧工具（视频管线的核心）
==============================
把"绿幕视频"变成小K 能播的透明帧序列。

用法：
    python tools/video_to_frames.py <视频文件> <动作名> [--fps 24] [--sim 0.28]

例子：
    python tools/video_to_frames.py inbox/xiaok_wave.mp4 wave

流水线（每一道工序都是之前踩坑趟出来的）：
1. ffmpeg chromakey 抠绿 + despill 去绿溢出
2. 全部帧求"全局内容包围盒"——统一裁剪，视频不抖
3. 预乘 alpha 缩放到显示尺寸（防边缘混色）
4. HSV 色相旋转去青绿（绿×蓝紫头发的混合色，之前的终极武器）
5. 底对齐放到 213×240 画布，写入 assets/vid_<动作名>/
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# 允许从项目根目录和 tools 目录两种位置运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DISPLAY_W, DISPLAY_H = 213, 240
CONTENT_H = 223  # 内容高度（和精灵表角色一致：268/288*240）


def hue_despill(arr: np.ndarray) -> np.ndarray:
    """HSV 色相旋转去绿/青（75°~200° → 215°蓝），饱和度打 75 折。"""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    rgb = arr[:, :, :3] / 255.0
    alpha = arr[:, :, 3]
    hsv = rgb_to_hsv(rgb)
    hue, sat = hsv[:, :, 0] * 360, hsv[:, :, 1]
    hit = (hue > 75) & (hue < 200) & (sat > 0.05) & (alpha > 0)
    hsv[:, :, 0] = np.where(hit, 215.0 / 360.0, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(hit, sat * 0.75, hsv[:, :, 1])
    out = arr.copy()
    out[:, :, :3] = hsv_to_rgb(hsv) * 255.0
    return out


def resize_premultiplied(img: Image.Image, size) -> Image.Image:
    """先预乘 alpha 再缩放，透明区颜色不会混进边缘。"""
    x = np.array(img).astype(np.float32)
    a = x[:, :, 3:4] / 255.0
    x[:, :, :3] *= a
    pre = Image.fromarray(x.astype(np.uint8), "RGBA").resize(size, Image.LANCZOS)
    o = np.array(pre).astype(np.float32)
    a2 = o[:, :, 3:4] / 255.0
    nz = a2[:, :, 0] > 1e-3
    o[:, :, :3][nz] /= a2[nz]
    o[:, :, :3] = np.clip(o[:, :, :3], 0, 255)
    return Image.fromarray(o.astype(np.uint8), "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="输入视频（绿幕）")
    ap.add_argument("name", help="动作名（输出到 assets/vid_<动作名>/）")
    ap.add_argument("--fps", type=int, default=24, help="抽帧帧率，默认 24")
    ap.add_argument("--key", default="0x00FF00", help="绿幕颜色，默认 0x00FF00")
    ap.add_argument("--sim", type=float, default=0.28, help="抠绿容差 0~1，默认 0.28")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"找不到视频：{video}")

    out_dir = ROOT / "assets" / f"vid_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第1步：ffmpeg 抠绿 + despill，抽出 RGBA 帧 ----
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%05d.png")
        cmd = [
            FFMPEG, "-y", "-i", str(video),
            "-vf", f"fps={args.fps},chromakey={args.key}:{args.sim}:0.05,despill=type=green,format=rgba",
            pattern,
        ]
        print(">> ffmpeg 抠绿抽帧中…")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit("ffmpeg 失败")

        files = sorted(Path(tmp).glob("f_*.png"))
        if not files:
            sys.exit("没有抽出任何帧，检查视频文件")
        print(f">> 抽出 {len(files)} 帧，开始统一裁剪与净化…")

        # ---- 第2步：全局内容包围盒（所有帧一起算，画面不抖）----
        frames = [np.array(Image.open(f).convert("RGBA")) for f in files]
        x0 = y0 = 10**9
        x1 = y1 = -1
        for arr in frames:
            ys, xs = (arr[:, :, 3] > 10).nonzero()
            if len(ys) == 0:
                continue
            x0, y0 = min(x0, xs.min()), min(y0, ys.min())
            x1, y1 = max(x1, xs.max() + 1), max(y1, ys.max() + 1)
        if x1 < 0:
            sys.exit("抠绿后啥都没剩——试试调大 --sim（比如 0.35）或检查绿幕颜色")
        bw, bh = x1 - x0, y1 - y0
        print(f">> 内容包围盒 {bw}x{bh}")

        # ---- 第3步：逐帧裁剪、缩放、净化、底对齐落版 ----
        scale = min(CONTENT_H / bh, (DISPLAY_W - 16) / bw)
        nw, nh = max(1, round(bw * scale)), max(1, round(bh * scale))
        total_despill = 0
        for i, arr in enumerate(frames):
            cell = Image.fromarray(arr[y0:y1, x0:x1], "RGBA")
            cell = resize_premultiplied(cell, (nw, nh))
            a = hue_despill(np.array(cell))
            total_despill += 1
            canvas = Image.new("RGBA", (DISPLAY_W, DISPLAY_H), (0, 0, 0, 0))
            canvas.paste(
                Image.fromarray(a, "RGBA"),
                ((DISPLAY_W - nw) // 2, DISPLAY_H - nh),
                Image.fromarray(a, "RGBA"),
            )
            canvas.save(out_dir / f"frame_{i + 1:04d}.png")

    print(f">> 完成！{len(frames)} 帧已写入 {out_dir}")
    print(f">> 在 main.py 的 IDLE_PROGRAMS 里加上 \"{args.name}\" 即可让她随机表演")


if __name__ == "__main__":
    main()
