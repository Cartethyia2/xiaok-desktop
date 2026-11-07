# -*- coding: utf-8 -*-
"""
程序化动画生成器 v2：呼吸 + 眨眼
=================================
v2 修复：
- 呼吸改用亚像素仿射变换（双三次插值 + 预乘 alpha），
  偏移量是小数（如 0.73px），不再量化成台阶，告别"抽搐弹"
- 呼吸叠加轻微纵向缩放（±0.6%），有"身体起伏"感
- 眨眼横条收窄到脸部区域（X0~X1），四周羽化 6px，
  头发丝不再被卷入压缩，接缝隐形

输出：assets/vid_breathe/（96 帧 + meta.json）
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "assets" / "spritesheet_display.png"
OUT = ROOT / "assets" / "vid_breathe"

FPS = 24
LOOP_SECONDS = 4
N_FRAMES = FPS * LOOP_SECONDS  # 96

# 呼吸参数
BOB_AMPLITUDE = 2.5      # 垂直浮沉（像素，亚像素）
SCALE_AMPLITUDE = 0.006  # 纵向起伏缩放（±0.6%）

# 眨眼参数（213×240 显示帧坐标）
EYE_TOP, EYE_BOTTOM = 88, 120   # 眼睛横条纵向范围
EYE_X0, EYE_X1 = 40, 173        # 只压脸部区域，不碰两侧头发
FEATHER = 6                     # 羽化宽度（像素）
BLINK_AT = 53
BLINK_SQUASH = [1.0, 0.45, 0.12, 0.45, 1.0]  # 睁->闭->睁

W, H = 213, 240


def premult_resize(arr: np.ndarray, size) -> np.ndarray:
    """预乘 alpha 缩放（numpy in/out）。"""
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    x = np.array(img).astype(np.float32)
    a = x[:, :, 3:4] / 255.0
    x[:, :, :3] *= a
    pre = Image.fromarray(x.astype(np.uint8), "RGBA").resize(size, Image.LANCZOS)
    o = np.array(pre).astype(np.float32)
    a2 = o[:, :, 3:4] / 255.0
    nz = a2[:, :, 0] > 1e-3
    o[:, :, :3][nz] /= a2[nz]
    o[:, :, :3] = np.clip(o[:, :, :3], 0, 255)
    return o


def breathe_transform(idle: Image.Image, ty: float, sy: float) -> np.ndarray:
    """亚像素仿射：垂直平移 ty + 以脚底为锚的纵向缩放 sy。"""
    x = np.array(idle).astype(np.float32)
    a = x[:, :, 3:4] / 255.0
    x[:, :, :3] *= a  # 预乘
    pre = Image.fromarray(x.astype(np.uint8), "RGBA")
    # PIL AFFINE: in_x = out_x - tx；in_y = (out_y - H)/sy + H - ty
    e = 1.0 / sy
    f = H - H * e - ty
    out = pre.transform(
        (W, H), Image.AFFINE, (1, 0, 0, 0, e, f),
        resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0),
    )
    o = np.array(out).astype(np.float32)
    a2 = o[:, :, 3:4] / 255.0
    nz = a2[:, :, 0] > 1e-3
    o[:, :, :3][nz] /= a2[nz]
    o[:, :, :3] = np.clip(o[:, :, :3], 0, 255)
    return o


def make_feather_mask() -> np.ndarray:
    """脸部矩形 + 四边羽化的合成蒙版（0~1）。"""
    mask = np.zeros((H, W), dtype=np.float32)
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    dy = np.minimum(ys - EYE_TOP, EYE_BOTTOM - ys)
    dx = np.minimum(xs - EYE_X0, EYE_X1 - xs)
    d = np.minimum(dy, dx).astype(np.float32)
    mask = np.clip(d / FEATHER, 0.0, 1.0)
    outside = (ys < EYE_TOP) | (ys > EYE_BOTTOM) | (xs < EYE_X0) | (xs > EYE_X1)
    mask[outside] = 0.0
    return mask[:, :, None]


def apply_blink(idle_arr: np.ndarray, sq: float, mask: np.ndarray) -> np.ndarray:
    """只压脸部横条，羽化合成，头发丝纹丝不动。"""
    band = idle_arr[EYE_TOP:EYE_BOTTOM, EYE_X0:EYE_X1]
    band_h, band_w = band.shape[:2]
    new_h = max(1, round(band_h * sq))
    scaled = premult_resize(band, (band_w, new_h))
    blinked = idle_arr.copy()
    region = np.empty_like(band)
    vac = band_h - new_h
    region[:vac] = band[0:1]      # 带顶行（额头）填充腾出区
    region[vac:] = scaled          # 压缩带贴底部（上眼皮压下）
    blinked[EYE_TOP:EYE_BOTTOM, EYE_X0:EYE_X1] = region
    return idle_arr * (1 - mask) + blinked * mask


def main():
    sheet = Image.open(SHEET).convert("RGBA")
    idle = sheet.crop((0, 0, W, H))
    idle_arr = np.array(idle).astype(np.float32)
    mask = make_feather_mask()

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("frame_*.png"):
        old.unlink()

    blink_map = {}
    for k, sq in enumerate(BLINK_SQUASH):
        if sq != 1.0:
            blink_map[BLINK_AT + k] = sq

    for i in range(N_FRAMES):
        t = i / N_FRAMES
        phase = np.sin(2 * np.pi * t)
        ty = BOB_AMPLITUDE * phase
        sy = 1.0 + SCALE_AMPLITUDE * phase
        frame = breathe_transform(idle, ty, sy)
        if i in blink_map:
            frame = apply_blink(frame, blink_map[i], mask)
        Image.fromarray(frame.astype(np.uint8), "RGBA").save(
            OUT / f"frame_{i + 1:04d}.png"
        )

    (OUT / "meta.json").write_text(
        json.dumps({"fps": FPS, "loop": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"生成 {N_FRAMES} 帧 -> {OUT}")


if __name__ == "__main__":
    main()
