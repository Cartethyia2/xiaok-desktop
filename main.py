# -*- coding: utf-8 -*-
"""
小K 桌宠 · 独立版 v0.3（节奏 + 弹簧手感）
==========================================
相比 v0.2 的改进：
1. 逐帧停留表（DWELL_MS）：每一帧可以停留不同时间，
   打哈欠从"抽一下"变成"真的打个哈欠"（约 3 秒）
2. 摸头用弹簧-阻尼物理模型：x'' = -k(x-1) - c·x'，60fps 实时积分。
   按到最深处松手，快速弹回 + 一次轻微回伸，脆弹不木
3. 素材离线预处理（归一化 + 去绿 + 预缩放），运行时零缩放

新概念：
- 弹簧-阻尼系统：刚度 k 决定弹回速度，阻尼比 ζ 决定"弹几下"
  ζ<1 欠阻尼（Q弹）、ζ=1 临界（木）、ζ>1 过阻尼（绵）
- 半隐式欧拉积分：游戏物理的标配数值方法
"""
import sys
import math
import random
from pathlib import Path

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent, QAction
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMenu

ASSETS = Path(__file__).parent / "assets"

# 显示版精灵表：离线已缩放到显示尺寸（213×240/帧）、去绿、归一化
# 运行时零缩放——Qt 缩放会把透明区颜色混进边缘（绿边的最后一处来源）
FRAME_W, FRAME_H = 213, 240

# 状态表：名字 -> 配置
#   row   精灵表第几行
#   cells 用这一行的哪几格（摸头跳过 1、2 两格"缺头发"的压扁帧）
#   fps   默认帧率（没有 dwell 时生效）
#   loop  是否循环
#   dwell 可选：每一帧停留的毫秒数（节奏的灵魂！）
STATES = {
    "idle": {
        "row": 0, "cells": [0], "fps": 12, "loop": True,
    },
    "flute": {
        "row": 6, "cells": list(range(8)), "fps": 10, "loop": False,
        # 举笛 -> 吹奏（主体，放慢）-> 收笛，一场 4 秒的小演出
        "dwell": [600, 500, 500, 600, 500, 600, 500, 600],
    },
    "yawn": {
        "row": 7, "cells": list(range(7)), "fps": 12, "loop": False,
        # 慢慢低头 -> 张大嘴停住 -> 缓缓回神，总共约 3 秒
        "dwell": [350, 250, 350, 700, 700, 400, 300],
    },
    "head_pat": {
        "row": 8, "cells": [0, 3, 4], "fps": 10, "loop": False,
        # 正常 -> 眯眼笑（停久一点）-> 睁开
        "dwell": [100, 450, 200],
    },
    "chair": {
        "row": 9, "cells": list(range(5)), "fps": 10, "loop": False,
        # 搬凳子 -> 坐下 -> 坐一会儿 -> 起身
        "dwell": [700, 500, 2800, 2600, 600],
    },
}
IDLE_PROGRAMS = ["flute", "yawn", "chair"]

# ---- 视频动作（video_to_frames.py 生成，每帧已是 213×240 显示尺寸）----
# 扫描 assets/vid_* 目录自动注册，fps 默认 24
VIDEO_FPS = 24

# 弹簧参数（弹簧-阻尼物理模型，手感是"脆弹"）
# 物理模型：x'' = -k(x-1) - c·x'
#   k 刚度：越大弹回越快
#   ζ 阻尼比：<1 欠阻尼（会回伸一点），=1 临界阻尼（最木）
#   取 ζ=0.7：一次 ~4% 的轻微回伸，是"弹"的灵魂，但不会振荡出第二下
SPRING_PRESS = 0.10       # 按下深度 10%（纯纵向）
SPRING_STIFFNESS = 250.0  # k 刚度
SPRING_ZETA = 0.45        # ζ 阻尼比：回伸 1.8%（弹的灵魂），二次下压 0.3%（看不见）


class PetWindow(QWidget):
    """小K 的窗口本体：透明、无边框、置顶，会播动画、会弹。"""

    def __init__(self):
        super().__init__()

        # ---- 窗口外观 ----
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ---- 切图（显示版精灵表，帧已是显示尺寸，无需缩放）----
        sheet = QPixmap(str(ASSETS / "spritesheet_display.png"))
        self.frames = {}  # 状态名 -> [帧图, ...]
        for name, cfg in STATES.items():
            row = cfg["row"]
            state_frames = []
            for col in cfg["cells"]:
                frame = sheet.copy(
                    col * FRAME_W, row * FRAME_H, FRAME_W, FRAME_H
                )
                state_frames.append(frame)
            self.frames[name] = state_frames

        # ---- 视频动作：扫描 assets/vid_* 目录自动注册 ----
        import json
        for vid_dir in sorted(ASSETS.glob("vid_*")):
            if not vid_dir.is_dir():
                continue
            name = vid_dir.name[4:]  # 去掉 vid_ 前缀
            files = sorted(vid_dir.glob("frame_*.png"))
            if not files:
                continue
            self.frames[name] = [QPixmap(str(f)) for f in files]
            # meta.json 可声明 fps 和 loop（呼吸就是 loop）
            meta_file = vid_dir / "meta.json"
            meta = {}
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            STATES[name] = {
                "row": -1, "cells": [],
                "fps": meta.get("fps", VIDEO_FPS),
                "loop": meta.get("loop", False),
            }
            print(f"视频动作已注册: {name}（{len(files)} 帧"
                  f"{' 循环' if STATES[name]['loop'] else ''}）")

        # ---- 待机状态：有呼吸循环就用呼吸，否则静态帧 ----
        self.IDLE_STATE = "breathe" if "breathe" in self.frames else "idle"

        # ---- 显示层：固定大小、底对齐（弹簧压缩时从脚下压）----
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        first = self.frames["idle"][0]
        self.label.resize(first.size())
        self.resize(first.size())

        # ---- 动画状态 ----
        self.state = self.IDLE_STATE
        self.frame_index = 0

        # ---- 弹簧状态（弹簧-阻尼系统的位移与速度）----
        self.squash = 1.0       # 纵向缩放（1 = 正常）
        self.squash_v = 0.0     # 变化速度
        self.spring_on = False

        # ---- 定时器1：逐帧播放 ----
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.advance_frame)
        self.play(self.IDLE_STATE)

        # ---- 定时器2：随机小节目（20~40 秒一次）----
        self.program_timer = QTimer(self)
        self.program_timer.setSingleShot(True)
        self.program_timer.timeout.connect(self.play_random_program)
        self.schedule_program()

        # ---- 定时器3：弹簧物理（60fps）----
        self.spring_timer = QTimer(self)
        self.spring_timer.setInterval(16)
        self.spring_timer.timeout.connect(self.update_spring)

        # ---- 拖动状态 ----
        self.drag_position = QPoint()
        self.moved = False

        # 初始位置：屏幕右下角
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            screen.right() - self.width() - 40,
            screen.bottom() - self.height() - 20,
        )

    # ================= 动画核心 =================

    def play(self, state: str):
        """切换到某个状态，从第 0 帧开始播。"""
        self.state = state
        self.frame_index = 0
        self.arm_frame_timer()
        self.render()

    def arm_frame_timer(self):
        """按当前帧的停留时间上发条。"""
        cfg = STATES[self.state]
        dwell = cfg.get("dwell")
        if dwell:
            interval = dwell[self.frame_index]
        else:
            interval = round(1000 / cfg["fps"])
        self.anim_timer.start(interval)

    def advance_frame(self):
        """定时器滴答：前进一帧。"""
        cfg = STATES[self.state]
        total = len(self.frames[self.state])
        self.frame_index += 1
        if self.frame_index >= total:
            if cfg["loop"]:
                self.frame_index = 0
            else:
                self.play(self.IDLE_STATE)  # 小节目播完回待机
                return
        self.arm_frame_timer()
        self.render()

    def render(self):
        """画出当前帧（带弹簧纵向压缩）。"""
        pm = self.frames[self.state][self.frame_index]
        if self.squash != 1.0:
            # 用"预乘 alpha"格式缩放：否则 Qt 插值会把透明像素的颜色
            # 混进边缘（绿边的最后 36 个像素就是这么来的）
            img = pm.toImage().convertToFormat(
                QImage.Format_ARGB32_Premultiplied
            )
            img = img.scaled(
                img.width(),
                max(1, round(img.height() * self.squash)),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
            pm = QPixmap.fromImage(img)
        self.label.setPixmap(pm)

    # ================= 弹簧（点按的"弹弹"手感）=================

    def start_spring(self):
        """点按：把弹簧"按"到最深处，松手让它自己弹回来。"""
        self.squash = 1 - SPRING_PRESS
        self.squash_v = 0.0
        self.spring_on = True
        self.render()
        self.spring_timer.start()

    def update_spring(self):
        """60fps 积分弹簧-阻尼方程：x'' = -k(x-1) - c·x'"""
        dt = 0.016
        k = SPRING_STIFFNESS
        c = 2 * SPRING_ZETA * math.sqrt(k)  # 由阻尼比算阻尼系数
        # 半隐式欧拉积分（先速度后位移，数值稳定）
        self.squash_v += (-k * (self.squash - 1.0) - c * self.squash_v) * dt
        self.squash += self.squash_v * dt
        # 足够接近静止就收工，避免永远抖小数点
        if abs(self.squash - 1.0) < 0.002 and abs(self.squash_v) < 0.02:
            self.spring_timer.stop()
            self.spring_on = False
            self.squash = 1.0
            self.squash_v = 0.0
        self.render()

    # ================= 随机小节目 =================

    def schedule_program(self):
        self.program_timer.start(random.randint(20_000, 40_000))

    def play_random_program(self):
        if self.state == self.IDLE_STATE:
            self.play(random.choice(IDLE_PROGRAMS))
        self.schedule_program()

    # ================= 鼠标交互 =================

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_position = (
                event.globalPosition().toPoint() - self.pos()
            )
            self.moved = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self.move(
                event.globalPosition().toPoint() - self.drag_position
            )
            self.moved = True

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and not self.moved:
            self.play("head_pat")   # 摸头动画
            self.start_spring()     # 同时弹一下
        elif event.button() == Qt.RightButton:
            self.show_menu(event.globalPosition().toPoint())

    # ================= 键盘交互 =================

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()

    # ================= 右键菜单 =================

    def show_menu(self, global_pos: QPoint):
        menu = QMenu(self)
        # 视频动作手动点播（测试用）
        video_states = [n for n, c in STATES.items() if c.get("row") == -1]
        if video_states:
            menu.addSection("点播")
            for name in video_states:
                act = QAction(name, self)
                act.triggered.connect(
                    lambda _checked=False, n=name: self.play(n)
                )
                menu.addAction(act)
            menu.addSeparator()
        action_quit = QAction("退出小K", self)
        action_quit.triggered.connect(self.close)
        menu.addAction(action_quit)
        menu.exec(global_pos)


def main():
    app = QApplication(sys.argv)

    # ---- 单实例 + 双击即重启 ----
    # 用户心智：双击 = 打开最新版小K。
    # 已有一只时：发"让位"消息请旧版退出，等她释放占位符后再启动，
    # 这样素材/代码更新后双击即生效，也不会出现一堆小K叠罗汉。
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    import time

    NAME = "xiaok-pet-v1"
    probe = QLocalSocket()
    probe.connectToServer(NAME)
    if probe.waitForConnected(300):
        # 旧版在：请她让位
        probe.write(b"restart")
        probe.flush()
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        # 等她退出（最多 3 秒；若有多只旧版，每轮都再请一次）
        for _ in range(30):
            time.sleep(0.1)
            again = QLocalSocket()
            again.connectToServer(NAME)
            if not again.waitForConnected(150):
                break  # 占位符已释放
            again.write(b"restart")
            again.flush()
            again.waitForBytesWritten(150)
            again.disconnectFromServer()

    pet = PetWindow()

    server = QLocalServer()
    if not server.listen(NAME):
        # 占位符没释放干净（进程被强杀等），清掉重试
        QLocalServer.removeServer(NAME)
        if not server.listen(NAME):
            server = None  # 实在占不上就裸奔，先保证能打开

    def on_call():
        if server is None:
            return
        conn = server.nextPendingConnection()
        if not conn:
            return
        # 连接建立时数据可能还在路上，等它到齐再读（最多 200ms）
        conn.waitForReadyRead(200)
        data = conn.readAll().data()
        if data == b"restart":
            QApplication.quit()      # 让位给新版
        elif data == b"raise":
            pet.showNormal()
            pet.raise_()
            pet.activateWindow()
        conn.deleteLater()

    if server is not None:
        server.newConnection.connect(on_call)

    pet.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
