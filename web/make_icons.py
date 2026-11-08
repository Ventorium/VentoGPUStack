#!/usr/bin/env python3
"""生成 VentoGPUStack 图标：GPU 芯片风格（深色圆角底 + 绿色芯片 + 引脚 + 核心网格）。"""
import struct, zlib, os

def png_chunk(tag, data):
    c = struct.pack(">I", len(data)) + tag + data
    c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return c

def make_png(w, h, pixels):
    """pixels: h 行 x w 列 的 (r,g,b,a) 列表"""
    raw = b""
    for row in pixels:
        raw += b"\x00" + b"".join(bytes(p) for p in row)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", ihdr)
            + png_chunk(b"IDAT", zlib.compress(raw, 9))
            + png_chunk(b"IEND", b""))

def clamp(v):
    return max(0, min(255, int(v)))

def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

def gen(size):
    S = 1024  # 在 1024 画布上画，最后缩放到 size
    canvas = [[(16, 18, 22, 255)] * S for _ in range(S)]

    def rect(x0, y0, x1, y1, color):
        for y in range(max(0, y0), min(S, y1)):
            for x in range(max(0, x0), min(S, x1)):
                canvas[y][x] = color

    def rounded_mask(x, y, r):
        # 圆角圆角判定
        cx0, cx1 = r, S - 1 - r
        cy0, cy1 = r, S - 1 - r
        if x < cx0 and y < cy0:
            dx, dy = x, y
        elif x >= cx1 and y < cy0:
            dx, dy = S - 1 - x, y
        elif x < cx0 and y >= cy1:
            dx, dy = x, S - 1 - y
        elif x >= cx1 and y >= cy1:
            dx, dy = S - 1 - x, S - 1 - y
        else:
            return True
        return dx * dx + dy * dy <= r * r

    # 背景圆角矩形（深灰）
    BG = (24, 26, 32, 255)
    for y in range(S):
        for x in range(S):
            if rounded_mask(x, y, 180):
                canvas[y][x] = BG

    # 芯片引脚（绿色细线）
    PIN = (61, 220, 132, 255)
    PIN_DARK = (34, 160, 88, 255)
    # 上下左右四侧各 5 根引脚
    side = 96
    for i in range(5):
        off = 240 + i * 128
        # 上
        rect(off - 18, 120, off + 18, 240, PIN_DARK)
        # 下
        rect(off - 18, S - 240, off + 18, S - 120, PIN_DARK)
        # 左
        rect(120, off - 18, 240, off + 18, PIN_DARK)
        # 右
        rect(S - 240, off - 18, S - 120, off + 18, PIN_DARK)

    # 芯片主体（深绿渐变底）
    CHIP_TOP = (30, 58, 44, 255)
    CHIP_BOT = (18, 40, 30, 255)
    for y in range(240, 784):
        t = (y - 240) / (784 - 240)
        color = tuple(clamp(v) for v in lerp(CHIP_TOP, CHIP_BOT, t))
        for x in range(240, 784):
            canvas[y][x] = color

    # 芯片内部核心网格：4x4 亮绿色方块
    CORE = (72, 240, 150, 255)
    CORE_DIM = (44, 170, 100, 255)
    gx0, gx1 = 330, 694
    cell = (gx1 - gx0) // 4
    gap = 22
    for r in range(4):
        for c in range(4):
            cx0 = gx0 + c * cell + gap
            cy0 = 330 + r * cell + gap
            cx1 = gx0 + (c + 1) * cell - gap
            cy1 = 330 + (r + 1) * cell - gap
            col = CORE if (r + c) % 2 == 0 else CORE_DIM
            rect(cx0, cy0, cx1, cy1, col)

    # 中心亮点（核心 die）
    cx0, cy0, cx1, cy1 = 440, 440, 584, 584
    BRIGHT = (140, 255, 190, 255)
    for y in range(cy0, cy1):
        for x in range(cx0, cx1):
            t = (x - cx0) / (cx1 - cx0)
            col = tuple(clamp(v) for v in lerp((190, 255, 220, 255), BRIGHT, t))
            canvas[y][x] = col

    # 缩放到目标尺寸（最近邻 + 轻微抗锯齿：2x2 盒采样）
    out = []
    for y in range(size):
        row = []
        sy0 = y * S // size
        sy1 = max(sy0 + 1, (y + 1) * S // size)
        for x in range(size):
            sx0 = x * S // size
            sx1 = max(sx0 + 1, (x + 1) * S // size)
            acc = [0.0, 0.0, 0.0, 0.0]
            n = 0
            for yy in range(sy0, sy1):
                for xx in range(sx0, sx1):
                    p = canvas[yy][xx]
                    for i in range(4):
                        acc[i] += p[i]
                    n += 1
            row.append(tuple(int(a / n) for a in acc))
        out.append(row)
    return make_png(size, size, out)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
open("favicon.png", "wb").write(gen(32))
open("apple-touch-icon.png", "wb").write(gen(180))
open("logo.png", "wb").write(gen(512))
print("icons written")
