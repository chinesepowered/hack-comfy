#!/usr/bin/env python3
"""Render the demo video's overlay graphics (transparent 1920x1080 PNGs)."""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FONT_DIR = "/usr/share/fonts/opentype/inter"
CYAN = (77, 225, 255)
VIOLET = (167, 123, 255)
PINK = (255, 95, 174)
WHITE = (238, 244, 255)
MUTED = (150, 165, 195)


def font(size, weight="Bold"):
    for name in (f"InterDisplay-{weight}.otf", f"Inter-{weight}.otf", "InterDisplay-Bold.otf"):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def gradient_text(img, xy, text, f, colors=(CYAN, VIOLET, PINK)):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = (int(round(v)) for v in d.textbbox(xy, text, font=f))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).text(xy, text, font=f, fill=255)
    grad = Image.new("RGBA", img.size)
    gd = ImageDraw.Draw(grad)
    span = max(1, x1 - x0)
    for x in range(x0, x1 + 1):
        t = (x - x0) / span
        c = colors[0] if len(colors) == 1 else None
        if t < 0.5:
            a, b, k = colors[0], colors[1], t / 0.5
        else:
            a, b, k = colors[1], colors[2], (t - 0.5) / 0.5
        c = tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3)) + (255,)
        gd.line([(x, y0 - 10), (x, y1 + 10)], fill=c)
    img.paste(grad, (0, 0), mask)


def pill(d, xy, text, f, fill=(5, 8, 16, 200), outline=(255, 255, 255, 60), color=WHITE, pad=(22, 12)):
    x, y = xy
    tw = d.textlength(text, font=f)
    asc, desc = f.getmetrics()
    h = asc + desc + 2 * pad[1]
    d.rounded_rectangle([x, y, x + tw + 2 * pad[0], y + h], radius=h // 2, fill=fill, outline=outline, width=2)
    d.text((x + pad[0], y + pad[1] - 2), text, font=f, fill=color)
    return x + tw + 2 * pad[0], y + h


def shadowed(base, draw_fn, blur=18, alpha=150):
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(sh), True)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    a = sh.getchannel("A").point(lambda v: min(255, int(v * alpha / 255 * 2)))
    sh.putalpha(a)
    base.alpha_composite(sh)
    draw_fn(ImageDraw.Draw(base), False)


def title_card(out):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=(3, 5, 10, 150))
    f1, f2, f3 = font(150, "Bold"), font(44, "Medium"), font(30, "SemiBold")
    t = "LiDAR Lab"
    tw = d.textlength(t, font=f1)
    gradient_text(img, ((W - tw) / 2, 330), t, f1)
    sub = "Open-source Higgsfield effects on ComfyUI"
    d.text(((W - d.textlength(sub, font=f2)) / 2, 525), sub, font=f2, fill=WHITE)
    chips = ["open weights only", "no video diffusion", "runs on a CPU"]
    widths = [d.textlength(c, font=f3) + 44 for c in chips]
    x = (W - (sum(widths) + 24 * (len(chips) - 1))) / 2
    for c, w in zip(chips, widths):
        pill(d, (x, 620), c, f3, fill=(77, 225, 255, 34), outline=(77, 225, 255, 140), color=(220, 248, 255))
        x += w + 24
    img.save(out)


def label(out, text, sub=None):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f, fs = font(46, "Bold"), font(26, "Medium")
    x2, y2 = pill(d, (64, 60), text, f, fill=(5, 8, 16, 190), outline=(77, 225, 255, 150))
    if sub:
        d.text((70, y2 + 14), sub, font=fs, fill=(220, 230, 250))
    img.save(out)


def badge(out, text, pos="tr"):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(34, "SemiBold")
    tw = d.textlength(text, font=f)
    x = W - tw - 44 - 64 if pos == "tr" else 64
    pill(d, (x, 60), text, f, fill=(255, 95, 174, 210), outline=(255, 255, 255, 90))
    img.save(out)


def bench_card(out):
    img = Image.new("RGBA", (W, H), (4, 6, 12, 238))
    d = ImageDraw.Draw(img)
    f1, f2, f3, f4 = font(64, "Bold"), font(34, "Medium"), font(40, "Bold"), font(28, "Medium")
    d.text((160, 150), "Depth Anything 3 on a 4-core CPU", font=f1, fill=WHITE)
    d.text((160, 240), "720p frame, Mono-Large. Same model, same machine.", font=f2, fill=MUTED)
    rows = [("ComfyUI CPU default (fp32)", 1.67, (110, 120, 145)), ("LiDAR Lab: bf16 on Intel AMX", 0.49, CYAN)]
    maxv = 1.8
    for i, (name, v, col) in enumerate(rows):
        y = 400 + i * 190
        d.text((160, y), name, font=f2, fill=WHITE)
        L = int(1300 * v / maxv)
        d.rounded_rectangle([160, y + 60, 160 + L, y + 130], radius=16, fill=col + (255,))
        d.text((180 + L, y + 68), f"{v:.2f} s / frame", font=f3, fill=WHITE)
    gradient_text(img, (160, 800), "3.4x faster. No GPU.", font(76, "Bold"))
    d.text((160, 905), "Measured with torch 2.14 on an AMX Xeon: bf16 matmul 3.6 TFLOPS vs fp32 0.7 TFLOPS.", font=f4, fill=MUTED)
    img.save(out)


def pipeline_card(out):
    img = Image.new("RGBA", (W, H), (4, 6, 12, 238))
    d = ImageDraw.Draw(img)
    f1, f2, f3 = font(58, "Bold"), font(32, "SemiBold"), font(25, "Medium")
    d.text((140, 120), "Geometry, not hallucination", font=f1, fill=WHITE)
    steps = [("Your clip", "Load / Trim Video", (200, 210, 230)),
             ("Depth Anything 3", "native ComfyUI node", CYAN),
             ("Stabilize + lift to 3D", "LiDAR Lab", VIOLET),
             ("Point-cloud renderer", "plain PyTorch, CPU", PINK),
             ("Your effect", "Save Video", (200, 210, 230))]
    bw, bh, gap, y = 332, 170, 34, 380
    x = (W - (len(steps) * bw + (len(steps) - 1) * gap)) / 2
    for i, (t, s, col) in enumerate(steps):
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=22, fill=(14, 22, 40, 255), outline=col + (255,), width=3)
        d.text((x + (bw - d.textlength(t, font=f2)) / 2, y + 42), t, font=f2, fill=WHITE)
        d.text((x + (bw - d.textlength(s, font=f3)) / 2, y + 98), s, font=f3, fill=MUTED)
        if i < len(steps) - 1:
            ax = x + bw + 8
            d.polygon([(ax, y + bh / 2 - 14), (ax + gap - 16, y + bh / 2), (ax, y + bh / 2 + 14)], fill=(120, 140, 180, 255))
        x += bw + gap
    lines = ["Deterministic and stable from frame to frame", "Full resolution, never redraws your face",
             "Mattes: clean plate from the footage, or BiRefNet"]
    for i, l in enumerate(lines):
        d.ellipse([160, 680 + i * 70 + 12, 178, 680 + i * 70 + 30], fill=CYAN + (255,))
        d.text((200, 680 + i * 70), l, font=font(36, "Medium"), fill=WHITE)
    img.save(out)


def end_card(out):
    img = Image.new("RGBA", (W, H), (3, 5, 10, 255))
    d = ImageDraw.Draw(img)
    f1 = font(140, "Bold")
    t = "LiDAR Lab"
    gradient_text(img, ((W - d.textlength(t, font=f1)) / 2, 260), t, f1)
    f2, f3, f4 = font(46, "SemiBold"), font(38, "Medium"), font(26, "Medium")
    url = "github.com/chinesepowered/hack-comfy"
    d.text(((W - d.textlength(url, font=f2)) / 2, 470), url, font=f2, fill=WHITE)
    tag = "#OpenSourceHiggsfield"
    d.text(((W - d.textlength(tag, font=f3)) / 2, 550), tag, font=f3, fill=CYAN)
    lines = ["Apache-2.0 code · Depth Anything 3 (Apache-2.0) · BiRefNet (MIT) · no partner nodes",
             "Footage: Mixkit free stock video · Voice-over, music and sound effects in this video: ElevenLabs"]
    for i, l in enumerate(lines):
        d.text(((W - d.textlength(l, font=f4)) / 2, 830 + i * 46), l, font=f4, fill=MUTED)
    img.save(out)


def ff_badge(out, text):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(36, "Bold")
    tw = d.textlength(text, font=f)
    pill(d, (64, 60), text, f, fill=(8, 12, 22, 225), outline=(77, 225, 255, 160), color=(220, 248, 255))
    img.save(out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cards"
    os.makedirs(out, exist_ok=True)
    title_card(f"{out}/title.png")
    for name, text, sub in [("lidar", "LiDAR Scan", "like Higgsfield's Lidar transition"),
                            ("clones", "Clones", "like Higgsfield's Infinite clones"),
                            ("vanish", "Vanish", "like Higgsfield's Vanish"),
                            ("bullet", "Bullet Time", "like Higgsfield's Bullet time"),
                            ("thermal", "LiDAR Scan · thermal", None), ("gold", "LiDAR Scan · gold", None),
                            ("rgb", "LiDAR Scan · rgb", None), ("matrix", "LiDAR Scan · matrix", None),
                            ("clones_lidar", "Clones · lidar look", None), ("clones_trail", "Clones · trail", None),
                            ("vanish_pixels", "Vanish · pixels", None), ("app", "Gallery app · Comfy SDK", None),
                            ("comfy", "Native ComfyUI graph", None), ("appmode", "ComfyUI App Mode", None),
                            ("plate", "Clean plate from the footage → Vanish", None),
                            ("matte", "BiRefNet matte · moving camera", None)]:
        label(f"{out}/label_{name}.png", text, sub)
    badge(f"{out}/badge_cpu.png", "rendered on CPU · no GPU")
    ff_badge(f"{out}/ff.png", "▶▶  fast-forward: CPU render took about a minute")
    bench_card(f"{out}/bench.png")
    pipeline_card(f"{out}/pipeline.png")
    end_card(f"{out}/end.png")
    print("cards written to", out)
