"""Clones: copies of the subject trail behind it (trail) or freeze along its
path like a multi-exposure photo (stamp)."""
from __future__ import annotations

import colorsys

import torch

MODES = ["trail", "stamp"]
LOOKS = ["natural", "rainbow", "ghost glow", "lidar"]


def _hue(k, n):
    r, g, b = colorsys.hsv_to_rgb((k / max(n, 1)) % 1.0, 0.65, 1.0)
    return torch.tensor([r, g, b])


def _clone_pixels(img, look, k, n, pattern):
    if look == "rainbow":
        luma = (img * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
        return (0.35 * img + 0.85 * luma * _hue(k, n)).clamp(0, 1)
    if look == "lidar":
        luma = (img * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
        return torch.tensor([0.35, 0.9, 1.0]) * (0.4 + 1.2 * luma) * pattern[..., None]
    return img


def clones(images: torch.Tensor, masks: torch.Tensor, fps: float = 24.0, *, mode="trail", count=6,
           interval=0.12, start=0.0, end=0.0, fade=0.65, look="natural", progress=None):
    """images (T,H,W,3), masks (T,H,W) soft subject mattes. `end` <= start means until the clip ends."""
    T, H, W, _ = images.shape
    masks = masks.float().clamp(0, 1)
    if masks.shape[1:] != (H, W):
        masks = torch.nn.functional.interpolate(masks[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    step = max(1, int(round(interval * fps)))
    f_start = int(round(start * fps))
    f_end = T if end <= start else min(T, int(round(end * fps)) + 1)
    pattern = torch.zeros(H, W)
    pattern[::3, ::2] = 1.0
    stamp_frames = [f for f in range(f_start, f_end, step)]
    out = images.clone()
    for i in range(T):
        if i < f_start:
            if progress:
                progress(i + 1, T)
            continue
        if mode == "stamp":
            src = [f for f in stamp_frames if f < i][-count:] if count > 0 else [f for f in stamp_frames if f < i]
            src = src if i < f_end + step else src  # stamps persist after the active window
        else:
            if i >= f_end:
                src = []
            else:
                src = [i - k * step for k in range(count, 0, -1) if i - k * step >= f_start]
        n = len(src)
        if n == 0:
            if progress:
                progress(i + 1, T)
            continue
        base = images[i].clone()
        additive = torch.zeros_like(base)
        for j, f in enumerate(src):
            age = (n - j) / max(n, 1)  # 1 = oldest
            opacity = 1.0 - fade * age if mode == "trail" else 1.0 - 0.5 * fade * age
            a = (masks[f] * max(0.05, opacity))[..., None]
            px = _clone_pixels(images[f], look, j if mode == "trail" else stamp_frames.index(f), max(count, 1), pattern)
            if look in ("ghost glow", "lidar"):
                glow = px * a * (1.2 if look == "lidar" else 0.9)
                if look == "ghost glow":
                    glow = glow * torch.tensor([0.75, 0.9, 1.25])
                additive = additive + glow
            else:
                base = base * (1 - a) + px * a
        base = 1 - (1 - base) * (1 - additive.clamp(0, 1))
        m = masks[i][..., None]
        out[i] = (base * (1 - m) + images[i] * m).clamp(0, 1)
        if progress:
            progress(i + 1, T)
    return out
