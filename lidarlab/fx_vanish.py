"""Vanish: the subject crumbles into particles that drift away on a curl-noise
wind, revealing the clean plate behind them."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import render as R
from .matte import relit_plate

DIRECTIONS = ["left to right", "right to left", "top to bottom", "bottom to top", "outside in"]
PARTICLES = ["ash", "embers", "sparks", "pixels"]


def _value_noise(H, W, cells=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(1, 1, H, W)
    amp, total = 1.0, 0.0
    for o in range(3):
        n = cells * (2 ** o)
        grid = torch.rand(1, 1, n, n, generator=g)
        out += amp * F.interpolate(grid, size=(H, W), mode="bicubic", align_corners=False)
        total += amp
        amp *= 0.5
    out = out / total
    return ((out - out.min()) / (out.max() - out.min() + 1e-6))[0, 0]


class _Curl:
    """Divergence-free 2D velocity field from a sum of travelling sine potentials."""

    def __init__(self, scale_px, seed=0, n=5):
        g = torch.Generator().manual_seed(seed + 7)
        ang = torch.rand(n, generator=g) * 2 * math.pi
        k = (2 * math.pi / scale_px) * (0.6 + torch.rand(n, generator=g) * 1.2)
        self.kx, self.ky = k * torch.cos(ang), k * torch.sin(ang)
        self.w = (torch.rand(n, generator=g) - 0.5) * 3.0
        self.ph = torch.rand(n, generator=g) * 2 * math.pi
        self.a = 1.0 / (k * n)

    def __call__(self, x, y, t):
        arg = x[:, None] * self.kx + y[:, None] * self.ky + self.w * t + self.ph
        c = torch.cos(arg) * self.a
        dpx = (c * self.kx).sum(1)
        dpy = (c * self.ky).sum(1)
        return dpy, -dpx  # (dpsi/dy, -dpsi/dx)


def vanish(images: torch.Tensor, masks: torch.Tensor, plate: torch.Tensor, fps: float = 24.0, *,
           start=1.0, duration=1.8, direction="left to right", particles="ash", color="#ff7a2f",
           wind_angle=-20.0, wind_speed=1.0, turbulence=1.0, density=0.45, freeze_subject=True, seed=0,
           relight=True, progress=None):
    """images (T,H,W,3), masks (T,H,W), plate (H,W,3) clean background."""
    T, H, W, _ = images.shape
    masks = masks.float().clamp(0, 1)
    plate = plate.float()
    if plate.shape[:2] != (H, W):
        plate = F.interpolate(plate.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    glow_col = torch.tensor(R.hex_to_rgb(color, (1.0, 0.48, 0.18)))
    noise_lr = _value_noise(256, 256, seed=seed)
    curl = _Curl(scale_px=0.18 * max(H, W), seed=seed)
    gen = torch.Generator().manual_seed(seed)
    f0 = int(round(start * fps))
    scale = max(H, W) / 1280.0
    wa = math.radians(wind_angle)
    wind = torch.tensor([math.cos(wa), math.sin(wa)]) * 260.0 * wind_speed * scale
    if particles == "embers":
        gravity = torch.tensor([0.0, -90.0 * scale])
    elif particles == "ash":
        gravity = torch.tensor([0.0, 25.0 * scale])
    else:
        gravity = torch.tensor([0.0, 0.0])

    # live particle state
    pos = torch.zeros(0, 2)
    vel = torch.zeros(0, 2)
    col = torch.zeros(0, 3)
    age = torch.zeros(0)
    life = torch.zeros(0)
    prev_p = -1.0
    dt = 1.0 / fps
    uu = torch.arange(W, dtype=torch.float32)[None, :].expand(H, W)
    vv = torch.arange(H, dtype=torch.float32)[:, None].expand(H, W)
    out = torch.empty_like(images)
    frozen = None
    for i in range(T):
        t = i / fps
        img = images[i]
        if i < f0:
            out[i] = img
            if progress:
                progress(i + 1, T)
            continue
        if freeze_subject:
            if frozen is None:
                frozen = (images[f0].clone(), masks[f0].clone())
            s_img, m = frozen
        else:
            s_img, m = img, masks[i]
        x = (t - start) / max(duration, 1e-3)
        p = min(max(x, 0.0), 1.0)
        p = p * p * (3 - 2 * p) * 1.12 - 0.04

        # dissolve coordinate in subject-relative space so the pattern rides along with it
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if ys.numel() > 16:
            x0, x1 = float(xs.min()), float(xs.max()) + 1
            y0, y1 = float(ys.min()), float(ys.max()) + 1
        else:
            x0, x1, y0, y1 = 0.0, float(W), 0.0, float(H)
        nx = ((uu - x0) / max(x1 - x0, 1)).clamp(0, 1)
        ny = ((vv - y0) / max(y1 - y0, 1)).clamp(0, 1)
        if direction == "right to left":
            s = 1 - nx
        elif direction == "top to bottom":
            s = ny
        elif direction == "bottom to top":
            s = 1 - ny
        elif direction == "outside in":
            s = 1 - (((nx - 0.5) ** 2 + (ny - 0.5) ** 2).sqrt() / 0.7071)
        else:
            s = nx
        grid = torch.stack([nx * 2 - 1, ny * 2 - 1], -1)[None]
        n = F.grid_sample(noise_lr[None, None], grid, mode="bilinear", align_corners=False)[0, 0]
        theta = 0.72 * s + 0.28 * n

        remain = m * ((theta - p) / 0.035).clamp(0, 1)
        front = m * torch.exp(-((theta - p) / 0.022) ** 2) * (theta > p - 0.01).float()

        # emit particles from pixels that crossed the front since the last frame
        newly = (m > 0.4) & (theta >= prev_p) & (theta < p)
        prev_p = max(prev_p, p)
        idx = newly.reshape(-1).nonzero().squeeze(1)
        if idx.numel():
            keep = torch.rand(idx.numel(), generator=gen) < density
            idx = idx[keep]
        if idx.numel():
            k = idx.numel()
            px = torch.stack([uu.reshape(-1)[idx], vv.reshape(-1)[idx]], 1)
            c = s_img.reshape(-1, 3)[idx]
            if particles == "embers":
                c = 0.35 * c + 0.9 * glow_col
            elif particles == "sparks":
                c = glow_col.expand(k, 3) * (0.8 + 0.6 * torch.rand(k, 1, generator=gen))
            jitter = torch.randn(k, 2, generator=gen) * 60.0 * scale
            v0 = wind * (0.4 + 0.9 * torch.rand(k, 1, generator=gen)) + jitter
            pos = torch.cat([pos, px + torch.rand(k, 2, generator=gen) - 0.5])
            vel = torch.cat([vel, v0])
            col = torch.cat([col, c])
            age = torch.cat([age, torch.zeros(k)])
            life = torch.cat([life, 0.9 + 1.1 * torch.rand(k, generator=gen)])

        # integrate
        if pos.shape[0]:
            cx, cy = curl(pos[:, 0], pos[:, 1], t)
            turb = torch.stack([cx, cy], 1) * 2600.0 * turbulence * scale
            vel = vel + (turb + gravity) * dt
            vel = vel * (1.0 - 0.6 * dt)
            pos = pos + vel * dt
            age = age + dt
            alive = (age < life) & (pos[:, 0] > -20) & (pos[:, 0] < W + 20) & (pos[:, 1] > -20) & (pos[:, 1] < H + 20)
            pos, vel, col, age, life = pos[alive], vel[alive], col[alive], age[alive], life[alive]

        # composite: background without the subject, the part that is left, the burning edge, the particles
        mb = masks[i] if not freeze_subject else torch.maximum(masks[i], m)
        bg = relit_plate(plate, img, mb) if relight else plate
        base = img * (1 - mb[..., None]) + bg * mb[..., None]
        base = base * (1 - remain[..., None]) + s_img * remain[..., None]
        if particles in ("embers", "sparks"):
            edge = front[..., None] * glow_col * 1.6
            base = 1 - (1 - base) * (1 - R.tonemap(R.bloom(edge, 0.8, 8)))
        else:
            dark = front[..., None] * 0.55
            base = base * (1 - dark) + dark * glow_col * 0.25
        if pos.shape[0]:
            fade = (1.0 - age / life).clamp(0, 1) ** 1.3
            if particles in ("embers", "sparks"):
                acc = R.splat_add_bilinear(pos[:, 0], pos[:, 1], col * fade[:, None] * 1.3, H, W)
                base = 1 - (1 - base) * (1 - R.tonemap(R.bloom(acc, 0.9, 6)))
            elif particles == "pixels":
                a = fade[:, None]
                img_p, cov, _ = R.splat_zbuffer(pos[:, 0], pos[:, 1], age, torch.cat([col * a, a], 1), H, W, size=3)
                A = img_p[..., 3:4].clamp(0, 1)
                base = base * (1 - A) + img_p[..., :3]
            else:  # ash: soft alpha-blended specks
                a = fade[:, None] * 0.9
                acc = R.splat_add_bilinear(pos[:, 0], pos[:, 1], torch.cat([col * a, a], 1), H, W)
                acc = R.blur(acc, 0.6)
                A = acc[..., 3:4].clamp(0, 1)
                C = acc[..., :3] / acc[..., 3:4].clamp_min(1e-6)
                base = base * (1 - A) + C.clamp(0, 1) * A
        out[i] = base.clamp(0, 1)
        if progress:
            progress(i + 1, T)
    return out
