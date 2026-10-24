"""Camera / depth helpers shared by the LiDAR Lab effects.

Conventions: OpenCV camera space (x right, y down, z forward), depth = distance
along z in the arbitrary-but-consistent units a monocular model returns.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- easing

def smoothstep(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_in_out(x):
    x = min(max(float(x), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ramp(t, start, duration):
    """0 before `start`, 1 after `start + duration`, linear in between."""
    if duration <= 0:
        return 1.0 if t >= start else 0.0
    return min(max((t - start) / duration, 0.0), 1.0)


# --------------------------------------------------------------------------- intrinsics

def intrinsics_from_fov(H: int, W: int, fov_deg: float = 60.0) -> torch.Tensor:
    """Pinhole K for a horizontal field of view in degrees."""
    f = 0.5 * W / math.tan(math.radians(max(5.0, min(fov_deg, 150.0))) / 2.0)
    return torch.tensor([[f, 0.0, (W - 1) / 2.0],
                         [0.0, f, (H - 1) / 2.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)


def pixel_grid(H: int, W: int):
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32), indexing="ij")
    return u, v


def unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(H,W) depth -> (H*W,3) camera-space points."""
    H, W = depth.shape
    u, v = pixel_grid(H, W)
    z = depth
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return torch.stack([x, y, z], -1).reshape(-1, 3)


# --------------------------------------------------------------------------- cameras

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def orbit_pose(pivot: torch.Tensor, yaw_deg: float, pitch_deg: float = 0.0,
               dolly: float = 0.0, lift: float = 0.0, roll_deg: float = 0.0):
    """World->camera transform (R, t) for a camera orbiting `pivot`.

    The original camera sits at the origin looking down +z. Positive yaw swings
    the camera to the right around the pivot, positive pitch raises it.
    `dolly` moves the camera forward along its view axis (same units as depth),
    `lift` moves it up.
    """
    # Rotating the camera by +yaw around the pivot == rotating the scene by -yaw.
    R = rot_z(math.radians(roll_deg)) @ rot_x(math.radians(pitch_deg)) @ rot_y(math.radians(-yaw_deg))
    t = pivot - R @ pivot
    t = t + torch.tensor([0.0, lift, -dolly], dtype=torch.float32)
    return R, t


def transform(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return points @ R.T + t


def project(points: torch.Tensor, K: torch.Tensor, near: float = 1e-3):
    """Camera-space points -> (u, v, z, valid) with float pixel coordinates."""
    z = points[:, 2]
    valid = z > near
    zs = torch.where(valid, z, torch.ones_like(z))
    u = K[0, 0] * points[:, 0] / zs + K[0, 2]
    v = K[1, 1] * points[:, 1] / zs + K[1, 2]
    return u, v, z, valid


# --------------------------------------------------------------------------- depth utils

def robust_range(depth: torch.Tensor, mask: torch.Tensor | None = None, lo=0.02, hi=0.98):
    d = depth if mask is None else depth[mask]
    if d.numel() == 0:
        d = depth.reshape(-1)
    d = d.reshape(-1).float()
    if d.numel() > 400_000:
        d = d[torch.randint(0, d.numel(), (400_000,))]
    q = torch.quantile(d, torch.tensor([lo, hi]))
    return float(q[0]), float(max(q[1], q[0] + 1e-6))


def normalize_depth(depth, d0, d1):
    return ((depth - d0) / (d1 - d0)).clamp(0.0, 1.0)


def stabilize_depth(depth: torch.Tensor, sky: torch.Tensor | None = None,
                    temporal: float = 0.6, edge_tolerance: float = 0.08) -> torch.Tensor:
    """Reduce per-frame flicker of monocular video depth.

    1. Scale-align every frame to a running reference with a robust median ratio
       (monocular models drift in global scale from frame to frame).
    2. Blend each frame with the previous stabilized frame where the two agree
       (static regions); moving regions keep the fresh estimate so nothing ghosts.
    """
    T = depth.shape[0]
    out = torch.empty_like(depth)
    ref = None
    for i in range(T):
        d = depth[i].float().clamp_min(1e-4)
        valid = torch.ones_like(d, dtype=torch.bool) if sky is None else sky[i] < 0.5
        if ref is not None:
            ratio = (ref[valid] / d[valid])
            if ratio.numel() > 200_000:
                ratio = ratio[torch.randint(0, ratio.numel(), (200_000,))]
            s = float(ratio.median()) if ratio.numel() > 16 else 1.0
            d = d * s
            agree = ((d - ref).abs() / ref.clamp_min(1e-4)) < edge_tolerance
            w = torch.where(agree, torch.full_like(d, temporal), torch.zeros_like(d))
            d = w * ref + (1.0 - w) * d
        out[i] = d
        ref = d
    return out


def resize_hw(x: torch.Tensor, H: int, W: int, mode="bilinear"):
    """Resize (T,H,W) or (T,H,W,C) tensors."""
    if x.ndim == 3:
        if x.shape[1:] == (H, W):
            return x
        return F.interpolate(x[:, None].float(), size=(H, W), mode=mode,
                             align_corners=False if mode != "nearest" else None)[:, 0]
    if x.shape[1:3] == (H, W):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2).float(), size=(H, W), mode=mode,
                         align_corners=False if mode != "nearest" else None).permute(0, 2, 3, 1)
