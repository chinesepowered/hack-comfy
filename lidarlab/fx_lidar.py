"""LiDAR Scan: a laser sweep turns live footage into a glowing 3D point cloud,
the camera drifts through it, then the sweep hands the shot back to video."""
from __future__ import annotations

import math

import torch

from . import geometry as G
from . import render as R

STYLES = ["cyan", "rgb", "thermal", "matrix", "gold", "custom"]
SWEEPS = ["depth (near to far)", "depth (far to near)", "left to right", "right to left", "top to bottom", "radial"]
CAMERAS = ["orbit", "dolly in", "crane up", "push and orbit", "none"]

_TINTS = {
    "cyan": (0.30, 0.88, 1.00),
    "matrix": (0.35, 1.00, 0.45),
    "gold": (1.00, 0.72, 0.30),
}


def _inferno(x: torch.Tensor) -> torch.Tensor:
    """Small inferno-like ramp (no matplotlib dependency)."""
    stops = torch.tensor([
        [0.00, 0.00, 0.02], [0.26, 0.04, 0.41], [0.58, 0.15, 0.40],
        [0.87, 0.32, 0.22], [0.99, 0.64, 0.04], [0.99, 0.99, 0.75]])
    x = x.clamp(0, 1) * (len(stops) - 1)
    i0 = x.floor().long().clamp(max=len(stops) - 2)
    f = (x - i0.float())[..., None]
    return stops[i0] * (1 - f) + stops[i0 + 1] * f


def _point_colors(style, tint, rgb, qd):
    """rgb: (N,3) source colours, qd: (N,) normalised depth 0=near."""
    luma = (rgb * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
    near = (1.0 - qd)[:, None]
    if style == "rgb":
        return rgb * (0.55 + 0.75 * near)
    if style == "thermal":
        return _inferno(0.15 + 0.85 * (0.65 * (1 - qd) + 0.35 * luma[:, 0])) * 1.1
    t = torch.tensor(tint)
    return t * (0.35 + 1.2 * luma) * (0.5 + 0.9 * near ** 1.2)


def _sweep_coord(kind, d, d0, d1, H, W):
    if kind.startswith("depth"):
        # log-depth so the front travels at a perceptually even pace; far haze may exceed 1
        q = (torch.log(d.clamp_min(1e-4)) - math.log(d0)) / max(math.log(d1) - math.log(d0), 1e-6)
        q = q.clamp(-0.05, 1.15)
        return 1.1 - q if "far to near" in kind else q
    u, v = G.pixel_grid(H, W)
    if kind == "left to right":
        return u / (W - 1)
    if kind == "right to left":
        return 1 - u / (W - 1)
    if kind == "top to bottom":
        return v / (H - 1)
    r = torch.sqrt((u - W / 2) ** 2 + (v - H / 2) ** 2)
    return r / r.max()


def lidar_scan(images: torch.Tensor, depth: torch.Tensor, sky: torch.Tensor | None = None, fps: float = 24.0, *,
               style="cyan", color="#4de1ff", sweep="depth (near to far)", scan_start=0.6, scan_duration=1.6,
               hold=2.4, camera="orbit", camera_amount=0.6, return_to_video=True, density=4, point_size=1,
               glow=1.0, ghost=0.06, fov=60.0, drop_sky=True, progress=None):
    """images (T,H,W,3) in [0,1]; depth (T,h,w); sky (T,h,w) probability or None."""
    T, H, W, _ = images.shape
    depth = G.resize_hw(depth, H, W)
    sky = G.resize_hw(sky, H, W) if sky is not None else None
    K = G.intrinsics_from_fov(H, W, fov)
    tint = R.hex_to_rgb(color) if style == "custom" else _TINTS.get(style, _TINTS["cyan"])

    ground = None if sky is None else (sky < 0.5)
    sample = depth[:: max(1, T // 12)]
    gmask = None if ground is None else ground[:: max(1, T // 12)]
    d0, d1 = G.robust_range(sample, gmask, 0.02, 0.97)

    # scanline sampling pattern (LiDAR "rings")
    step_v = max(1, int(density))
    step_u = max(1, int(round(density * 0.6)))
    pattern = torch.zeros(H, W, dtype=torch.bool)
    pattern[::step_v, ::step_u] = True
    band_pattern = torch.zeros(H, W, dtype=torch.bool)
    band_pattern[::max(1, step_v // 2), :] = True
    uu, vv = G.pixel_grid(H, W)
    uu_f, vv_f = uu.reshape(-1), vv.reshape(-1)

    band = 0.035
    q_end = 1.0 + 3 * band              # sky / dropped pixels sit just past the far plane
    t_in0, t_in1 = scan_start, scan_start + scan_duration
    t_hold1 = t_in1 + hold
    t_out1 = t_hold1 + (scan_duration if return_to_video else 0.0)
    travel = q_end + 2 * band

    out = torch.empty_like(images)
    for i in range(T):
        t = i / fps
        img = images[i]
        d = depth[i]
        sweeping = False
        if t < t_in0:
            front = -band
        elif t < t_in1:
            front = -band + travel * G.ease_in_out((t - t_in0) / scan_duration)
            sweeping = True
        elif t < t_hold1 or not return_to_video:
            front = q_end + band
        elif t < t_out1:
            front = q_end + band - travel * G.ease_in_out((t - t_hold1) / scan_duration)
            sweeping = True
        else:
            front = -band

        if front <= -band + 1e-6:
            out[i] = img
            if progress:
                progress(i + 1, T)
            continue

        # camera move only while the whole frame is a point cloud
        cam = 0.0
        if t_in1 <= t < t_hold1 and hold > 0:
            x = (t - t_in1) / hold
            cam = math.sin(math.pi * x) if return_to_video else G.ease_in_out(min(1.0, x * 1.4))
        elif t >= t_hold1 and not return_to_video and hold > 0:
            cam = 1.0
        cam *= camera_amount

        keep = torch.ones(H, W, dtype=torch.bool) if not (drop_sky and ground is not None) else ground[i]
        q = _sweep_coord(sweep, d, d0, d1, H, W).clamp(0.0, 1.0)
        q = torch.where(keep, q, torch.full_like(q, 1.0 + 2 * band))
        qd = G.normalize_depth(d, d0, d1)
        scanned_any = q < front - band
        scanned = scanned_any & keep
        in_band = ((q - (front - band)).abs() < band * 0.55) & keep if sweeping else torch.zeros_like(keep)

        # depth edges make the structure read (silhouettes, rooftops, kerbs)
        ld = torch.log(d.clamp_min(1e-4))
        gx = torch.zeros_like(ld); gy = torch.zeros_like(ld)
        gx[:, 1:] = (ld[:, 1:] - ld[:, :-1]).abs(); gy[1:, :] = (ld[1:, :] - ld[:-1, :]).abs()
        edge = ((gx + gy) * 18).clamp(0, 1).reshape(-1)

        pts_mask = scanned & pattern
        idx = pts_mask.reshape(-1).nonzero().squeeze(1)
        bidx = (in_band & band_pattern).reshape(-1).nonzero().squeeze(1)
        rgb = img.reshape(-1, 3)
        qd_f = qd.reshape(-1)
        d_f = d.reshape(-1)

        def pts(ix):
            z = d_f[ix]
            return torch.stack([(uu_f[ix] - K[0, 2]) / K[0, 0] * z, (vv_f[ix] - K[1, 2]) / K[1, 1] * z, z], -1)

        pivot = torch.tensor([0.0, 0.0, float(d[keep].median()) if keep.any() else float(d.median())])
        if cam > 1e-4:
            if camera == "orbit":
                Rm, tv = G.orbit_pose(pivot, yaw_deg=14 * cam, pitch_deg=-4 * cam)
            elif camera == "dolly in":
                Rm, tv = G.orbit_pose(pivot, yaw_deg=0, dolly=0.35 * float(pivot[2]) * cam)
            elif camera == "crane up":
                Rm, tv = G.orbit_pose(pivot, yaw_deg=0, pitch_deg=-12 * cam, lift=0.12 * float(pivot[2]) * cam)
            elif camera == "push and orbit":
                Rm, tv = G.orbit_pose(pivot, yaw_deg=10 * cam, pitch_deg=-3 * cam, dolly=0.2 * float(pivot[2]) * cam)
            else:
                Rm, tv = torch.eye(3), torch.zeros(3)
        else:
            Rm, tv = torch.eye(3), torch.zeros(3)

        layer = torch.zeros(H, W, 3)
        if idx.numel():
            P = G.transform(pts(idx), Rm, tv)
            u, v, z, ok = G.project(P, K)
            cols = _point_colors(style, tint, rgb[idx], qd_f[idx]) * (1.0 + 1.3 * edge[idx])[:, None]
            layer = R.splat_add(u[ok], v[ok], cols[ok], H, W, size=point_size)
        if bidx.numel():
            P = G.transform(pts(bidx), Rm, tv)
            u, v, z, ok = G.project(P, K)
            hot = torch.tensor(tint) * 0.6 + 0.6 if style != "rgb" else torch.tensor([0.9, 0.95, 1.0])
            cols = hot.expand(int(ok.sum()), 3) * 1.25
            layer = layer + R.splat_add(u[ok], v[ok], cols, H, W, size=1)
        layer = R.bloom(layer, strength=0.55 * glow, radius=6)
        layer = R.tonemap(layer)

        if cam > 1e-4:
            frame = layer
        else:
            vis = (~scanned_any).float()[..., None]  # video still visible where not yet scanned
            frame = img * vis + img * ghost * (1 - vis)
            frame = 1 - (1 - frame) * (1 - layer)  # screen blend
        out[i] = frame.clamp(0, 1)
        if progress:
            progress(i + 1, T)
    return out
