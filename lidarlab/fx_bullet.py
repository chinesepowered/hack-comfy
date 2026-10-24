"""Bullet Time 2.5D: freeze the decisive frame, swing the camera around the
subject using monocular depth, then let time run again."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import geometry as G
from . import render as R
from .matte import mask_stats, relit_plate


def _blend_frame(images, tau):
    """Sample the clip at fractional frame index tau (linear frame blend)."""
    T = images.shape[0]
    tau = min(max(tau, 0.0), T - 1.0)
    a = int(math.floor(tau))
    b = min(a + 1, T - 1)
    w = tau - a
    return images[a] if w < 1e-3 else images[a] * (1 - w) + images[b] * w


def auto_freeze_frame(masks: torch.Tensor) -> int:
    """Apex of a jump: the frame whose lowest subject pixel (the feet) is highest in frame."""
    cx, cy, top, area, bottom = mask_stats(masks)
    valid = area > 0.25 * area.max()
    score = torch.where(valid, bottom, torch.full_like(bottom, float("inf")))
    return int(torch.argmin(score))


def bullet_time(images: torch.Tensor, masks: torch.Tensor, depth: torch.Tensor, fps: float = 24.0, *,
                plate: torch.Tensor | None = None, freeze_at=-1.0, duration=2.2, angle=14.0, pitch=3.0, zoom=0.35,
                ramp=0.5, flash=True, motion_blur=True, relief=0.25, fov=60.0, progress=None):
    """Returns a longer clip: frames up to the freeze (slowing down over `ramp` seconds), an orbit of
    `duration` seconds around the frozen subject, then the rest of the clip."""
    T, H, W, _ = images.shape
    masks = masks.float().clamp(0, 1)
    depth = G.resize_hw(depth.float(), H, W)
    f0 = auto_freeze_frame(masks) if freeze_at < 0 else min(T - 1, int(round(freeze_at * fps)))
    K = G.intrinsics_from_fov(H, W, fov)

    img0, m0, d0 = images[f0], masks[f0], depth[f0]
    grown = F.max_pool2d(m0[None, None], 21, stride=1, padding=10)[0, 0] > 0.2
    # background behind the subject: true pixels from a clean plate when we have one, otherwise a smooth fill
    if plate is not None:
        bg = relit_plate(plate.float(), img0, m0)
        bg = torch.where(grown[..., None], bg, img0)
    else:
        bg = R.push_pull_fill(img0, ~grown)
    bg_d = R.push_pull_fill(d0[..., None], ~grown)[..., 0]

    # monocular depth across a body is noisy; a gently relieved billboard swings far more cleanly
    if (m0 > 0.5).any():
        dm = float(d0[m0 > 0.5].median())
        d_s = R.blur(d0[..., None], 6.0)[..., 0]
        d0 = torch.where(m0 > 0.5, dm + relief * (d_s - dm), d0)

    # reflect-pad the background so the orbiting camera always has picture at the frame edges
    ph, pw = int(H * 0.25), int(W * 0.25)
    bg_p = F.pad(bg.permute(2, 0, 1)[None], (pw, pw, ph, ph), mode="reflect")[0].permute(1, 2, 0)
    bgd_p = F.pad(bg_d[None, None], (pw, pw, ph, ph), mode="replicate")[0, 0]
    Kp = K.clone()
    Kp[0, 2] += pw
    Kp[1, 2] += ph
    Hp, Wp = bg_p.shape[:2]
    up, vp = G.pixel_grid(Hp, Wp)
    zb = bgd_p.reshape(-1)
    P_bg = torch.stack([(up.reshape(-1) - Kp[0, 2]) / Kp[0, 0] * zb, (vp.reshape(-1) - Kp[1, 2]) / Kp[1, 1] * zb, zb], -1)
    C_bg = bg_p.reshape(-1, 3)

    subj = (m0 > 0.5).reshape(-1)
    uu, vv = G.pixel_grid(H, W)

    def lift(d, sel=None):
        u, v, z = uu.reshape(-1), vv.reshape(-1), d.reshape(-1)
        if sel is not None:
            u, v, z = u[sel], v[sel], z[sel]
        return torch.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)

    P_s = lift(d0, subj)
    C_s = img0.reshape(-1, 3)[subj]
    ys, xs = torch.nonzero(m0 > 0.5, as_tuple=True)
    if ys.numel():
        pc = lift(d0, subj).median(0).values
        pivot = torch.tensor([float(pc[0]), float(pc[1]), float(pc[2])])
    else:
        pivot = torch.tensor([0.0, 0.0, float(d0.median())])
    u0 = float(K[0, 0] * pivot[0] / pivot[2] + K[0, 2])
    v0 = float(K[1, 1] * pivot[1] / pivot[2] + K[1, 2])
    n_bg = P_bg.shape[0]
    P_all = torch.cat([P_bg, P_s])
    C_all = torch.cat([C_bg, C_s])

    def render_view(s):
        e = math.sin(math.pi * s)  # out and back
        Rm, tv = G.orbit_pose(pivot, yaw_deg=angle * e, pitch_deg=pitch * e)
        P = G.transform(P_all, Rm, tv)
        # push in (longer lens) while swinging round so the frame edges never run out of picture
        Kr = K.clone()
        Kr[0, 0] *= 1 + zoom * e
        Kr[1, 1] *= 1 + zoom * e
        # keep the subject in shot: ease its screen position toward the centre while zooming on it
        tx = u0 + (W / 2 - u0) * 0.75 * e
        ty = v0 + (H / 2 - v0) * 0.75 * e
        Kr[0, 2] = tx - Kr[0, 0] * float(pivot[0]) / float(pivot[2])
        Kr[1, 2] = ty - Kr[1, 1] * float(pivot[1]) / float(pivot[2])
        u, v, z, ok = G.project(P, Kr)
        okb, oks = ok[:n_bg], ok[n_bg:]
        img, cov, zbuf = R.splat_zbuffer(u[:n_bg][okb], v[:n_bg][okb], z[:n_bg][okb], C_all[:n_bg][okb], H, W, size=2)
        img = R.push_pull_fill(img, cov)
        # subject on top with fatter splats so it never shows holes
        us, vs, zs = u[n_bg:][oks], v[n_bg:][oks], z[n_bg:][oks]
        simg, scov, _ = R.splat_zbuffer(us, vs, zs, C_all[n_bg:][oks], H, W, size=3)
        return torch.where(scov[..., None], simg, img)

    frames = []
    n_ramp = int(round(ramp * fps))
    # normal playback, slowing to ~25% speed over the ramp window before the freeze
    tau, k = 0.0, 0
    while tau < f0:
        frames.append(_blend_frame(images, tau))
        dist = f0 - tau
        speed = 1.0 if dist > n_ramp or n_ramp == 0 else 0.25 + 0.75 * (dist / n_ramp)
        tau += speed
        k += 1
    n_freeze = max(2, int(round(duration * fps)))
    for j in range(n_freeze):
        s = j / (n_freeze - 1)
        if motion_blur and 0 < j < n_freeze - 1:
            ds = 0.35 / (n_freeze - 1)
            fr = (render_view(max(0.0, s - ds)) + render_view(s) + render_view(min(1.0, s + ds))) / 3
        else:
            fr = render_view(s)
        if flash and j < 3:
            fr = fr + (0.55, 0.3, 0.12)[j]
        frames.append(fr.clamp(0, 1))
        if progress:
            progress(j + 1, n_freeze)
    # ease back up to full speed after the freeze
    tau = float(f0)
    while tau <= T - 1:
        frames.append(_blend_frame(images, tau))
        dist = tau - f0
        speed = 1.0 if dist > n_ramp or n_ramp == 0 else 0.25 + 0.75 * (dist / n_ramp)
        tau += speed
    return torch.stack(frames)
