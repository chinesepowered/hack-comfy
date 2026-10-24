"""Tiny CPU-friendly point renderer: z-buffered and additive splats, bloom,
push-pull hole filling and backward warps. Pure torch, no CUDA required."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _offsets(size: int):
    size = max(1, int(size))
    lo = -((size - 1) // 2)
    return [(dx, dy) for dy in range(lo, lo + size) for dx in range(lo, lo + size)]


def splat_zbuffer(u, v, z, colors, H, W, size=1, background=None):
    """Nearest-point-wins splatting.

    u, v, z: (N,) float pixel coords / depth (already filtered to z > 0).
    colors: (N,C). Returns (image (H,W,C), coverage mask (H,W) bool, zbuf (H,W)).
    """
    C = colors.shape[1]
    ui = u.round().long()
    vi = v.round().long()
    zbuf = torch.full((H * W,), float("inf"))
    offs = _offsets(size)
    idx_list = []
    for dx, dy in offs:
        uu, vv = ui + dx, vi + dy
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        idx = vv[ok] * W + uu[ok]
        zz = z[ok]
        zbuf.scatter_reduce_(0, idx, zz, reduce="amin")
        idx_list.append((ok, idx, zz))
    img = torch.zeros(H * W, C) if background is None else background.reshape(H * W, C).clone()
    for ok, idx, zz in idx_list:
        win = zz <= zbuf[idx] * (1.0 + 1e-5)
        img[idx[win]] = colors[ok][win]
    cov = torch.isfinite(zbuf)
    return img.reshape(H, W, C), cov.reshape(H, W), zbuf.reshape(H, W)


def splat_add(u, v, colors, H, W, size=1):
    """Additive splatting (glowing points, no occlusion)."""
    C = colors.shape[1]
    ui = u.round().long()
    vi = v.round().long()
    acc = torch.zeros(H * W, C)
    for dx, dy in _offsets(size):
        uu, vv = ui + dx, vi + dy
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        acc.index_add_(0, vv[ok] * W + uu[ok], colors[ok])
    return acc.reshape(H, W, C)


def splat_add_bilinear(u, v, colors, H, W):
    """Additive splat with bilinear weights: smooth sub-pixel motion for particles."""
    C = colors.shape[1]
    u0 = u.floor()
    v0 = v.floor()
    fu = (u - u0)[:, None]
    fv = (v - v0)[:, None]
    u0 = u0.long()
    v0 = v0.long()
    acc = torch.zeros(H * W, C)
    for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
                      (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        uu, vv = u0 + du, v0 + dv
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        acc.index_add_(0, vv[ok] * W + uu[ok], (colors * w)[ok])
    return acc.reshape(H, W, C)


def _gauss_kernel(sigma: float):
    r = max(1, int(math.ceil(sigma * 3)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum(), r


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of an (H,W,C) image."""
    if sigma <= 0.3:
        return img
    g, r = _gauss_kernel(sigma)
    C = img.shape[-1]
    x = img.permute(2, 0, 1)[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x[0].permute(1, 2, 0)


def bloom(img: torch.Tensor, strength=0.8, radius=12.0, threshold=0.0) -> torch.Tensor:
    """Cheap multi-scale bloom: blur a downsampled copy at two radii."""
    if strength <= 0:
        return img
    H, W, C = img.shape
    src = (img - threshold).clamp_min(0) if threshold > 0 else img
    x = src.permute(2, 0, 1)[None]
    out = torch.zeros_like(img)
    for scale, w in ((4, 0.6), (8, 0.4)):
        small = F.avg_pool2d(x, scale, ceil_mode=True)[0].permute(1, 2, 0)
        small = blur(small, max(0.5, radius / scale))
        up = F.interpolate(small.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)
        out = out + w * up[0].permute(1, 2, 0)
    return img + strength * out


def push_pull_fill(img: torch.Tensor, valid: torch.Tensor, levels: int = 8) -> torch.Tensor:
    """Fill holes (valid==False) with a smooth pyramid interpolation of nearby pixels."""
    H, W, C = img.shape
    w = valid.float()[None, None]
    x = (img.permute(2, 0, 1)[None] * w)
    pyr = [(x, w)]
    for _ in range(levels):
        x, w = pyr[-1]
        if min(x.shape[-2:]) <= 2:
            break
        pyr.append((F.avg_pool2d(x, 2, ceil_mode=True), F.avg_pool2d(w, 2, ceil_mode=True)))
    filled = pyr[-1][0] / pyr[-1][1].clamp_min(1e-6)
    for x, w in reversed(pyr[:-1]):
        up = F.interpolate(filled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cur = x / w.clamp_min(1e-6)
        a = w.clamp(0, 1)
        filled = cur * a + up * (1 - a)
    out = filled[0].permute(1, 2, 0)
    return torch.where(valid[..., None], img, out)


def backward_warp(img: torch.Tensor, du: torch.Tensor, dv: torch.Tensor, mode="bilinear") -> torch.Tensor:
    """Sample img (H,W,C) at (u+du, v+dv) for every output pixel."""
    H, W, C = img.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    gx = (u + du) / max(W - 1, 1) * 2 - 1
    gy = (v + dv) / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    out = F.grid_sample(img.permute(2, 0, 1)[None], grid, mode=mode, padding_mode="border", align_corners=True)
    return out[0].permute(1, 2, 0)


def hex_to_rgb(color: str, default=(0.3, 0.9, 1.0)):
    try:
        c = color.strip().lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return default


def tonemap(img: torch.Tensor) -> torch.Tensor:
    """Soft shoulder so additive glows saturate gracefully instead of clipping."""
    return 1.0 - torch.exp(-img.clamp_min(0) * 1.15)
