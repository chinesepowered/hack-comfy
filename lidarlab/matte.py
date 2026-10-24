"""Static-camera clean plates and subject mattes in milliseconds per frame.

For tripod shots the background is recoverable from the footage itself: the
temporal median of the clip is an empty "clean plate", and the subject is
whatever differs from it. That is the fast CPU path; any MASK from another
node (BiRefNet, SAM 3, ...) can replace the difference matte.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def clean_plate(images: torch.Tensor, max_samples: int = 48, masks: torch.Tensor | None = None,
                refine: int = 1, threshold: float = 0.10, progress=None) -> torch.Tensor:
    """Temporal median of (T,H,W,3) frames -> (H,W,3).

    Masked pixels are ignored when masks are given. Without masks, `refine`
    extra passes matte the subject against the previous plate and re-take the
    median without it, which removes the ghost of a subject that lingers.
    """
    T = images.shape[0]
    idx = torch.linspace(0, T - 1, min(T, max_samples)).round().long().unique()
    stack = images[idx].float()
    if masks is None and refine > 0:
        plate = torch.median(stack, dim=0).values
        n = stack.shape[0]
        for r in range(refine):
            cb = (lambda i, _n, r=r: progress(r * n + i, refine * n)) if progress else None
            m = difference_matte(stack, plate, threshold=threshold, softness=0.0, grow=6, max_subjects=3, progress=cb)
            plate = clean_plate(stack, max_samples=stack.shape[0], masks=m, refine=0)
        return plate
    if masks is not None:
        m = masks[idx].float() if masks.shape[0] == T else masks.float()
        m = F.max_pool2d(m[:, None], 9, stride=1, padding=4)[:, 0] > 0.15  # grow the subject a little
        stack = stack.clone()
        stack[m] = float("nan")
        plate = torch.nanmedian(stack, dim=0).values
        seen = (~m).sum(0)
        holes = seen < max(2, int(0.06 * stack.shape[0]))
        plate = torch.where(torch.isnan(plate), torch.zeros_like(plate), plate)
        if holes.any():
            from .render import push_pull_fill
            plate = push_pull_fill(plate, ~holes)
        return plate
    return torch.median(stack, dim=0).values


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean filter over (N,C,H,W) with radius r (window 2r+1)."""
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int = 8, eps: float = 1e-3) -> torch.Tensor:
    """Edge-aware smoothing of src (N,1,H,W) using a grey guide (N,1,H,W)."""
    mI, mp = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mI * mp
    var = box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * guide + box(b, r)


def _largest_components(binary: np.ndarray, keep: int, min_frac: float) -> np.ndarray:
    from scipy import ndimage
    lab, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes = ndimage.sum(binary, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    big = sizes[order[0]]
    keep_ids = [order[k] + 1 for k in range(min(keep, n)) if sizes[order[k]] >= big * min_frac]
    out = np.isin(lab, keep_ids)
    return ndimage.binary_fill_holes(out)


def difference_matte(images: torch.Tensor, plate: torch.Tensor, threshold: float = 0.10, softness: float = 1.0,
                     grow: int = 2, max_subjects: int = 2, progress=None) -> torch.Tensor:
    """(T,H,W,3) frames vs (H,W,3) plate -> (T,H,W) soft subject masks."""
    T, H, W, _ = images.shape
    out = torch.empty(T, H, W)
    p = plate.permute(2, 0, 1)[None]
    # work at <= 640 px for the binary decision, refine edges at full res
    s = min(1.0, 640.0 / max(H, W))
    h, w = max(8, int(H * s)), max(8, int(W * s))
    k = max(3, int(round(5 * s)) | 1)
    for i in range(T):
        x = images[i].permute(2, 0, 1)[None]
        diff = (x - p).abs()
        # chroma-weighted distance: shadows (darker, same hue) count less than real colour changes
        lum = diff.mean(1, keepdim=True)
        chroma = (diff - lum).abs().mean(1, keepdim=True)
        dist = 0.65 * diff.amax(1, keepdim=True) + 0.9 * chroma
        small = F.interpolate(dist, size=(h, w), mode="area")
        small = F.avg_pool2d(F.pad(small, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        binary = small[0, 0] > threshold
        # open (remove speckle) then close (bridge gaps)
        b = binary[None, None].float()
        b = -F.max_pool2d(-b, k, stride=1, padding=k // 2)
        b = F.max_pool2d(b, k, stride=1, padding=k // 2)
        b = F.max_pool2d(b, 2 * k + 1, stride=1, padding=k)
        b = -F.max_pool2d(-b, 2 * k + 1, stride=1, padding=k)
        bn = _largest_components(b[0, 0].numpy() > 0.5, max_subjects, 0.08)
        b = torch.from_numpy(bn.astype(np.float32))[None, None]
        if grow > 0:
            b = F.max_pool2d(b, 2 * grow + 1, stride=1, padding=grow)
        m = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
        if softness > 0:
            g = x.mean(1, keepdim=True)
            m = guided_filter(g, m, r=max(2, int(6 * softness)), eps=2e-3)
        out[i] = m[0, 0].clamp(0, 1)
        if progress:
            progress(i + 1, T)
    return out


def mask_stats(masks: torch.Tensor):
    """Per-frame centroid (x,y in px), top y, area and bottom y for (T,H,W) masks."""
    T, H, W = masks.shape
    ys = torch.arange(H, dtype=torch.float32)[:, None]
    xs = torch.arange(W, dtype=torch.float32)[None, :]
    area = masks.sum((1, 2)).clamp_min(1e-6)
    cx = (masks * xs).sum((1, 2)) / area
    cy = (masks * ys).sum((1, 2)) / area
    rows = (masks > 0.5).any(2)
    top = torch.where(rows.any(1), rows.float().argmax(1).float(), torch.full((T,), float(H)))
    bottom = torch.where(rows.any(1), (H - 1 - rows.flip(1).float().argmax(1)).float(), torch.full((T,), float(H)))
    return cx, cy, top, area, bottom


# --------------------------------------------------------------------------- BiRefNet (any camera)

def _deform_conv_fp32_shim():
    """torchvision's CPU deform_conv2d has no bf16 kernel: run just that op in fp32."""
    import torchvision.ops
    import comfy.background_removal.birefnet as bm
    if getattr(bm, "_lidarlab_patched", False):
        return
    orig = torchvision.ops.deform_conv2d

    def deform_conv2d(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), mask=None):
        dt = input.dtype
        f = (lambda t: t.float() if t is not None else None)
        with torch.autocast(device_type="cpu", enabled=False):
            return orig(f(input), f(offset), f(weight), f(bias), stride, padding, dilation, f(mask)).to(dt)

    bm.deform_conv2d = deform_conv2d
    bm._lidarlab_patched = True


def birefnet_matte(bg_model, images: torch.Tensor, resolution: int = 768, stride: int = 1, progress=None) -> torch.Tensor:
    """Run ComfyUI's BiRefNet background-removal model on every `stride`-th frame,
    at `resolution`, with bf16 autocast on CPUs that support it; masks between
    keyframes are blended linearly."""
    import comfy.clip_model
    import comfy.model_management as mm
    from .accel import cpu_bf16_ok

    mm.load_model_gpu(bg_model.patcher)
    device = bg_model.load_device
    use_bf16 = device.type == "cpu" and cpu_bf16_ok()
    if use_bf16:
        _deform_conv_fp32_shim()
    T, H, W, _ = images.shape
    keys = list(range(0, T, max(1, stride)))
    if keys[-1] != T - 1:
        keys.append(T - 1)
    km = {}
    for n, i in enumerate(keys):
        px = comfy.clip_model.clip_preprocess(images[i:i + 1].to(device), size=resolution, mean=bg_model.image_mean,
                                              std=bg_model.image_std, crop=False)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = bg_model.model(pixel_values=px.to(torch.bfloat16) if use_bf16 else px)
        m = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False).sigmoid()
        km[i] = m[0, 0].cpu()
        if progress:
            progress(n + 1, len(keys))
    out = torch.empty(T, H, W)
    for a, b in zip(keys[:-1], keys[1:]):
        for i in range(a, b + 1):
            w = 0.0 if b == a else (i - a) / (b - a)
            out[i] = km[a] * (1 - w) + km[b] * w
    if len(keys) == 1:
        out[0] = km[keys[0]]
    return out


def relit_plate(plate: torch.Tensor, frame: torch.Tensor, mask: torch.Tensor | None = None, scale: int = 8,
                sigma: float = 4.0) -> torch.Tensor:
    """Match a clean plate's low-frequency lighting to `frame` (all (H,W,3)).

    Lighting drifts over a long clip (clouds, window light); a smooth per-pixel
    gain estimated on background pixels keeps revealed plate regions seamless.
    """
    from .render import blur, push_pull_fill
    H, W, _ = frame.shape
    w = torch.ones(H, W) if mask is None else 1.0 - F.max_pool2d(mask[None, None].float(), 15, stride=1, padding=7)[0, 0]
    h2, w2 = max(4, H // scale), max(4, W // scale)
    dn = lambda x: F.interpolate(x.permute(2, 0, 1)[None], size=(h2, w2), mode="area")[0].permute(1, 2, 0)
    ws = dn(w[..., None])
    num_f = blur(dn(frame * w[..., None]), sigma)
    num_p = blur(dn(plate * w[..., None]), sigma)
    den = blur(ws, sigma).clamp_min(1e-4)
    gain = ((num_f / den) / (num_p / den).clamp_min(1e-3)).clamp(0.5, 2.0)
    reliable = blur(ws, sigma)[..., 0] > 0.35
    if not reliable.all():
        gain = push_pull_fill(gain, reliable) if reliable.any() else torch.ones_like(gain)
    gain = F.interpolate(gain.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return (plate * gain).clamp(0, 1)
