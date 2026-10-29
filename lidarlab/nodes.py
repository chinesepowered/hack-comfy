"""ComfyUI nodes for LiDAR Lab (V3 node API)."""
from __future__ import annotations

import torch
from typing_extensions import override

import comfy.utils
from comfy_api.latest import ComfyExtension, io

from . import fx_bullet, fx_clones, fx_lidar, fx_vanish
from . import geometry as G
from . import matte as M

DA3Geometry = io.Custom("DA3_GEOMETRY")
CATEGORY = "LiDAR Lab"
FX = "LiDAR Lab/effects"
UTIL = "LiDAR Lab/utils"


def _pbar(total):
    bar = comfy.utils.ProgressBar(total)
    return lambda i, n: bar.update_absolute(i, n)


def _fps_input():
    return io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001,
                          tooltip="Frame rate of the clip. Wire it from Get Video Components so effect timings are in seconds.")


def _depth_from(geometry: dict, T: int):
    depth = geometry["depth"]
    sky = geometry.get("sky")
    if depth.shape[0] != T:
        raise ValueError(f"DA3 geometry has {depth.shape[0]} frames but the clip has {T}. "
                         "Run Depth Anything 3 on the same images you feed this effect.")
    return depth.float(), (sky.float() if sky is not None else None)


# --------------------------------------------------------------------------- utils

class VideoDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabVideoDepth",
            display_name="Video Depth (Depth Anything 3)",
            category=UTIL,
            search_aliases=["video depth", "depth anything 3", "da3", "lidar", "depth video"],
            description="Runs ComfyUI's native Depth Anything 3 over a whole clip with a progress bar, optionally on every "
                        "Nth frame (interpolating in between), then removes frame-to-frame flicker. Outputs standard "
                        "DA3_GEOMETRY plus a depth preview.",
            inputs=[
                io.Custom("DA3_MODEL").Input("da3_model", tooltip="From Load Depth Anything 3 (Mono-Large recommended; bf16 on CPU)."),
                io.Image.Input("images"),
                io.Int.Input("resolution", default=504, min=140, max=2520, step=14),
                io.Int.Input("frame_stride", default=1, min=1, max=6,
                             tooltip="Estimate depth on every Nth frame and interpolate the rest. 2 roughly halves the time."),
                io.Float.Input("temporal_smoothing", default=0.6, min=0.0, max=0.95, step=0.05),
            ],
            outputs=[DA3Geometry.Output("da3_geometry"), io.Image.Output("depth_preview")],
        )

    @classmethod
    def execute(cls, da3_model, images, resolution, frame_stride, temporal_smoothing) -> io.NodeOutput:
        from comfy_extras.nodes_depth_anything_3 import _run_da3
        imgs = images[..., :3]
        T = imgs.shape[0]
        keys = list(range(0, T, frame_stride))
        if keys[-1] != T - 1:
            keys.append(T - 1)
        bar = comfy.utils.ProgressBar(len(keys))
        kd, ks = {}, {}
        for n in range(0, len(keys), 4):
            chunk = keys[n:n + 4]
            depth, _, sky = _run_da3(da3_model, imgs[chunk], resolution)
            for j, k in enumerate(chunk):
                kd[k] = depth[j]
                ks[k] = sky[j] if sky is not None else None
            bar.update_absolute(min(n + 4, len(keys)), len(keys))
        depth = torch.empty(T, *kd[keys[0]].shape)
        sky = None if ks[keys[0]] is None else torch.empty_like(depth)
        for a, b in zip(keys[:-1], keys[1:]):
            for i in range(a, b + 1):
                w = 0.0 if b == a else (i - a) / (b - a)
                depth[i] = kd[a] * (1 - w) + kd[b] * w
                if sky is not None:
                    sky[i] = ks[a] * (1 - w) + ks[b] * w
        if len(keys) == 1:
            depth[0] = kd[keys[0]]
            if sky is not None:
                sky[0] = ks[keys[0]]
        depth = G.stabilize_depth(depth, sky, temporal=temporal_smoothing)
        geometry = {"depth": depth.contiguous(), "image": imgs.cpu(), "mode": "mono"}
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        d0, d1 = G.robust_range(depth[:: max(1, T // 8)], None if sky is None else sky[:: max(1, T // 8)] < 0.5)
        prev = 1.0 - G.normalize_depth(depth, d0, d1)
        return io.NodeOutput(geometry, prev[..., None].expand(-1, -1, -1, 3).contiguous())


class StabilizeVideoDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabStabilizeDepth",
            display_name="Stabilize Video Depth",
            category=UTIL,
            search_aliases=["depth flicker", "temporal depth", "video depth", "lidar"],
            description="Removes frame-to-frame flicker from per-frame Depth Anything 3 video depth: "
                        "aligns every frame's scale to a running reference and blends static regions over time.",
            inputs=[
                DA3Geometry.Input("da3_geometry"),
                io.Float.Input("temporal_smoothing", default=0.6, min=0.0, max=0.95, step=0.05,
                               tooltip="How strongly static areas keep the previous frame's depth. 0 = scale alignment only."),
            ],
            outputs=[DA3Geometry.Output("da3_geometry")],
        )

    @classmethod
    def execute(cls, da3_geometry, temporal_smoothing) -> io.NodeOutput:
        g = dict(da3_geometry)
        g["depth"] = G.stabilize_depth(g["depth"].float(), g.get("sky"), temporal=temporal_smoothing)
        return io.NodeOutput(g)


class CleanPlate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabCleanPlate",
            display_name="Clean Plate (static camera)",
            category=UTIL,
            search_aliases=["background plate", "remove person", "temporal median", "empty background"],
            description="Recovers the empty background of a tripod shot as the temporal median of the clip, "
                        "re-taken without the subject for `refine` passes.",
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks", optional=True, tooltip="Optional subject masks to ignore. Without them the subject is found automatically."),
                io.Int.Input("max_samples", default=64, min=3, max=512, tooltip="Frames sampled across the clip."),
                io.Int.Input("refine", default=2, min=0, max=4, tooltip="Extra passes that remove a lingering subject's ghost."),
            ],
            outputs=[io.Image.Output("plate")],
        )

    @classmethod
    def execute(cls, images, max_samples, refine, masks=None) -> io.NodeOutput:
        plate = M.clean_plate(images[..., :3].float(), max_samples=max_samples, masks=masks, refine=refine,
                              progress=_pbar(max(1, refine * min(images.shape[0], max_samples))))
        return io.NodeOutput(plate[None])


class SubjectMatte(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabSubjectMatte",
            display_name="Subject Matte (clean plate)",
            category=UTIL,
            search_aliases=["matte", "mask person", "rotoscope", "difference key", "segment subject"],
            description="Fast subject masks for tripod shots: whatever differs from the clean plate, cleaned up "
                        "and edge-refined with a guided filter. ~0.1 s per 720p frame on a CPU.",
            inputs=[
                io.Image.Input("images"),
                io.Image.Input("plate"),
                io.Float.Input("threshold", default=0.10, min=0.01, max=0.5, step=0.005),
                io.Float.Input("softness", default=1.0, min=0.0, max=4.0, step=0.1),
                io.Int.Input("grow", default=2, min=0, max=32),
                io.Int.Input("max_subjects", default=2, min=1, max=8),
            ],
            outputs=[io.Mask.Output("masks")],
        )

    @classmethod
    def execute(cls, images, plate, threshold, softness, grow, max_subjects) -> io.NodeOutput:
        imgs = images[..., :3].float()
        m = M.difference_matte(imgs, plate[0, ..., :3].float(), threshold=threshold, softness=softness, grow=grow,
                               max_subjects=max_subjects, progress=_pbar(imgs.shape[0]))
        return io.NodeOutput(m)


class BiRefNetVideoMatte(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabBiRefNetMatte",
            display_name="Subject Matte (BiRefNet video)",
            category=UTIL,
            search_aliases=["birefnet", "remove background video", "video matte", "segment person"],
            description="Runs ComfyUI's BiRefNet background-removal model over a video for moving-camera shots. "
                        "Uses bf16 on AMX / AVX512-BF16 CPUs (about 1.5x faster) and can key every Nth frame.",
            inputs=[
                io.BackgroundRemoval.Input("bg_removal_model", tooltip="From Load Background Removal Model (birefnet.safetensors)."),
                io.Image.Input("images"),
                io.Int.Input("resolution", default=768, min=256, max=2048, step=32,
                             tooltip="Model input size. 512 is fast on CPU, 1024 is the model's native size."),
                io.Int.Input("stride", default=1, min=1, max=8, tooltip="Run on every Nth frame and blend masks in between."),
            ],
            outputs=[io.Mask.Output("masks")],
        )

    @classmethod
    def execute(cls, bg_removal_model, images, resolution, stride) -> io.NodeOutput:
        imgs = images[..., :3].float()
        keys = (imgs.shape[0] + stride - 1) // stride + 1
        m = M.birefnet_matte(bg_removal_model, imgs, resolution=resolution, stride=stride, progress=_pbar(keys))
        return io.NodeOutput(m)


# --------------------------------------------------------------------------- effects

class LidarScan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabScan",
            display_name="LiDAR Scan",
            category=FX,
            search_aliases=["lidar", "point cloud", "scan", "3d scan", "lidar transition", "depth effect", "higgsfield"],
            description="A laser sweep turns live footage into a glowing 3D point cloud, the camera drifts through it, "
                        "then the sweep hands the shot back to video. Uses Depth Anything 3 geometry.",
            inputs=[
                io.Image.Input("images"),
                DA3Geometry.Input("da3_geometry", tooltip="From Run Depth Anything 3 on the same frames (optionally stabilized)."),
                _fps_input(),
                io.Combo.Input("style", options=fx_lidar.STYLES, default="cyan"),
                io.Color.Input("color", default="#4de1ff", tooltip="Point colour when style is 'custom'."),
                io.Combo.Input("sweep", options=fx_lidar.SWEEPS, default="depth (near to far)"),
                io.Float.Input("scan_start", default=0.6, min=0.0, max=600.0, step=0.05, tooltip="Seconds."),
                io.Float.Input("scan_duration", default=1.6, min=0.1, max=30.0, step=0.05, tooltip="Seconds."),
                io.Float.Input("hold", default=2.4, min=0.0, max=60.0, step=0.05, tooltip="Seconds in full point-cloud mode."),
                io.Combo.Input("camera", options=fx_lidar.CAMERAS, default="orbit"),
                io.Float.Input("camera_amount", default=0.6, min=0.0, max=2.0, step=0.05),
                io.Boolean.Input("return_to_video", default=True),
                io.Int.Input("density", default=4, min=1, max=16, tooltip="Scanline spacing in pixels (lower = denser)."),
                io.Float.Input("glow", default=1.0, min=0.0, max=3.0, step=0.05),
                io.Float.Input("fov", default=60.0, min=20.0, max=120.0, step=1.0, advanced=True,
                               tooltip="Horizontal field of view used to lift depth into 3D."),
                io.Boolean.Input("drop_sky", default=True, advanced=True),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, da3_geometry, fps, style, color, sweep, scan_start, scan_duration, hold, camera,
                camera_amount, return_to_video, density, glow, fov, drop_sky) -> io.NodeOutput:
        imgs = images[..., :3].float()
        depth, sky = _depth_from(da3_geometry, imgs.shape[0])
        out = fx_lidar.lidar_scan(imgs, depth, sky, fps, style=style, color=color, sweep=sweep, scan_start=scan_start,
                                  scan_duration=scan_duration, hold=hold, camera=camera, camera_amount=camera_amount,
                                  return_to_video=return_to_video, density=density, glow=glow, fov=fov,
                                  drop_sky=drop_sky, progress=_pbar(imgs.shape[0]))
        return io.NodeOutput(out)


class Clones(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabClones",
            display_name="Clones",
            category=FX,
            search_aliases=["clones", "infinite clones", "trail", "echo", "multi exposure", "chronophotography", "higgsfield"],
            description="Copies of the subject trail behind it, or freeze along its path like a multi-exposure photo.",
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                _fps_input(),
                io.Combo.Input("mode", options=fx_clones.MODES, default="trail"),
                io.Combo.Input("look", options=fx_clones.LOOKS, default="natural"),
                io.Int.Input("count", default=6, min=1, max=64),
                io.Float.Input("interval", default=0.12, min=0.02, max=5.0, step=0.01, tooltip="Seconds between clones."),
                io.Float.Input("start", default=0.0, min=0.0, max=600.0, step=0.05),
                io.Float.Input("end", default=0.0, min=0.0, max=600.0, step=0.05, tooltip="0 = until the end of the clip."),
                io.Float.Input("fade", default=0.65, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, masks, fps, mode, look, count, interval, start, end, fade) -> io.NodeOutput:
        imgs = images[..., :3].float()
        out = fx_clones.clones(imgs, masks, fps, mode=mode, look=look, count=count, interval=interval, start=start,
                               end=end, fade=fade, progress=_pbar(imgs.shape[0]))
        return io.NodeOutput(out)


class Vanish(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabVanish",
            display_name="Vanish (Disintegrate)",
            category=FX,
            search_aliases=["vanish", "disintegrate", "thanos", "dust", "particles", "dissolve", "higgsfield"],
            description="The subject crumbles into particles carried off by a curl-noise wind, revealing the clean plate.",
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                io.Image.Input("plate", tooltip="Clean background (Clean Plate node)."),
                _fps_input(),
                io.Float.Input("start", default=1.0, min=0.0, max=600.0, step=0.05),
                io.Float.Input("duration", default=1.8, min=0.1, max=30.0, step=0.05),
                io.Combo.Input("direction", options=fx_vanish.DIRECTIONS, default="left to right"),
                io.Combo.Input("particles", options=fx_vanish.PARTICLES, default="ash"),
                io.Color.Input("color", default="#ff7a2f", tooltip="Glow colour for embers / sparks."),
                io.Float.Input("wind_angle", default=-20.0, min=-180.0, max=180.0, step=1.0),
                io.Float.Input("wind_speed", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Float.Input("turbulence", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Float.Input("density", default=0.45, min=0.02, max=1.0, step=0.01),
                io.Boolean.Input("freeze_subject", default=True, tooltip="Hold the subject still while it crumbles."),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, advanced=True),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, masks, plate, fps, start, duration, direction, particles, color, wind_angle, wind_speed,
                turbulence, density, freeze_subject, seed) -> io.NodeOutput:
        imgs = images[..., :3].float()
        out = fx_vanish.vanish(imgs, masks, plate[0, ..., :3].float(), fps, start=start, duration=duration,
                               direction=direction, particles=particles, color=color, wind_angle=wind_angle,
                               wind_speed=wind_speed, turbulence=turbulence, density=density,
                               freeze_subject=freeze_subject, seed=seed, progress=_pbar(imgs.shape[0]))
        return io.NodeOutput(out)


class BulletTime(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LidarLabBulletTime",
            display_name="Bullet Time 2.5D",
            category=FX,
            search_aliases=["bullet time", "matrix", "freeze frame", "orbit", "frozen in motion", "higgsfield"],
            description="Slows into the decisive moment (auto: the top of a jump), freezes it and swings the camera "
                        "around the subject using Depth Anything 3 geometry, then lets time run again. "
                        "The output is longer than the input by the freeze duration.",
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                DA3Geometry.Input("da3_geometry"),
                _fps_input(),
                io.Image.Input("plate", optional=True, tooltip="Clean plate for tripod shots; without it the background behind the subject is filled smoothly."),
                io.Float.Input("freeze_at", default=-1.0, min=-1.0, max=600.0, step=0.05, tooltip="Seconds; -1 = automatic (top of the jump)."),
                io.Float.Input("duration", default=2.2, min=0.3, max=20.0, step=0.05),
                io.Float.Input("angle", default=14.0, min=-60.0, max=60.0, step=0.5),
                io.Float.Input("zoom", default=0.35, min=0.0, max=1.5, step=0.05),
                io.Float.Input("ramp", default=0.5, min=0.0, max=5.0, step=0.05, tooltip="Seconds of slow motion into and out of the freeze."),
                io.Boolean.Input("flash", default=True),
                io.Boolean.Input("motion_blur", default=True),
                io.Float.Input("relief", default=0.25, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="How much of the subject's own depth relief to keep (0 = flat cut-out)."),
                io.Float.Input("fov", default=60.0, min=20.0, max=120.0, step=1.0, advanced=True),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, masks, da3_geometry, fps, freeze_at, duration, angle, zoom, ramp, flash, motion_blur,
                relief, fov, plate=None) -> io.NodeOutput:
        imgs = images[..., :3].float()
        depth, _ = _depth_from(da3_geometry, imgs.shape[0])
        out = fx_bullet.bullet_time(imgs, masks, depth, fps, plate=None if plate is None else plate[0, ..., :3].float(),
                                    freeze_at=freeze_at, duration=duration, angle=angle, zoom=zoom, ramp=ramp,
                                    flash=flash, motion_blur=motion_blur, relief=relief, fov=fov,
                                    progress=_pbar(max(2, int(round(duration * fps)))))
        return io.NodeOutput(out)


NODES = [VideoDepth, StabilizeVideoDepth, CleanPlate, SubjectMatte, BiRefNetVideoMatte, LidarScan, Clones, Vanish, BulletTime]


class LidarLabExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES
