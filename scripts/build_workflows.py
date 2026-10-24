#!/usr/bin/env python3
"""Generate the LiDAR Lab preset workflows in ComfyUI API format.

Every preset is a plain graph of core ComfyUI nodes (Load Video, Trim Video,
Depth Anything 3, BiRefNet, Create/Save Video) plus LiDAR Lab effect nodes.
The gallery app loads these files, swaps in the user's clip and knobs, and
runs them through the Comfy SDK.

    python scripts/build_workflows.py          # writes workflows/api/*.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "workflows", "api")

DA3_MODEL = "depth_anything_3_mono_large.safetensors"
BIREFNET = "birefnet.safetensors"


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, class_type, title=None, **inputs):
        nid = str(len(self.nodes) + 1)
        node = {"class_type": class_type, "inputs": inputs}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[nid] = node
        return nid

    def to_json(self):
        return self.nodes


def head(g, max_side=1280, duration=8.0):
    """Load a clip, trim it, split it into frames and cap the resolution."""
    load = g.add("LoadVideo", "Input clip", file="lidarlab_sample.mp4")
    trim = g.add("Video Slice", "Trim", video=[load, 0], start_time=0.0, duration=duration, strict_duration=False)
    comp = g.add("GetVideoComponents", "Frames + audio + fps", video=[trim, 0])
    scale = g.add("ImageScaleToMaxDimension", "Working resolution", image=[comp, 0], upscale_method="area",
                  largest_size=max_side)
    return load, comp, scale


def depth(g, images, stabilize=True):
    m = g.add("LoadDA3Model", "Depth Anything 3", model_name=DA3_MODEL, weight_dtype="bf16")
    d = g.add("DA3Inference", "Video depth", da3_model=[m, 0], image=[images, 0], resolution=504,
              resize_method="upper_bound_resize", mode="mono")
    if stabilize:
        d = g.add("LidarLabStabilizeDepth", "Stabilize depth", da3_geometry=[d, 0], temporal_smoothing=0.6)
    return d


def tail(g, frames, comp, prefix, audio=True):
    vid = {"images": [frames, 0], "fps": [comp, 2]}
    if audio:
        vid["audio"] = [comp, 1]
    v = g.add("CreateVideo", "Encode", **vid)
    return g.add("SaveVideo", "Save", video=[v, 0], filename_prefix=f"lidarlab/{prefix}", format="auto", codec="auto")


def static_matte(g, images):
    plate = g.add("LidarLabCleanPlate", "Clean plate", images=[images, 0], max_samples=64, refine=2)
    masks = g.add("LidarLabSubjectMatte", "Subject matte", images=[images, 0], plate=[plate, 0], threshold=0.10,
                  softness=1.0, grow=2, max_subjects=2)
    return plate, masks


def any_camera_matte(g, images, resolution=768, stride=2):
    bg = g.add("LoadBackgroundRemovalModel", "BiRefNet", bg_removal_name=BIREFNET)
    return g.add("LidarLabBiRefNetMatte", "Subject matte (any camera)", bg_removal_model=[bg, 0],
                 images=[images, 0], resolution=resolution, stride=stride)


def lidar_scan():
    g = Graph()
    _, comp, img = head(g)
    d = depth(g, img)
    fx = g.add("LidarLabScan", "LiDAR Scan", images=[img, 0], da3_geometry=[d, 0], fps=[comp, 2], style="cyan",
               color="#4de1ff", sweep="depth (near to far)", scan_start=0.6, scan_duration=1.6, hold=2.4,
               camera="orbit", camera_amount=0.6, return_to_video=True, density=4, glow=1.0, fov=60.0, drop_sky=True)
    tail(g, fx, comp, "lidar_scan")
    return g


def clones(any_camera=False):
    g = Graph()
    _, comp, img = head(g)
    masks = any_camera_matte(g, img) if any_camera else static_matte(g, img)[1]
    fx = g.add("LidarLabClones", "Clones", images=[img, 0], masks=[masks, 0], fps=[comp, 2], mode="stamp",
               look="rainbow", count=10, interval=0.25, start=0.0, end=0.0, fade=0.65)
    tail(g, fx, comp, "clones")
    return g


def vanish():
    g = Graph()
    _, comp, img = head(g)
    plate, masks = static_matte(g, img)
    fx = g.add("LidarLabVanish", "Vanish", images=[img, 0], masks=[masks, 0], plate=[plate, 0], fps=[comp, 2],
               start=0.8, duration=2.0, direction="left to right", particles="ash", color="#ff7a2f",
               wind_angle=-20.0, wind_speed=1.0, turbulence=1.0, density=0.45, freeze_subject=True, seed=0)
    tail(g, fx, comp, "vanish")
    return g


def bullet_time():
    g = Graph()
    _, comp, img = head(g)
    masks = any_camera_matte(g, img)
    d = depth(g, img)
    fx = g.add("LidarLabBulletTime", "Bullet Time", images=[img, 0], masks=[masks, 0], da3_geometry=[d, 0],
               fps=[comp, 2], freeze_at=-1.0, duration=2.2, angle=14.0, zoom=0.35, ramp=0.5, flash=True,
               motion_blur=True, relief=0.25, fov=60.0)
    tail(g, fx, comp, "bullet_time", audio=False)  # the freeze changes the clip length
    return g


PRESETS = {
    "lidar_scan": lidar_scan,
    "clones": clones,
    "clones_any_camera": lambda: clones(any_camera=True),
    "vanish": vanish,
    "bullet_time": bullet_time,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in PRESETS.items():
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as f:
            json.dump(fn().to_json(), f, indent=2)
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
