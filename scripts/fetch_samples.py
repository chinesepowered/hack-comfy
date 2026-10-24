#!/usr/bin/env python3
"""Fetch the demo clips used by the gallery app and the example workflows.

The clips are free stock footage from Mixkit (https://mixkit.co, Mixkit Stock
Video Free License). The license lets you use them in projects but not
redistribute them as-is, so they are downloaded on demand instead of living in
this repository. Needs ffmpeg on PATH.

    python scripts/fetch_samples.py [--comfyui /path/to/ComfyUI]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "app", "samples")

# name: (mixkit id, start seconds, duration seconds, description)
CLIPS = {
    "city_skyline": (30403, 0.0, 7.0, "Aerial view of a city skyline with a cathedral"),
    "street_walk": (4855, 0.0, 7.0, "Man walking across a city square"),
    "dancer_studio": (452, 6.5, 5.0, "Breakdancer in a white studio (tripod)"),
    "dancer_kick": (452, 0.0, 5.0, "Breakdancer in a white studio (tripod)"),
    "parkour_flip": (21756, 2.0, 4.0, "Parkour backflip off a wall"),
    "rock_field": (2789, 0.0, 7.0, "Drone shot over boulders in a field"),
    "night_city": (16125, 0.0, 6.5, "Aerial night view of a city"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", help="also copy a sample into ComfyUI/input as lidarlab_sample.mp4")
    a = ap.parse_args()
    os.makedirs(SAMPLES, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, (mid, start, dur, desc) in CLIPS.items():
            out = os.path.join(SAMPLES, f"{name}.mp4")
            if os.path.exists(out):
                print(f"ok   {name}.mp4 (already there)")
                continue
            src = os.path.join(tmp, f"{mid}.mp4")
            if not os.path.exists(src):
                url = f"https://assets.mixkit.co/videos/{mid}/{mid}-720.mp4"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (LiDAR Lab sample fetcher)"})
                with urllib.request.urlopen(req, timeout=120) as r, open(src, "wb") as f:
                    shutil.copyfileobj(r, f)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", src, "-an",
                            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out], check=True)
            print(f"ok   {name}.mp4  ({desc}; mixkit.co video {mid})")
    if a.comfyui:
        dst = os.path.join(a.comfyui, "input", "lidarlab_sample.mp4")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(SAMPLES, "street_walk.mp4"), dst)
        print(f"ok   copied street_walk.mp4 to {dst}")


if __name__ == "__main__":
    main()
