#!/usr/bin/env python3
"""Download the open-weight models LiDAR Lab uses into a ComfyUI models folder.

    python scripts/download_models.py /path/to/ComfyUI            # depth + matting
    python scripts/download_models.py /path/to/ComfyUI --small    # DA3-Small instead of Mono-Large

All files come from Comfy-Org's official Hugging Face mirrors:
  * Depth Anything 3 Mono-Large (Apache-2.0)  -> models/geometry_estimation/
  * BiRefNet (MIT)                            -> models/background_removal/
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

FILES = {
    "geometry_estimation/depth_anything_3_mono_large.safetensors":
        "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_mono_large.safetensors",
    "background_removal/birefnet.safetensors":
        "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors",
}
SMALL = {
    "geometry_estimation/depth_anything_3_small.safetensors":
        "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_small.safetensors",
}


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"ok   {os.path.relpath(dest)} (already there)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r     {os.path.basename(dest)} {done / total:6.1%} of {total / 1e9:.2f} GB")
                sys.stdout.flush()
    os.replace(tmp, dest)
    print(f"\rok   {os.path.relpath(dest)}" + " " * 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comfyui_dir")
    ap.add_argument("--small", action="store_true", help="also fetch Depth Anything 3 Small (fast, lower detail)")
    a = ap.parse_args()
    models = os.path.join(os.path.abspath(a.comfyui_dir), "models")
    todo = dict(FILES)
    if a.small:
        todo.update(SMALL)
    for rel, url in todo.items():
        fetch(url, os.path.join(models, rel))


if __name__ == "__main__":
    main()
