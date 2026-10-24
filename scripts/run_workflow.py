#!/usr/bin/env python3
"""Run a LiDAR Lab API workflow against a local ComfyUI over its HTTP API.

    python scripts/run_workflow.py workflows/api/lidar_scan.json my_clip.mp4 \
        --set LidarLabScan.style=thermal --set "Video Slice.start_time=1.5" --server http://127.0.0.1:8188

`--set NODE.INPUT=VALUE` overrides any input (VALUE is parsed as JSON when possible). NODE is a node id
or a class name such as LidarLabScan (applies to every node of that class).
Prints the output files and the run time. Only uses the standard library.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import urllib.request
import uuid


def _post(url, data, headers):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def upload(server, path):
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        payload = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + payload + (
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    r = _post(f"{server}/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return r["name"] if not r.get("subfolder") else f"{r['subfolder']}/{r['name']}"


def run(server, workflow, timeout=3600, log=print):
    pid = _post(f"{server}/prompt", json.dumps({"prompt": workflow, "client_id": "lidarlab-cli"}).encode(),
                {"Content-Type": "application/json"})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{server}/history/{pid}", timeout=60) as r:
            h = json.load(r)
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(json.dumps(msgs[-1][1] if msgs else status)[:4000])
            outs = []
            for node_out in entry.get("outputs", {}).values():
                for key in ("images", "videos", "gifs", "video"):
                    for item in node_out.get(key, []) or []:
                        if isinstance(item, dict) and "filename" in item:
                            outs.append(os.path.join(item.get("subfolder", ""), item["filename"]))
            return outs, time.time() - t0
        time.sleep(2)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("clip")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    with open(a.workflow) as f:
        wf = json.load(f)
    name = upload(a.server, a.clip)
    loaders = [k for k, v in wf.items() if v["class_type"] == "LoadVideo"]
    for k in loaders:
        wf[k]["inputs"]["file"] = name
    for item in a.set:
        key, val = item.split("=", 1)
        nid, inp = key.split(".", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        targets = [nid] if nid in wf else [k for k, v in wf.items() if v["class_type"] == nid]
        if not targets:
            raise SystemExit(f"no node {nid!r} in {a.workflow}")
        for t in targets:
            wf[t]["inputs"][inp] = val
    outs, dt = run(a.server, wf)
    print(f"done in {dt:.1f}s")
    for o in outs:
        print("output:", o)


if __name__ == "__main__":
    main()
