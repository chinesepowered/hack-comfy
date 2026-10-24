"""LiDAR Lab gallery app.

A small FastAPI backend that turns the preset workflows in workflows/api/ into
one-click effects. Jobs run on ComfyUI through the Comfy SDK (comfy-sdk),
talking to a local ComfyUI via comfy-api-proxy (or to a Comfy API deployment:
only COMFY_BASE_URL / COMFY_API_KEY change).

    COMFY_BASE_URL=http://127.0.0.1:8189 uvicorn app.server:app --port 7860
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
from comfy_sdk import AsyncComfy, JobFailed, Progress, StatusChange  # noqa: E402

from .presets import BY_ID, PRESETS  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORKFLOWS = REPO / "workflows" / "api"
MEDIA = REPO / "docs" / "media"
SAMPLES = ROOT / "samples"
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
for d in (UPLOADS, OUTPUTS, SAMPLES):
    d.mkdir(parents=True, exist_ok=True)

MAX_SECONDS = float(os.environ.get("LIDARLAB_MAX_SECONDS", "8"))

# rough relative cost of each node type, for an honest progress bar on CPU
WEIGHTS = {
    "DA3Inference": 6.0, "LidarLabVideoDepth": 6.0, "LidarLabBiRefNetMatte": 10.0, "LidarLabSubjectMatte": 2.0, "LidarLabCleanPlate": 1.5,
    "LidarLabScan": 2.0, "LidarLabClones": 1.5, "LidarLabVanish": 2.0, "LidarLabBulletTime": 4.0,
    "LidarLabStabilizeDepth": 0.3, "CreateVideo": 0.3, "SaveVideo": 1.0, "GetVideoComponents": 0.5,
}

app = FastAPI(title="LiDAR Lab")
JOBS: dict[str, dict] = {}
_client: AsyncComfy | None = None


async def client() -> AsyncComfy:
    global _client
    if _client is None:
        _client = AsyncComfy(timeout=300)
    return _client


def build_workflow(preset: dict, options: dict, start: float, duration: float, any_camera: bool) -> dict:
    name = preset.get("workflow_any_camera") if any_camera and preset.get("workflow_any_camera") else preset["workflow"]
    graph = json.loads((WORKFLOWS / name).read_text())
    by_class: dict[str, list[str]] = {}
    for nid, node in graph.items():
        by_class.setdefault(node["class_type"], []).append(nid)
    for nid in by_class.get("Video Slice", []):
        graph[nid]["inputs"]["start_time"] = max(0.0, float(start))
        graph[nid]["inputs"]["duration"] = min(MAX_SECONDS, max(0.5, float(duration)))
    known = {o["id"]: o for o in preset["options"]}
    for key, value in options.items():
        opt = known.get(key)
        if opt is None:
            continue
        if "choices" in opt and value not in opt["choices"]:
            raise HTTPException(400, f"bad value for {key}")
        if opt.get("type") == "range":
            value = min(opt["max"], max(opt["min"], float(value)))
            if isinstance(opt["step"], int) and float(opt["step"]).is_integer():
                value = int(round(value))
        for nid in by_class.get(opt["node"], []):
            graph[nid]["inputs"][opt["input"]] = value
    return graph


def stage_plan(graph: dict) -> dict:
    total = sum(WEIGHTS.get(n["class_type"], 0.2) for n in graph.values())
    return {"total": total, "weights": {nid: WEIGHTS.get(n["class_type"], 0.2) for nid, n in graph.items()},
            "titles": {nid: n.get("_meta", {}).get("title", n["class_type"]) for nid, n in graph.items()}}


async def run_job(job_id: str, graph: dict, upload: Path):
    job = JOBS[job_id]
    plan = stage_plan(graph)
    done: set[str] = set()
    try:
        c = await client()
        wf = c.workflows.from_json(copy.deepcopy(graph))
        for nid, node in graph.items():
            if node["class_type"] == "LoadVideo":
                wf.set_input(nid, "file", c.assets.from_file(str(upload)))
        job["stage"] = "Uploading clip to ComfyUI"
        handle = await c.submit(wf)
        job["comfy_job"] = handle.id
        job["status"] = "queued"
        current = None
        async for ev in handle.events():
            if isinstance(ev, StatusChange):
                job["status"] = ev.status
                if ev.queue_position is not None:
                    job["stage"] = f"Waiting in queue (#{ev.queue_position})"
                elif ev.status == "running" and job["stage"] in ("Uploading", "Starting"):
                    job["stage"] = "Reading frames"
            elif isinstance(ev, Progress):
                node = ev.current_node
                if node != current and current is not None:
                    done.add(current)
                current = node
                frac = (ev.step / ev.steps) if (ev.step is not None and ev.steps) else 0.0
                spent = sum(plan["weights"].get(n, 0.2) for n in done)
                if node:
                    spent += plan["weights"].get(node, 0.2) * frac
                    title = plan["titles"].get(node, node)
                    job["stage"] = f"{title} — {ev.step}/{ev.steps}" if ev.steps else title
                job["progress"] = max(job["progress"], min(0.99, spent / plan["total"]))
        await handle.refresh()
        if handle.status != "succeeded":
            raise JobFailed(f"job ended {handle.status}", error=handle.error)
        save_nodes = [nid for nid, n in graph.items() if n["class_type"] == "SaveVideo"]
        outs = [o for nid in save_nodes for o in handle.get_outputs(nid)]
        if not outs:
            raise RuntimeError("workflow produced no video")
        dest = OUTPUTS / f"{job_id}.mp4"
        await outs[0].to_file(str(dest))
        job.update(status="done", progress=1.0, stage="Done", video=f"/api/jobs/{job_id}/video",
                   finished=time.time())
    except Exception as e:  # surface the reason in the UI
        err = getattr(e, "error", None)
        job.update(status="error", stage="Failed", error=str(err or e)[:2000], finished=time.time())


def _thumb(video: Path) -> None:
    jpg = video.with_suffix(".jpg")
    if not jpg.exists():
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.6", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=320:-2", str(jpg)], check=False)


@app.get("/api/presets")
async def presets():
    samples = sorted(p.name for p in SAMPLES.glob("*.mp4"))
    for name in samples:
        _thumb(SAMPLES / name)
    return {"presets": PRESETS, "samples": samples, "max_seconds": MAX_SECONDS}


@app.post("/api/jobs")
async def create_job(preset: str = Form(...), options: str = Form("{}"), start: float = Form(0.0),
                     duration: float = Form(5.0), any_camera: bool = Form(False), sample: str = Form(""),
                     file: UploadFile | None = File(None)):
    p = BY_ID.get(preset)
    if p is None:
        raise HTTPException(404, "unknown preset")
    job_id = uuid.uuid4().hex[:12]
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower() or ".mp4"
        if ext not in (".mp4", ".mov", ".webm", ".mkv", ".m4v"):
            raise HTTPException(400, "upload a video file")
        upload = UPLOADS / f"{job_id}{ext}"
        with upload.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    elif sample:
        src = (SAMPLES / Path(sample).name)
        if not src.exists():
            raise HTTPException(404, "unknown sample")
        upload = UPLOADS / f"{job_id}{src.suffix}"
        shutil.copyfile(src, upload)
    else:
        raise HTTPException(400, "attach a clip or pick a sample")
    graph = build_workflow(p, json.loads(options or "{}"), start, duration, any_camera)
    JOBS[job_id] = {"id": job_id, "preset": preset, "status": "submitting", "progress": 0.0, "stage": "Uploading",
                    "created": time.time(), "workflow": graph}
    asyncio.create_task(run_job(job_id, graph, upload))
    return {"id": job_id}


def _public(job):
    return {k: v for k, v in job.items() if k != "workflow"} | {"elapsed": round(time.time() - job["created"], 1)}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404)
    return _public(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)

    async def stream():
        last = None
        while True:
            job = JOBS[job_id]
            snap = _public(job)
            key = (snap["status"], round(snap["progress"], 3), snap.get("stage"))
            if key != last:
                yield f"data: {json.dumps(snap)}\n\n"
                last = key
            else:
                yield ": keepalive\n\n"
            if job["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    path = OUTPUTS / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4", filename=f"lidarlab-{JOBS.get(job_id, {}).get('preset', 'fx')}-{job_id}.mp4")


@app.get("/api/jobs/{job_id}/workflow")
async def job_workflow(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404)
    return JSONResponse(job["workflow"], headers={"Content-Disposition": f'attachment; filename="lidarlab-{job["preset"]}.json"'})


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/media", StaticFiles(directory=MEDIA, check_dir=False), name="media")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
