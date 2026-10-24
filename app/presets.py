"""Effect presets exposed by the gallery app.

Each preset points at a ComfyUI API workflow in workflows/api/ and lists the
knobs the app shows. Knobs address nodes by class_type, so the workflow files
can be regenerated without touching this table.
"""

PRESETS = [
    {
        "id": "lidar_scan",
        "title": "LiDAR Scan",
        "tagline": "A laser sweep turns your shot into a live 3D point cloud, the camera drifts through it, then it snaps back.",
        "higgsfield": "Lidar transition · Blue depth",
        "workflow": "lidar_scan.json",
        "preview": "lidar_scan.mp4",
        "best_with": "Any clip: drone shots, streets, rooms, people.",
        "options": [
            {"id": "style", "label": "Look", "node": "LidarLabScan", "input": "style",
             "choices": ["cyan", "rgb", "thermal", "matrix", "gold"], "default": "cyan"},
            {"id": "camera", "label": "Camera", "node": "LidarLabScan", "input": "camera",
             "choices": ["orbit", "push and orbit", "dolly in", "crane up", "none"], "default": "orbit"},
            {"id": "sweep", "label": "Sweep", "node": "LidarLabScan", "input": "sweep",
             "choices": ["depth (near to far)", "depth (far to near)", "left to right", "top to bottom", "radial"],
             "default": "depth (near to far)"},
            {"id": "density", "label": "Point spacing", "node": "LidarLabScan", "input": "density",
             "type": "range", "min": 2, "max": 8, "step": 1, "default": 4},
        ],
    },
    {
        "id": "clones",
        "title": "Clones",
        "tagline": "Copies of you trail behind every move, or freeze along your path like a multi-exposure photo.",
        "higgsfield": "Clones · Infinite clones",
        "workflow": "clones.json",
        "workflow_any_camera": "clones_any_camera.json",
        "preview": "clones.mp4",
        "best_with": "A tripod shot of someone moving: dance, tricks, a walk across the frame.",
        "options": [
            {"id": "mode", "label": "Mode", "node": "LidarLabClones", "input": "mode",
             "choices": ["stamp", "trail"], "default": "stamp"},
            {"id": "look", "label": "Look", "node": "LidarLabClones", "input": "look",
             "choices": ["rainbow", "natural", "ghost glow", "lidar"], "default": "rainbow"},
            {"id": "count", "label": "Clones", "node": "LidarLabClones", "input": "count",
             "type": "range", "min": 2, "max": 24, "step": 1, "default": 10},
            {"id": "interval", "label": "Spacing (s)", "node": "LidarLabClones", "input": "interval",
             "type": "range", "min": 0.05, "max": 0.6, "step": 0.01, "default": 0.25},
        ],
    },
    {
        "id": "vanish",
        "title": "Vanish",
        "tagline": "Crumble into ash, embers or pixels that blow away on the wind.",
        "higgsfield": "Vanish · Particles",
        "workflow": "vanish.json",
        "preview": "vanish.mp4",
        "best_with": "A tripod shot with the subject in frame, ideally moving around a little.",
        "options": [
            {"id": "particles", "label": "Particles", "node": "LidarLabVanish", "input": "particles",
             "choices": ["ash", "embers", "sparks", "pixels"], "default": "ash"},
            {"id": "direction", "label": "Direction", "node": "LidarLabVanish", "input": "direction",
             "choices": ["left to right", "right to left", "top to bottom", "bottom to top", "outside in"],
             "default": "left to right"},
            {"id": "start", "label": "Start (s)", "node": "LidarLabVanish", "input": "start",
             "type": "range", "min": 0.0, "max": 6.0, "step": 0.1, "default": 0.8},
            {"id": "duration", "label": "Duration (s)", "node": "LidarLabVanish", "input": "duration",
             "type": "range", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0},
        ],
    },
    {
        "id": "bullet_time",
        "title": "Bullet Time",
        "tagline": "Freeze the top of the jump and swing the camera around it, Matrix-style.",
        "higgsfield": "Bullet time · Frozen in motion",
        "workflow": "bullet_time.json",
        "preview": "bullet_time.mp4",
        "best_with": "A jump, flip or throw against a fairly simple background. Any camera.",
        "options": [
            {"id": "angle", "label": "Swing (deg)", "node": "LidarLabBulletTime", "input": "angle",
             "type": "range", "min": -30, "max": 30, "step": 1, "default": 14},
            {"id": "duration", "label": "Freeze (s)", "node": "LidarLabBulletTime", "input": "duration",
             "type": "range", "min": 1.0, "max": 4.0, "step": 0.1, "default": 2.2},
            {"id": "zoom", "label": "Push in", "node": "LidarLabBulletTime", "input": "zoom",
             "type": "range", "min": 0.0, "max": 0.8, "step": 0.05, "default": 0.35},
        ],
    },
]

BY_ID = {p["id"]: p for p in PRESETS}
