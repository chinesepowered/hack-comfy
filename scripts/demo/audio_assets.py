#!/usr/bin/env python3
"""Generate the demo video's music bed and sound effects with ElevenLabs.

Only the submission video uses these; the app and workflows do not.
    ELEVENLABS_API_KEY=... python scripts/demo/audio_assets.py out_dir/
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

MUSIC = {
    "prompt": "Minimal modern electronic tech demo background track. Warm analog synth pads, soft pulsing bass, "
              "light glitchy percussion, subtle arpeggios, optimistic and focused, 112 BPM, no vocals, steady energy, "
              "clean mix that sits under a voice-over, gentle ending.",
    "music_length_ms": 114000,
}
SFX = {
    "laser_sweep": ("Futuristic LiDAR laser scanner sweep, bright rising electronic shimmer, sci-fi", 2.2),
    "whoosh": ("Fast cinematic air whoosh transition, clean, short", 1.0),
    "freeze": ("Time freeze bullet time effect, deep bass drop with reverse swoosh and slowed shimmer", 2.4),
    "dust": ("Magical disintegration, fine sand and ash blowing away in the wind, soft crackle", 2.4),
    "shutter": ("Rapid digital camera shutter bursts with a soft glitch", 1.4),
    "logo": ("Short modern tech logo sting, soft synth pluck with airy reverb tail", 2.0),
}


def post(url, body, key):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "audio_out"
    os.makedirs(out, exist_ok=True)
    key = os.environ["ELEVENLABS_API_KEY"]
    path = os.path.join(out, "music.mp3")
    if not os.path.exists(path):
        open(path, "wb").write(post("https://api.elevenlabs.io/v1/music", MUSIC, key))
        print("music ok")
    for name, (text, dur) in SFX.items():
        path = os.path.join(out, f"sfx_{name}.mp3")
        if os.path.exists(path):
            continue
        open(path, "wb").write(post("https://api.elevenlabs.io/v1/sound-generation",
                                    {"text": text, "duration_seconds": dur, "prompt_influence": 0.5}, key))
        print("sfx", name, "ok")


if __name__ == "__main__":
    main()
