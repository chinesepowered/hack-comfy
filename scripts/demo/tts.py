#!/usr/bin/env python3
"""Render the demo video narration with ElevenLabs (voice-over only; not used by the app).

    ELEVENLABS_API_KEY=... python scripts/demo/tts.py out_dir/

Writes one mp3 + one JSON of character timings per segment from narration.json.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "narration_out"
    os.makedirs(out, exist_ok=True)
    key = os.environ["ELEVENLABS_API_KEY"]
    cfg = json.load(open(os.path.join(HERE, "narration.json")))
    speed = float(os.environ.get("TTS_SPEED", "1.0"))
    prev = None
    for seg in cfg["segments"]:
        mp3 = os.path.join(out, f"{seg['id']}.mp3")
        if os.path.exists(mp3) and not os.environ.get("FORCE"):
            print("skip", seg["id"])
            prev = seg["text"]
            continue
        body = {"text": seg["text"], "model_id": cfg["model_id"],
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.3, "use_speaker_boost": True,
                                   "speed": speed}}
        if prev:
            body["previous_text"] = prev
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{cfg['voice_id']}/with-timestamps?output_format=mp3_44100_192",
            data=json.dumps(body).encode(), headers={"xi-api-key": key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        open(mp3, "wb").write(base64.b64decode(d["audio_base64"]))
        json.dump(d.get("alignment") or d.get("normalized_alignment"), open(os.path.join(out, f"{seg['id']}.json"), "w"))
        end = (d.get("alignment") or {}).get("character_end_times_seconds", [0])[-1]
        print(f"{seg['id']}: {end:.2f}s  ({len(seg['text'])} chars)")
        prev = seg["text"]


if __name__ == "__main__":
    main()
