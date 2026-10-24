#!/usr/bin/env python3
"""Assemble the LiDAR Lab demo video (<= 2 min) from renders, screen recordings,
overlay cards, ElevenLabs narration / music / SFX, with burned-in captions.

    python scripts/demo/make_video.py timeline.json out.mp4

timeline.json lists segments; each segment has a narration id and visual pieces:
  {"narration": "s1_hook", "pieces": [{"src": "...mp4", "in": 0.4, "dur": 2.7, "speed": 1.0,
    "overlay": "cards/label_lidar.png", "blur": false, "sfx": [["laser_sweep", 0.1]]}, ...]}
Paths are relative to the timeline file's "root".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

FPS = 30
W, H = 1920, 1080
LEAD, TAIL = 0.3, 0.4


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(cmd))
        print(r.stderr[-3000:])
        raise SystemExit(1)


def dur_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def build_piece(p, root, tmp, idx):
    out = os.path.join(tmp, f"piece_{idx:03d}.mp4")
    speed = float(p.get("speed", 1.0))
    dur = float(p["dur"])
    src = os.path.join(root, p["src"])
    chain = []
    is_img = src.lower().endswith((".png", ".jpg"))
    if is_img:
        inputs = ["-loop", "1", "-t", f"{dur:.3f}", "-i", src]
    else:
        inputs = ["-ss", f"{float(p.get('in', 0.0)):.3f}", "-t", f"{dur * speed + 0.2:.3f}", "-i", src]
        chain.append(f"setpts=(PTS-STARTPTS)/{speed}")
    if p.get("crop"):
        cx, cy, cw, ch = p["crop"]
        chain.append(f"crop={cw}:{ch}:{cx}:{cy}")
    if p.get("fit"):  # letterbox instead of filling the frame
        chain += [f"scale={W}:{H}:force_original_aspect_ratio=decrease", f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x05070d",
                  f"fps={FPS}", "format=rgba"]
    else:
        chain += [f"scale={W}:{H}:force_original_aspect_ratio=increase", f"crop={W}:{H}", f"fps={FPS}", "format=rgba"]
    if p.get("zoom"):  # slow push-in
        z = float(p["zoom"])
        chain += [f"scale=iw*(1+{z}):-1", f"crop={W}:{H}:(iw-{W})/2:(ih-{H})/2"]
    if p.get("blur"):
        chain += ["gblur=sigma=18", "eq=brightness=-0.18:saturation=0.8"]
    vf = ",".join(chain)
    fc = f"[0:v]{vf}[base]"
    extra = []
    last = "[base]"
    for k, ov in enumerate([o for o in [p.get("overlay"), p.get("overlay2")] if o]):
        extra += ["-loop", "1", "-t", f"{dur:.3f}", "-i", os.path.join(root, ov)]
        fc += f";[{k + 1}:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1[o{k}];{last}[o{k}]overlay=0:0[v{k}]"
        last = f"[v{k}]"
    fc += f";{last}trim=duration={dur:.3f},tpad=stop_mode=clone:stop_duration=1,trim=duration={dur:.3f},format=yuv420p[out]"
    run(["ffmpeg", "-v", "error", "-y", *inputs, *extra, "-filter_complex", fc, "-map", "[out]", "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-r", str(FPS), out])
    return out


def words_from_alignment(al):
    chars, st, en = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    words, cur, t0, t1 = [], "", None, None
    for c, a, b in zip(chars, st, en):
        if c.isspace():
            if cur:
                words.append((cur, t0, t1))
            cur, t0 = "", None
            continue
        if t0 is None:
            t0 = a
        cur += c
        t1 = b
    if cur:
        words.append((cur, t0, t1))
    return words


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_ass(lines, path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter Display,60,&H00FFE14D,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,2,140,140,64,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, "w") as f:
        f.write(head)
        for start, end, text in lines:
            f.write(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{text}\n")


def caption_lines(words, offset, max_chars=38):
    """Group words into short lines with karaoke (\\k) highlighting of the spoken word."""
    lines, cur = [], []

    def flush():
        if not cur:
            return
        start, end = cur[0][1], cur[-1][2]
        parts = []
        prev_end = start
        for w, a, b in cur:
            gap_cs = max(0, int(round((a - prev_end) * 100)))
            dur_cs = max(1, int(round((b - a) * 100)))
            if gap_cs:
                parts.append(f"{{\\k{gap_cs}}}")
            parts.append(f"{{\\k{dur_cs}}}{w} ")
            prev_end = b
        lines.append((offset + start, offset + end + 0.25, "".join(parts).strip()))
        cur.clear()

    for w, a, b in words:
        length = sum(len(x[0]) + 1 for x in cur) + len(w)
        if cur and length > max_chars:
            flush()
        cur.append((w, a, b))
        if w.endswith((".", "?", "!", ":")) and sum(len(x[0]) + 1 for x in cur) > 14:
            flush()
    flush()
    # never let consecutive lines overlap
    for i in range(len(lines) - 1):
        s, e, t = lines[i]
        if e > lines[i + 1][0]:
            lines[i] = (s, lines[i + 1][0] - 0.02, t)
    return lines


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    spec = json.load(open(spec_path))
    root = spec["root"]
    tmp = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_build")
    os.makedirs(tmp, exist_ok=True)
    vo_dir = os.path.join(root, spec["narration_dir"])
    sfx_dir = os.path.join(root, spec["audio_dir"])

    pieces, t, idx = [], 0.0, 0
    narr_events, sfx_events, cap_lines = [], [], []
    for seg in spec["segments"]:
        nid = seg.get("narration")
        seg_start = t
        if nid:
            vo = os.path.join(vo_dir, f"{nid}.mp3")
            vo_dur = dur_of(vo)
            narr_events.append((vo, seg_start + LEAD))
            al = json.load(open(os.path.join(vo_dir, f"{nid}.json")))
            cap_lines += caption_lines(words_from_alignment(al), seg_start + LEAD)
            target = LEAD + vo_dur + TAIL + float(seg.get("extra", 0.0))
        else:
            target = sum(float(p["dur"]) for p in seg["pieces"])
        fixed = sum(float(p["dur"]) for p in seg["pieces"][:-1])
        seg["pieces"][-1]["dur"] = max(0.5, target - fixed)
        for p in seg["pieces"]:
            print(f"  [{t:6.2f}] {p['src']}  {p['dur']:.2f}s")
            pieces.append(build_piece(p, root, tmp, idx))
            for name, at in p.get("sfx", []):
                sfx_events.append((os.path.join(sfx_dir, f"sfx_{name}.mp3"), t + at, p.get("sfx_gain", 0.55)))
            t += float(p["dur"])
            idx += 1
    total = t
    print(f"total {total:.2f}s")

    concat = os.path.join(tmp, "concat.txt")
    with open(concat, "w") as f:
        for p in pieces:
            f.write(f"file '{p}'\n")
    video = os.path.join(tmp, "video.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", video])

    ass = os.path.join(tmp, "captions.ass")
    make_ass(cap_lines, ass)

    # audio: narration + ducked music + sfx
    inputs, fc, labels = [], [], []
    for k, (path, at) in enumerate(narr_events):
        inputs += ["-i", path]
        fc.append(f"[{k}:a]aresample=48000,adelay={int(at * 1000)}|{int(at * 1000)},apad[n{k}]")
        labels.append(f"[n{k}]")
    n = len(narr_events)
    fc.append(f"{''.join(labels)}amix=inputs={n}:normalize=0:duration=longest,atrim=0:{total:.3f},asplit=2[voice][key]")
    inputs += ["-i", os.path.join(sfx_dir, "music.mp3")]
    mi = n
    fc.append(f"[{mi}:a]aresample=48000,atrim=0:{total:.3f},volume={spec.get('music_gain', 0.5)},"
              f"afade=t=in:st=0:d=1.5,afade=t=out:st={total - 3:.3f}:d=3[mus]")
    fc.append("[mus][key]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=450:makeup=1[ducked]")
    sl = []
    for k, (path, at, gain) in enumerate(sfx_events):
        inputs += ["-i", path]
        j = mi + 1 + k
        fc.append(f"[{j}:a]aresample=48000,volume={gain},adelay={int(at * 1000)}|{int(at * 1000)},apad[s{k}]")
        sl.append(f"[s{k}]")
    mix_in = "[voice][ducked]" + "".join(sl)
    fc.append(f"{mix_in}amix=inputs={2 + len(sl)}:normalize=0:duration=first,atrim=0:{total:.3f},"
              f"loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    audio = os.path.join(tmp, "audio.m4a")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[aout]",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", audio])

    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", audio, "-vf", f"ass={ass}", "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", "-shortest", out_path])
    print("wrote", out_path, f"{dur_of(out_path):.2f}s")


if __name__ == "__main__":
    main()
