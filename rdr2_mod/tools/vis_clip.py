#!/usr/bin/env python3
"""Labelled side-by-side clip of one crop over a whole cache: target, FSR 2 and each checkpoint.

    vis_clip.py CACHE.pt OUT.mp4 --ckpt A.pt [B.pt ...] --crop X,Y,W,H [--zoom 3] [--device cuda] [--fps 30]

X,Y,W,H are in output pixels. Panels are point-zoomed (no smoothing) so edge flicker and fringes stay visible.
The last frame is also written next to OUT as a JPEG.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from fsrmamba.config import load_checkpoint


def panel(t, box, zoom, label):
    x, y, w, h = box
    a = (t[y:y + h, x:x + w].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(a).resize((w * zoom, h * zoom), Image.NEAREST)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle((0, 0, 12 + 13 * len(label), 32), fill=(0, 0, 0))
    d.text((6, 4), label, fill=(255, 255, 0), font=font)
    return img


def tile(panels):
    w, h = panels[0].size
    cols = min(len(panels), 3)
    rows = (len(panels) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (40, 40, 40))
    for i, p in enumerate(panels):
        out.paste(p, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    # libx264 yuv420p needs even dimensions
    return out.crop((0, 0, out.width // 2 * 2, out.height // 2 * 2))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--crop", required=True)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()
    box = tuple(int(v) for v in args.crop.split(","))
    frames = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.frames:
        frames = frames[: args.frames]
    h, w = frames[0]["lr"].shape[:2]
    H, W = frames[0]["gt"].shape[:2]
    labels = args.labels + [p.stem for p in args.ckpt[len(args.labels):]]
    outputs = []
    for path in args.ckpt:
        model, _ = load_checkpoint(path, (h, w), (H, W), args.device)
        state, seq = model.init_state(), []
        for f in frames:
            out, state = model(state, f["lr"].to(args.device).float(), f["mv"].to(args.device).float(),
                               f["depth"].to(args.device).float(), tuple(float(v) for v in f["jitter"]))
            x, y, cw, ch = box
            seq.append(out[y:y + ch, x:x + cw].clamp(0, 1).cpu())
        outputs.append(seq)
        print(f"ran {path.name}", flush=True)
    local = (0, 0, box[2], box[3])
    sheets = []
    for i, f in enumerate(frames):
        x, y, cw, ch = box
        panels = [panel(f["gt"][y:y + ch, x:x + cw], local, args.zoom, "target (game, accumulated)")]
        if "fsr_out" in f:
            panels.append(panel(f["fsr_out"][y:y + ch, x:x + cw], local, args.zoom, "FSR 2"))
        panels += [panel(seq[i], local, args.zoom, lab) for seq, lab in zip(outputs, labels)]
        sheets.append(tile(panels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheets[-1].save(args.out.with_suffix(".jpg"), quality=92)
    sw, sh = sheets[0].size
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{sw}x{sh}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", "16",
                           "-pix_fmt", "yuv420p", str(args.out)], stdin=subprocess.PIPE)
    for s in sheets:
        ff.stdin.write(s.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {args.out} ({len(sheets)} frames, {sw}x{sh}) and {args.out.with_suffix('.jpg')}")


if __name__ == "__main__":
    main()
