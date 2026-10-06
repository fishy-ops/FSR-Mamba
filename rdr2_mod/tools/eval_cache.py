#!/usr/bin/env python3
"""Score checkpoints on a training cache against its gt (masked where the cache has a mask), and FSR's output.

    eval_cache.py CACHE.pt --ckpt A.pt [B.pt ...] [--device cuda] [--warmup 8] [--flip-jitter]

--flip-jitter negates the cache's jitter: on a freshly converted cache the convention that scores clearly
better is the correct one.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
import torch
from fsrmamba.config import load_checkpoint
from fsrmamba.metrics import psnr, ssim, temporal_deviation


def masked(metric, a, b, m):
    if m is None:
        return float(metric(a, b))
    return float(metric(torch.where(m[..., None] > 0, a, b), b))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flip-jitter", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="score only the first N frames (0 = all)")
    ap.add_argument("--png", type=Path, help="save the last frame of each model (+gt, fsr) as PNG here")
    args = ap.parse_args()
    frames = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.frames:
        frames = frames[: args.frames]
    h, w = frames[0]["lr"].shape[:2]
    H, W = frames[0]["gt"].shape[:2]
    cover = [float(f["mask"].float().mean()) for f in frames if "mask" in f]
    print(f"{len(frames)} frames {w}x{h} -> {W}x{H}; mask coverage {sum(cover) / len(cover):.3f}" if cover else f"{len(frames)} frames")
    sign = -1.0 if args.flip_jitter else 1.0
    fsr = []
    for i, f in enumerate(frames):
        if i >= args.warmup and "fsr_out" in f:
            m = f.get("mask")
            fsr.append((masked(psnr, f["fsr_out"].float(), f["gt"].float(), m), masked(ssim, f["fsr_out"].float(), f["gt"].float(), m)))
    if fsr:
        print(f"FSR 2 (downsampled) vs gt: PSNR {sum(r[0] for r in fsr) / len(fsr):.3f} SSIM {sum(r[1] for r in fsr) / len(fsr):.4f}")
    for path in args.ckpt:
        model, _ = load_checkpoint(path, (h, w), (H, W), args.device)
        state, rows, devs, prev = model.init_state(), [], [], None
        for i, f in enumerate(frames):
            j = tuple(float(v) * sign for v in f["jitter"])
            out, state = model(state, f["lr"].to(args.device).float(), f["mv"].to(args.device).float(),
                               f["depth"].to(args.device).float(), j)
            out = out.clamp(0, 1 - 1 / 1024)
            if i >= args.warmup:
                g = f["gt"].to(args.device).float()
                m = f["mask"].to(args.device) if "mask" in f else None
                rows.append((masked(psnr, out, g, m), masked(ssim, out, g, m)))
                if prev is not None:   # flicker: output's frame-to-frame instability minus the target's
                    mv_hr = torch.nn.functional.interpolate(f["mv"].to(args.device).float().permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                    d, _ = temporal_deviation(out, prev[0], g, prev[1], mv_hr)
                    devs.append(abs(d))
            prev = (out, f["gt"].to(args.device).float())
        if args.png:
            from PIL import Image
            args.png.mkdir(parents=True, exist_ok=True)
            save = lambda t, n: Image.fromarray((t.clamp(0, 1).cpu() * 255).round().byte().numpy()).save(args.png / n)
            save(out, f"{path.stem}.png"); save(frames[-1]["gt"].float(), "gt.png")
            if "fsr_out" in frames[-1]: save(frames[-1]["fsr_out"].float(), "fsr2_down.png")
        print(f"{path.name}{' (jitter flipped)' if args.flip_jitter else ''}: PSNR {sum(r[0] for r in rows) / len(rows):.3f} "
              f"SSIM {sum(r[1] for r in rows) / len(rows):.4f} |flicker dev| {sum(devs) / max(len(devs), 1):.5f} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
