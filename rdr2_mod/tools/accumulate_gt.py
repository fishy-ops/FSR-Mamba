#!/usr/bin/env python3
"""Build a true supersampled reference for a FROZEN-scene dump and score against it.

Input is a training cache written by ``analyze_dump.py --export-cache`` (list of dicts: lr, mv, depth, gt,
fsr_out, jitter; colour in Reinhard space). The rendered sample of low-res pixel (x, y) sits at
(x + 0.5 + s*jx, y + 0.5 + s*jy); s is chosen as the sign that gives the sharper accumulation.

Every low-res sample of every frame is dropped into the output pixel that contains it and the linear radiance
is averaged (box-filter supersampling). With 2x upscaling and a Halton jitter each output pixel collects about
N/4 real samples from N frames. Nothing from FSR is used, so the result is a reference in the game's own look.
It is only valid where the scene did not change during the recording: a per-pixel stability mask is written
alongside and used for scoring.

    accumulate_gt.py CACHE.pt --out OUT_CACHE.pt [--ckpt CKPT ...] [--frames A:B] [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
import torch
import torch.nn.functional as F
from fsrmamba.config import load_checkpoint
from fsrmamba.metrics import psnr, ssim


def to_linear(t):
    t = t.float().clamp(0, 1 - 1 / 1024)
    return t / (1 - t)


def to_reinhard(x):
    return x / (1 + x)


def accumulate(frames, sign=1.0, scale=2):
    h, w = frames[0]["lr"].shape[:2]
    H, W = h * scale, w * scale
    total = torch.zeros(H * W, 3, dtype=torch.float64)
    square = torch.zeros(H * W, dtype=torch.float64)
    count = torch.zeros(H * W, dtype=torch.float64)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    for f in frames:
        jx, jy = (float(v) * sign for v in f["jitter"])
        ox = ((xs + 0.5 + jx) * scale).floor().long().clamp(0, W - 1)
        oy = ((ys + 0.5 + jy) * scale).floor().long().clamp(0, H - 1)
        index = (oy * W + ox).reshape(-1)
        linear = to_linear(f["lr"]).reshape(-1, 3).double()
        total.index_add_(0, index, linear)
        luma = to_reinhard(linear).mean(-1)
        square.index_add_(0, index, luma * luma)
        count.index_add_(0, index, torch.ones_like(luma))
    filled = count > 0
    mean = total / count.clamp(min=1)[:, None]
    luma_mean = to_reinhard(mean).mean(-1)
    spread = float(((square / count.clamp(min=1)) - luma_mean * luma_mean).clamp(min=0)[filled].mean())
    reference = to_reinhard(mean).float().reshape(H, W, 3)
    return reference, count.reshape(H, W), filled.reshape(H, W), spread


def stability_mask(frames, threshold=0.02):
    """Low-res pixels whose colour did not drift between the first and last third of the recording."""
    n = len(frames)
    third = max(n // 3, 1)
    first = torch.stack([f["lr"].float() for f in frames[:third]]).mean(0)
    last = torch.stack([f["lr"].float() for f in frames[-third:]]).mean(0)
    drift = (first - last).abs().amax(-1)
    drift = F.max_pool2d(drift[None, None], 5, stride=1, padding=2)[0, 0]
    return drift < threshold


def gradient_energy(image):
    return float((image[1:] - image[:-1]).abs().mean() + (image[:, 1:] - image[:, :-1]).abs().mean())


def masked(metric, a, b, mask):
    # Score only stable pixels: copy the reference into unstable ones so they contribute zero error.
    a = torch.where(mask[..., None], a, b)
    return float(metric(a, b))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ckpt", type=Path, nargs="*", default=[])
    parser.add_argument("--frames", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--save-png", type=Path)
    parser.add_argument("--sign", type=float, default=1.0)
    parser.add_argument("--frozen", action="store_true",
                        help="photo-mode capture: the world is frozen, so every covered pixel is valid. The "
                             "drift test flags sharp edges as unstable because different jitter subsets "
                             "average them differently")
    args = parser.parse_args()

    frames = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.frames:
        a, b = (int(v) if v else None for v in args.frames.split(":"))
        frames = frames[a:b]
    h, w = frames[0]["lr"].shape[:2]
    print(f"{len(frames)} frames, render {w}x{h}")
    motion = torch.stack([(f["mv"].float() * torch.tensor([w, h])).norm(dim=-1).quantile(0.99) for f in frames])
    print(f"motion p99 per frame (pixels): max {float(motion.max()):.3f} mean {float(motion.mean()):.3f}")

    # The cache uses the training convention: sample at texel centre + jitter (see fast.py jitter_sign comment,
    # measured on the SSAA captures). Gradient energy must NOT pick the sign: misplaced samples create stair-step
    # edges that look sharper. The within-pixel spread is reported as a check (correct sign = lower spread).
    for trial in (1.0, -1.0):
        _, count, filled, spread = accumulate(frames, trial)
        print(f"jitter sign {trial:+.0f}: samples/pixel min {int(count.min())} mean {float(count.mean()):.1f}; "
              f"within-pixel spread {spread:.6f}")
    sign = args.sign
    reference, count, filled, _ = accumulate(frames, sign)
    print(f"using sign {sign:+.0f}")
    stable = torch.ones_like(frames[0]["lr"][..., 0], dtype=torch.bool) if args.frozen else stability_mask(frames)
    stable_hr = stable.repeat_interleave(2, 0).repeat_interleave(2, 1) & filled
    print(f"stable pixels: {float(stable_hr.float().mean()) * 100:.1f}%")

    fsr = frames[-1]["fsr_out"].float()
    print(f"FSR 2 output (last frame) vs reference, stable pixels: PSNR {masked(psnr, fsr, reference, stable_hr):.3f} "
          f"SSIM {masked(ssim, fsr, reference, stable_hr):.4f}")

    device = args.device
    for path in args.ckpt:
        model, _ = load_checkpoint(path, (h, w), (2 * h, 2 * w), device)
        state = model.init_state()
        scores, fsr_scores = [], []
        ref_d, mask_d = reference.to(device), stable_hr.to(device)
        for index, f in enumerate(frames):
            jitter = f["jitter"]
            jitter = tuple(float(v) for v in jitter)
            out, state = model(state, f["lr"].to(device).float(), f["mv"].to(device).float(),
                               f["depth"].to(device).float(), jitter)
            out = out.clamp(0, 1 - 1 / 1024)
            if index >= args.warmup:
                scores.append((masked(psnr, out, ref_d, mask_d), masked(ssim, out, ref_d, mask_d)))
                g = f["fsr_out"].to(device).float()
                fsr_scores.append((masked(psnr, g, ref_d, mask_d), masked(ssim, g, ref_d, mask_d)))
        mean = lambda rows, i: sum(r[i] for r in rows) / len(rows)
        print(f"{path.name}: ours PSNR {mean(scores, 0):.3f} SSIM {mean(scores, 1):.4f} | "
              f"FSR 2 PSNR {mean(fsr_scores, 0):.3f} SSIM {mean(fsr_scores, 1):.4f}  ({len(scores)} frames after warm-up)")
        if args.save_png:
            from PIL import Image
            args.save_png.mkdir(parents=True, exist_ok=True)
            to_img = lambda t: Image.fromarray((t.clamp(0, 1).cpu() * 255).round().byte().numpy())
            to_img(reference).save(args.save_png / "reference.png")
            to_img(out).save(args.save_png / f"ours_{path.stem}.png")
            to_img(frames[-1]["fsr_out"].float()).save(args.save_png / "fsr2.png")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        mask_lr = stable
        result = []
        for f in frames:
            g = dict(f)
            g["gt"] = reference.half()
            g["mask"] = stable_hr.to(torch.uint8)
            g["static"] = True   # frozen world: one reference for every frame
            g["fsr_out"] = f["fsr_out"].half()
            g["lr"], g["mv"], g["depth"] = f["lr"].half(), f["mv"].half(), f["depth"].half()
            g["jitter"] = tuple(float(v) for v in f["jitter"])   # unchanged: the model's own convention
            result.append(g)
        torch.save(result, args.out)
        torch.save(dict(stable_lr=mask_lr, count=count), args.out.with_suffix(".mask.pt"))
        print(f"wrote {args.out} ({len(result)} frames, gt = accumulated reference)")


if __name__ == "__main__":
    main()
