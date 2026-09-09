"""Run the FSR baseline over a synthetic sequence and report the numbers to beat.

Usage:
    ../.venv/bin/python run_baseline.py [--frames 32] [--scale 2] [--dump out/]

Compares three things at the same output resolution:

  bilinear  -- naive single-frame upscale, the "did anything happen" floor
  lanczos   -- FSR's spatial resolve with *no* temporal accumulation, which
               isolates how much of the quality comes from time rather than
               from the filter
  fsr       -- the full ported temporal accumulator

If `fsr` does not clearly beat `lanczos`, temporal accumulation is not working
and there is a bug -- that comparison is the point of this script.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn.functional as F

from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.metrics import psnr, ssim, temporal_instability, temporal_deviation
from fsrmamba.synth import default_scene, halton_jitter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--scale", type=float, default=2.0, help="upscale ratio")
    ap.add_argument("--gt-supersample", type=int, default=6)
    ap.add_argument("--dump", type=str, default="", help="directory for PNG dumps")
    args = ap.parse_args()

    out_size = (args.height, args.width)
    render_size = (int(args.height / args.scale), int(args.width / args.scale))
    scene = default_scene(out_size)

    print(f"render {render_size[1]}x{render_size[0]} -> output {out_size[1]}x{out_size[0]}"
          f"  ({args.scale}x), {args.frames} frames")

    acc = FSRAccumulator(render_size, out_size)
    state = FSRState.zeros(out_size)

    totals = {k: {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0} for k in ("bilinear", "lanczos", "fsr")}
    prev = {k: None for k in totals}
    prev_gt = None
    prev_depth = None
    n_ti = 0

    dump_dir = pathlib.Path(args.dump) if args.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.frames):
        jitter = halton_jitter(i)
        lr = scene.render(i, render_size, jitter=jitter, supersample=1)
        hr_aux = scene.render(i, out_size, jitter=(0.0, 0.0), supersample=1)
        gt = scene.render(i, out_size, jitter=(0.0, 0.0), supersample=args.gt_supersample)

        lr_nchw = lr.color.permute(2, 0, 1).unsqueeze(0)
        outputs = {
            "bilinear": F.interpolate(
                lr_nchw, size=out_size, mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0),
            # Spatial-only: run the resolve, discard the history.
            "lanczos": None,
            "fsr": None,
        }

        upsampled, _, _, _ = acc._upsample(lr.color, jitter)
        from fsrmamba.colorspace import ycocg_to_rgb

        outputs["lanczos"] = ycocg_to_rgb(upsampled).clamp(0, 1)

        fsr_out, state = acc(
            state, lr.color, hr_aux.mv, hr_aux.depth, jitter, prev_depth_hr=prev_depth
        )
        outputs["fsr"] = fsr_out
        prev_depth = hr_aux.depth

        for name, img in outputs.items():
            totals[name]["psnr"] += psnr(img, gt.color)
            totals[name]["ssim"] += ssim(img, gt.color)
            if prev[name] is not None:
                totals[name]["ti"] += temporal_instability(img, prev[name], hr_aux.mv)
                if prev_gt is not None:
                    dev, _ = temporal_deviation(
                        img, prev[name], gt.color, prev_gt, hr_aux.mv
                    )
                    totals[name]["dev"] += abs(dev)
            prev[name] = img.detach()
        prev_gt = gt.color.detach()
        if i > 0:
            n_ti += 1

        if dump_dir and i == args.frames - 1:
            from PIL import Image
            import numpy as np

            strip = np.concatenate(
                [
                    (outputs[k].clamp(0, 1) * 255).byte().cpu().numpy()
                    for k in ("bilinear", "lanczos", "fsr")
                ]
                + [(gt.color.clamp(0, 1) * 255).byte().cpu().numpy()],
                axis=0,
            )
            Image.fromarray(strip).save(dump_dir / "comparison.png")
            print(f"wrote {dump_dir/'comparison.png'} (bilinear / lanczos / fsr / gt)")

    print()
    print(f"{'method':<10} {'PSNR (dB)':>10} {'SSIM':>8} "
          f"{'temporal instab.':>18} {'|dev|':>10}")
    print("-" * 50)
    for name, t in totals.items():
        print(
            f"{name:<10} {t['psnr']/args.frames:>10.2f} {t['ssim']/args.frames:>8.4f}"
            f" {t['ti']/max(1,n_ti):>18.5f}"
            f" {t.get('dev', 0.0)/max(1,n_ti):>10.5f}"
        )
    print("\n(temporal: raw motion-compensated instability. Lower is NOT")
    print(" automatically better -- a blurrier output scores lower.")
    print(" |dev|: distance from the ground truth's own instability;")
    print(" 0 is the target. Select on |dev|.)")


if __name__ == "__main__":
    main()
