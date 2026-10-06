"""Write synthetic float16 caches accepted by the engine capture loader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.synth import halton_jitter, random_scene


def make_fake_engine(out, scenes=4, frames=12, render=(96, 128), scale=2):
    output = tuple(size * scale for size in render)
    with torch.no_grad():
        for seed in range(scenes):
            scene = random_scene(seed, output)
            baseline = FSRAccumulator(render, output)
            state = FSRState.zeros(output)
            prev_depth, capture = None, []
            for i in range(frames):
                jitter = halton_jitter(i)
                lr = scene.render(i, render, jitter=jitter)
                aux_lr = scene.render(i, render)
                aux_hr = scene.render(i, output)
                gt = scene.render(i, output, supersample=4)
                fsr, state = baseline(state, lr.color, aux_hr.mv, aux_hr.depth, jitter,
                                      prev_depth_hr=prev_depth)
                prev_depth = aux_hr.depth
                capture.append({"lr": lr.color.half(), "mv": aux_lr.mv.half(),
                                "depth": aux_lr.depth.half(), "gt": gt.color.half(),
                                "fsr_out": fsr.half(), "jitter": jitter})
            directory = Path(out) / f"scene_{seed:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(capture, directory / "_cache_tm1.pt")
            print(f"Wrote {directory.name}: {frames} frames ({render} -> {output})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--render", default="96x128", help="Render height x width.")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()
    try:
        render = tuple(int(x) for x in args.render.lower().split("x"))
        if len(render) != 2 or min(*render, args.scale, args.scenes, args.frames) < 1:
            raise ValueError
    except ValueError:
        ap.error("render must be HxW and all sizes/counts must be positive")
    make_fake_engine(args.out, args.scenes, args.frames, render, args.scale)


if __name__ == "__main__":
    main()
