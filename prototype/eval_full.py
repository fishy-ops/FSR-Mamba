"""Evaluate captured sequences or stream a synthetic recurrent drift test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.config import load_checkpoint, new_state
from fsrmamba.engine_data import list_scenes, load_engine_scene
from fsrmamba.evalkit import paired_scene_bootstrap, run_scene, scored_frames, summarise
from fsrmamba.fused import FusedFast
from fsrmamba.metrics import psnr
from fsrmamba.synth import halton_jitter, random_scene


COLUMNS = ("psnr", "ssim", "ti", "gt_ti", "dev", "abs_dev")


def print_row(label, summary):
    print(f"{label:32s} {summary['psnr']:8.3f} {summary['ssim']:8.5f} "
          + " ".join(f"{summary[key]:+10.5f}" if key == "dev" else f"{summary[key]:10.5f}"
                     for key in COLUMNS[2:]))
    if "lpips" in summary:
        print(f"  LPIPS {summary['lpips']:.5f}")


def lpips_hook(metric, values):
    def hook(i, frame, out, prev_out, prev_gt, model):
        def prep(x):
            return x.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
        values.append(float(metric(prep(out), prep(frame["gt"])).item()))
    return hook


def compare(results, later, earlier, block, skip=0):
    rows = {}
    for key in ("psnr", "ssim", "abs_dev"):
        def series(source):
            paired = [scene for scene in results.values() if later in scene and earlier in scene]
            values = [scored_frames(scene[source]["per_frame"], skip) for scene in paired]
            return [[abs(v) for v in scores["dev"]] if key == "abs_dev" else scores[key]
                    for scores in values]
        a, b = series(later), series(earlier)
        finite = all(math.isfinite(v) for scenes in (a, b) for scene in scenes for v in scene)
        mean, lo, hi = (paired_scene_bootstrap(a, b, block=block) if finite else
                        (float("nan"), float("nan"), float("nan")))
        if any(math.isnan(v) for v in (mean, lo, hi)):
            verdict = "n/a"
        elif lo <= 0 <= hi:
            verdict = "tie"
        elif (lo > 0) == (key != "abs_dev"):
            verdict = "better"
        else:
            verdict = "worse"
        rows[key] = {"mean_diff": mean, "lo": lo, "hi": hi, "verdict": verdict}
        print(f"{later} - {earlier}  {key:7s} {mean:+.5f} "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  {verdict}"
              + (" (no finite paired estimate)" if verdict == "n/a" else ""))
    return rows


def report_ranges(results, skip):
    print("\nFrame ranges (PSNR / SSIM)")
    for name, sources in results.items():
        for source, result in sources.items():
            scores = result["per_frame"]
            for start, end, label in ((0, 4, "0-3"), (4, 8, "4-7"), (8, 16, "8-15"),
                                      (16, 32, "16-31"), (32, None, "32+")):
                psnrs = scores["psnr"][max(start, skip):end]
                ssims = scores["ssim"][max(start, skip):end]
                if psnrs:
                    print(f"{name}  {source}  {label}: "
                          f"{sum(psnrs) / len(psnrs):.3f} / {sum(ssims) / len(ssims):.5f}")


def evaluate(args):
    names = args.scenes.split(",") if args.scenes else list_scenes(args.engine_data)[:3]
    if not names:
        raise ValueError("No capture scenes found")
    metric = None
    if args.lpips:
        try:
            import lpips
        except ImportError:
            print("LPIPS unavailable; continuing without it.")
        else:
            metric = lpips.LPIPS(net="alex").to(args.device).eval()
    overrides = {} if args.robust_disocc is None else {"robust_disocc": bool(args.robust_disocc)}
    models, results = {}, {}
    baseline = "FSR (captured)"
    forwards = {path: path + (" [amp]" if args.amp else "") for path in args.ckpt}
    sources, fused_sources = [], {}
    configs = {}
    fused_available = args.fused
    if fused_available:
        if args.device != "cuda" or not torch.cuda.is_available():
            print("Fused fp16 skipped: requires CUDA and CuPy on a CUDA device.")
            fused_available = False
        else:
            try:
                import cupy
            except ImportError:
                print("Fused fp16 skipped: requires CUDA and CuPy on a CUDA device.")
                fused_available = False
    rejected = set()
    for name in names:
        scene = load_engine_scene(str(Path(args.engine_data) / name), half=True, mmap=True)
        if not scene:
            raise ValueError(f"Empty scene: {name}")
        render = tuple(scene[0]["lr"].shape[:2])
        output = tuple(scene[0]["gt"].shape[:2])
        results[name] = {}
        print(f"\nScene {name} ({render} -> {output}); frames >= {args.skip} scored")
        print("source                               PSNR     SSIM     instab  GT instab        dev      |dev|")
        runs = [(baseline, None, None)]
        for path in args.ckpt:
            key = (path, render, output)
            if key not in models:
                models[key], cfg = load_checkpoint(path, render, output, args.device, overrides)
                configs[path] = cfg
            model = models[key]
            runs.append((forwards[path], model, None))
            if fused_available:
                try:
                    FusedFast(model)
                except ValueError as exc:
                    if path not in rejected:
                        print(f"{path} [fused fp16] skipped: {exc}")
                        rejected.add(path)
                else:
                    source = f"{path} [fused fp16]"
                    fused_sources[source] = forwards[path]
                    runs.append((source, model, "cuda"))
        for source, model, backend in runs:
            if source not in sources:
                sources.append(source)
            perceptual = []
            scores = run_scene(scene, model, args.device, args.frames,
                               lpips_hook(metric, perceptual) if metric is not None else None,
                               amp=args.amp, native_lr=args.native_lr or args.fused,
                               fused_backend=backend)
            if metric is not None:
                scores["lpips"] = perceptual
            summary = summarise(scores, args.skip)
            results[name][source] = {"per_frame": scores, "summary": summary}
            print_row(source, summary)
    print("\nMean over scenes")
    print("source                               PSNR     SSIM     instab  GT instab        dev      |dev|")
    means = {}
    for source in sources:
        summaries = [scene[source]["summary"] for scene in results.values() if source in scene]
        keys = summaries[0]
        means[source] = {key: sum(summary[key] for summary in summaries) / len(summaries)
                         for key in keys}
        print_row(source, means[source])
    if args.report_ranges:
        report_ranges(results, args.skip)
    print("\nPaired differences (within-scene block bootstrap)")
    comparisons = {}
    pairs = [(source, baseline) for source in sources if source != baseline]
    pairs += [(forwards[path], forwards[args.ckpt[0]]) for path in args.ckpt[1:]]
    pairs += list(fused_sources.items())
    for later, earlier in pairs:
        comparisons[f"{later} - {earlier}"] = compare(results, later, earlier, args.block, args.skip)
    return {"scenes": results, "summary": means, "comparisons": comparisons, "configs": configs,
            "skip": args.skip}


@torch.no_grad()
def drift(args):
    render = (args.height, args.width)
    output = (args.height * args.scale, args.width * args.scale)
    overrides = {} if args.robust_disocc is None else {"robust_disocc": bool(args.robust_disocc)}
    models = {path: load_checkpoint(path, render, output, args.device, overrides)[0]
              for path in args.ckpt}
    baseline = FSRAccumulator(render, output, device=args.device)
    sources = ["FSR (ported)", *args.ckpt]
    result = {}
    for seed in (int(x) for x in args.seeds.split(",")):
        scene = random_scene(seed, output, device=args.device)
        states = {source: (FSRState.zeros(output, args.device) if source == sources[0]
                           else new_state(models[source], args.device))
                  for source in sources}
        scores = {source: [] for source in sources}
        prev_depth = None
        for i in range(args.drift):
            jitter = halton_jitter(i)
            lr = scene.render(i, render, jitter=jitter)
            aux = scene.render(i, output)
            gt = scene.render(i, output, supersample=4)
            for source in sources:
                model = baseline if source == sources[0] else models[source]
                out, states[source] = model(states[source], lr.color, aux.mv, aux.depth,
                                             jitter, prev_depth_hr=prev_depth)
                scores[source].append(float(psnr(out.float(), gt.color.float())))
            prev_depth = aux.depth
        print(f"\nDrift scene seed={seed}; mean PSNR per 100-frame bucket")
        result[str(seed)] = {}
        for source, values in scores.items():
            buckets = [sum(values[i:i + 100]) / len(values[i:i + 100])
                       for i in range(0, len(values), 100)]
            x = torch.arange(len(values), dtype=torch.float64)
            y = torch.tensor(values, dtype=torch.float64)
            slope = float(((x - x.mean()) * (y - y.mean())).sum()
                          / (x - x.mean()).square().sum()) if len(values) > 1 else 0.0
            verdict = "DRIFT" if len(buckets) >= 3 and buckets[-1] < buckets[1] - 0.5 else "stable"
            print(f"{source}: " + " ".join(f"{i * 100}-{min((i + 1) * 100, len(values)) - 1}: {v:.3f}"
                                           for i, v in enumerate(buckets))
                  + f"; slope {slope:+.6f} dB/frame; {verdict}"
                  + (" (insufficient buckets for drift verdict)" if len(buckets) < 3 else ""))
            result[str(seed)][source] = {"psnr": values, "buckets": buckets,
                                        "slope": slope, "verdict": verdict}
    return {"drift": result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-data")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--skip", type=int, default=0, help="Score frames >= N while retaining all history.")
    ap.add_argument("--report-ranges", action="store_true")
    ap.add_argument("--native-lr", action="store_true")
    ap.add_argument("--fused", action="store_true", help="Also score fused fp16 fast inference; implies --native-lr.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--robust-disocc", type=int, choices=(0, 1))
    ap.add_argument("--drift", nargs="?", type=int, const=1000)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seeds", default="0", help="Comma-separated synthetic scene seeds.")
    args = ap.parse_args()
    if args.drift is None and not args.engine_data:
        ap.error("--engine-data is required unless --drift is used")
    if args.frames is not None and args.frames < 1:
        ap.error("--frames must be positive")
    if args.skip < 0:
        ap.error("--skip must be nonnegative")
    if args.block < 1 or (args.drift is not None and args.drift < 1):
        ap.error("--block and --drift must be positive")
    if min(args.height, args.width, args.scale) < 1:
        ap.error("height, width, and scale must be positive")
    if len(set(args.ckpt)) != len(args.ckpt):
        ap.error("Checkpoint paths must be distinct")
    report = drift(args) if args.drift is not None else evaluate(args)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
