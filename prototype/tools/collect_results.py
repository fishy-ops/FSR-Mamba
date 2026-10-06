"""Collect full-frame evaluations, architecture metadata, and latency into Markdown."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import torch


BASELINE = "FSR (captured)"
METRICS = (("psnr", 2), ("ssim", 4), ("dev", 5))
SHAPE = re.compile(r"widths\s*=\s*(\([^)]*\)|\d+(?:,\d+)*)\s+"
                   r"depths\s*=\s*(\([^)]*\)|\d+(?:,\d+)*)")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def checkpoint_name(source):
    return Path(re.sub(r"(?:\s+\[[^\]]*\])+$", "", source)).name


def shape_key(config):
    if "widths" not in config or "depths" not in config:
        return None
    return tuple(config["widths"]), tuple(config["depths"])


def read_bench(paths):
    timings = {}
    for path in paths:
        for line in path.read_text().splitlines():
            match = SHAPE.search(line)
            if match is None:
                continue
            key = tuple(tuple(int(v) for v in re.findall(r"\d+", group))
                        for group in match.groups())
            if line.startswith("SUMMARY "):
                value = re.search(r"\btotal_with_display=(" + NUMBER + r")\b", line)
                kind = "fused"
            elif line.startswith("E2E "):
                value = re.search(r"\bTOTAL TRT\s+(" + NUMBER + r")\s+ms\b", line)
                kind = "trt"
            else:
                continue
            if value:
                timings.setdefault(key, {})[kind] = float(value[1])
    return timings


def read_sidecars(directories):
    configs = {}
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Sidecar directory does not exist: {directory}")
        for path in sorted(directory.glob("*.pt.json")):
            configs.setdefault(path.name[:-5], json.loads(path.read_text()))
    return configs


def architecture(config):
    if not config:
        return "-"
    parts = [config.get("arch", "mamba")]
    key = shape_key(config)
    if key:
        parts.append("/".join(",".join(map(str, values)) for values in key))
    else:
        for field, label in (("state_channels", "state"), ("feature_channels", "features"),
                             ("encoder_depth", "enc")):
            if field in config:
                parts.append(f"{label}={config[field]}")
        if config.get("lr_upsampler") and "lr_dim" in config:
            parts.append(f"lr={config['lr_dim']}")
    for field, label in (("accum", "accum"), ("depth_test", "dt")):
        if config.get(field):
            parts.append(label)
    if "hist_filter" in config:
        parts.append(config["hist_filter"])
    for field, label in (("nearest_sample", "ns"), ("conf_consistent", "cc"),
                         ("hist_residual", "hres"), ("conf_motion", "cm"), ("film", "film"),
                         ("learned_clamp", "clamp"), ("residual", "residual"),
                         ("sharpen", "sharp"), ("rectify", "rectify"),
                         ("lr_upsampler", "lr"), ("lr_film", "lr-film"),
                         ("kernel_predict", "kpn"), ("robust_disocc", "rd"),
                         ("history_input", "hist"), ("ablate_ssm", "no-ssm"),
                         ("learned_resolve", "learned-resolve"), ("swin_resolve", "swin"),
                         ("separable", "sep"), ("lock_feature", "lock")):
        if config.get(field):
            parts.append(label)
    for field, label in (("n_state", "state"), ("detail_ch", "detail"), ("unet", "unet")):
        if config.get(field):
            parts.append(f"{label}={config[field]}")
    if config.get("stem_kernel", 1) != 1:
        parts.append(f"stem={config['stem_kernel']}")
    if config.get("resolve", "nearest") != "nearest":
        parts.append(config["resolve"])
    return " ".join(parts)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def paired_bootstrap(a_scenes, b_scenes):
    """Moving blocks within scenes, following evalkit with equal scene weights."""
    if len(a_scenes) != len(b_scenes):
        raise ValueError("Scene counts differ")
    generator = torch.Generator().manual_seed(0)
    samples = torch.zeros(2000, dtype=torch.float64)
    observed = []
    for a, b in zip(a_scenes, b_scenes):
        if len(a) != len(b):
            raise ValueError("Paired series must have equal lengths")
        if not a:
            return (float("nan"),) * 3
        diff = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
        if not bool(torch.isfinite(diff).all()):
            return (float("nan"),) * 3
        observed.append(diff.mean().item())
        n, block = len(a), min(8, len(a))
        if bool((diff == diff[0]).all()):
            samples += diff[0]
            continue
        offsets = torch.arange(block)
        counts = (n + block - 1) // block
        for start in range(0, 2000, 128):
            size = min(128, 2000 - start)
            starts = torch.randint(n - block + 1, (size, counts), generator=generator)
            indices = (starts[..., None] + offsets).flatten(1)[:, :n]
            samples[start:start + size] += diff[indices].mean(dim=1)
    if not observed:
        return (float("nan"),) * 3
    samples /= len(observed)
    lo, hi = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return mean(observed), lo, hi


def metric_series(entry, metric, skip):
    values = entry["per_frame"][metric]
    if metric == "dev":
        return [abs(v) for v in values[max(skip - 1, 0):]]
    return values[skip:]


def score_source(data, source, scenes, skip):
    result, per_scene = {}, {}
    for metric, _ in METRICS:
        a, b = [], []
        for scene in scenes:
            sources = data["scenes"][scene]
            if source not in sources or BASELINE not in sources:
                raise ValueError(f"Scene {scene!r} must contain {source!r} and {BASELINE!r}")
            a.append(metric_series(sources[source], metric, skip))
            b.append(metric_series(sources[BASELINE], metric, skip))
            per_scene.setdefault(scene, {})[metric] = mean(a[-1])
        delta, lo, hi = paired_bootstrap(a, b)
        if not math.isfinite(lo) or not math.isfinite(hi):
            verdict = "n/a"
        elif lo <= 0 <= hi:
            verdict = "tie"
        elif (metric == "dev" and hi < 0) or (metric != "dev" and lo > 0):
            verdict = "better"
        else:
            verdict = "worse"
        result[metric] = dict(value=mean([mean(values) for values in a]), delta=delta,
                              lo=lo, hi=hi, verdict=verdict)
    return result, per_scene


def cell(value):
    return str(value).replace("&", "&amp;").replace("|", "&#124;").replace("\n", " ")


def row(values):
    return "| " + " | ".join(cell(value) for value in values) + " |"


def number(value, digits, signed=False):
    if not math.isfinite(value):
        return "-"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def metric_cell(score, digits):
    value = number(score["value"], digits)
    delta = number(score["delta"], digits, signed=True)
    lo, hi = number(score["lo"], digits, True), number(score["hi"], digits, True)
    return f"{value} ({delta}, {score['verdict']}; CI [{lo}, {hi}])"


def collect(paths, sidecars, selected_scenes):
    models = {}
    for path in paths:
        data = json.loads(path.read_text())
        scenes = selected_scenes if selected_scenes is not None else list(data["scenes"])
        if not scenes or any(scene not in data["scenes"] for scene in scenes):
            raise ValueError(f"Requested scenes missing or empty in {path}")
        configs = {checkpoint_name(source): config
                   for source, config in data.get("configs", {}).items()}
        sources = dict.fromkeys(source for entries in data["scenes"].values() for source in entries)
        if BASELINE not in sources:
            raise ValueError(f"Missing baseline in {path}")
        for source in sources:
            if source in models:
                continue
            name = checkpoint_name(source)
            config = sidecars.get(name, configs.get(name, {})) if source != BASELINE else {}
            models[source] = dict(data=data, path=path, scenes=scenes, config=config)
    return models


def report(models, timings, skips, title):
    lines = [f"# {title}", ""]
    for skip in skips:
        scored = {source: score_source(model["data"], source, model["scenes"], skip)
                  for source, model in models.items()}
        order = [BASELINE] + sorted((source for source in models if source != BASELINE),
                                   key=lambda source: -scored[source][0]["ssim"]["delta"]
                                   if math.isfinite(scored[source][0]["ssim"]["delta"])
                                   else math.inf)
        scene_sets = list(dict.fromkeys(tuple(model["scenes"]) for model in models.values()))
        scene_text = "; ".join(", ".join(scenes) for scenes in scene_sets)
        lines += [f"## Frames >= {skip}", "",
                  f"Protocol: scenes {cell(scene_text)}; frames >= {skip}; baseline {BASELINE}. "
                  f"Temporal pairs end at t >= {max(skip, 1)}. Means weight scenes equally; "
                  "differences use the baseline in each model's eval JSON. "
                  "Paired within-scene block bootstrap: 95% CI, block 8, 2000 resamples, seed 0. "
                  "Lower |dev| is better. Fused ms is cuDNN total_with_display; "
                  "TRT ms is end-to-end TOTAL TRT.", "",
                  row(["model", "architecture", "fused ms", "TRT ms", "PSNR (Δ, verdict)",
                       "SSIM (Δ, verdict)", "|dev| (Δ, verdict)", "wins"]),
                  row(["---"] * 8)]
        for source in order:
            scores = scored[source][0]
            timing = timings.get(shape_key(models[source]["config"]), {})
            lines.append(row([source, architecture(models[source]["config"]),
                              number(timing.get("fused", float("nan")), 2),
                              number(timing.get("trt", float("nan")), 2),
                              *(metric_cell(scores[key], digits) for key, digits in METRICS),
                              sum(scores[key]["verdict"] == "better" for key, _ in METRICS)]))
        lines += ["", "### Per scene", "", row(["scene", "model", "PSNR", "SSIM"]),
                  row(["---"] * 4)]
        for scene in dict.fromkeys(scene for scenes in scene_sets for scene in scenes):
            for source in order:
                values = scored[source][1].get(scene)
                if values is not None:
                    lines.append(row([scene, source, number(values["psnr"], 2),
                                      number(values["ssim"], 4)]))
        lines.append("")
    lines += ["### Eval inputs (first occurrence per source)", ""]
    for source, model in models.items():
        lines.append(f"- {cell(source)}: {cell(model['path'])}; scenes {cell(', '.join(model['scenes']))}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", nargs="+", type=Path, required=True)
    ap.add_argument("--sidecars", nargs="+", type=Path, default=[])
    ap.add_argument("--bench", nargs="+", type=Path, default=[])
    ap.add_argument("--skip", type=int, action="append")
    ap.add_argument("--scenes", help="Comma-separated scene names")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title", default="Results")
    args = ap.parse_args()
    skips = args.skip if args.skip is not None else [0]
    if any(skip < 0 for skip in skips):
        ap.error("--skip must be nonnegative")
    scenes = list(dict.fromkeys(s.strip() for s in args.scenes.split(","))) if args.scenes else None
    try:
        models = collect(args.eval, read_sidecars(args.sidecars), scenes)
        output = report(models, read_bench(args.bench), skips, args.title)
        if args.out:
            args.out.write_text(output)
        else:
            print(output, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
