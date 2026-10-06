"""Locate spatial error and ground-truth-referenced temporal excess."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fsrmamba.config import load_checkpoint
from fsrmamba.engine_data import list_scenes, load_engine_scene
from fsrmamba.evalkit import run_scene, temporal_diff_map


REGIONS = ("disoccluded", "edge", "moving", "flat")
SOURCES = ("model", "FSR")


def region_masks(frame, model):
    gt, mv = frame["gt"].float(), frame["mv"].float()
    h, w = gt.shape[:2]
    luma = gt @ gt.new_tensor([0.2126, 0.7152, 0.0722])
    dx = F.pad(luma[:, 1:] - luma[:, :-1], (0, 1))
    dy = F.pad(luma[1:] - luma[:-1], (0, 0, 0, 1))
    gradient = (dx.square() + dy.square()).sqrt()
    reset = model._last_reset.float()
    if reset.shape[-2:] != (h, w):
        reset = F.interpolate(reset, size=(h, w), mode="nearest")
    disoccluded = reset[0, 0] > 0.5
    edge = gradient > torch.quantile(gradient.flatten(), 0.9)
    moving = torch.linalg.vector_norm(mv * mv.new_tensor([w, h]), dim=-1) > 2.0
    masks, occupied = {}, torch.zeros_like(disoccluded)
    for name, candidate in zip(REGIONS[:-1], (disoccluded, edge, moving)):
        masks[name] = candidate & ~occupied
        occupied |= masks[name]
    masks["flat"] = ~occupied
    return masks


def new_totals():
    return {source: {region: {key: 0.0 for key in
                             ("pixels", "l1_sum", "sq_sum", "valid", "ti_sum", "gt_ti_sum", "excess")}
                     for region in REGIONS} for source in SOURCES}


def region_summary(totals):
    result = {}
    for source, regions in totals.items():
        pixels = sum(r["pixels"] for r in regions.values())
        sq_sum = sum(r["sq_sum"] for r in regions.values())
        excess = sum(r["excess"] for r in regions.values())
        result[source] = {}
        for name, r in regions.items():
            mse = r["sq_sum"] / r["pixels"] if r["pixels"] else 0.0
            ti = r["ti_sum"] / r["valid"] if r["valid"] else 0.0
            gt_ti = r["gt_ti_sum"] / r["valid"] if r["valid"] else 0.0
            result[source][name] = {
                "pixel_fraction": r["pixels"] / pixels if pixels else 0.0,
                "l1": r["l1_sum"] / r["pixels"] if r["pixels"] else 0.0,
                "psnr": (-10 * math.log10(mse) if mse > 1e-12 else float("inf"))
                        if r["pixels"] else float("nan"),
                "ti": ti, "gt_ti": gt_ti, "dev": ti - gt_ti,
                "squared_error_share": r["sq_sum"] / sq_sum if sq_sum else 0.0,
                "positive_excess_share": r["excess"] / excess if excess else 0.0,
                "pixels": int(r["pixels"]), "valid_pixels": int(r["valid"]),
            }
    return result


def accumulate(scene, model, device="cpu", max_frames=None, totals=None):
    """Score both sources on identical masks; optionally add to prior scene totals."""
    if totals is None:
        totals = new_totals()
    prev_fsr = None

    def hook(i, frame, out, prev_out, prev_gt, current_model):
        nonlocal prev_fsr
        fsr = frame["fsr_out"]
        if i > 0:
            masks = region_masks(frame, current_model)
            gt_diff, valid = temporal_diff_map(frame["gt"], prev_gt, frame["mv"])
            for source, curr, prev in (("model", out, prev_out), ("FSR", fsr, prev_fsr)):
                diff, _ = temporal_diff_map(curr, prev, frame["mv"])
                error = curr.float().clamp(0, 1) - frame["gt"].float().clamp(0, 1)
                l1, square = error.abs().mean(-1), error.square().mean(-1)
                positive = (diff - gt_diff).clamp(min=0)
                for name, mask in masks.items():
                    r = totals[source][name]
                    selected = mask & valid
                    r["pixels"] += int(mask.sum())
                    r["valid"] += int(selected.sum())
                    r["l1_sum"] += float(l1[mask].sum())
                    r["sq_sum"] += float(square[mask].sum())
                    r["ti_sum"] += float(diff[selected].sum())
                    r["gt_ti_sum"] += float(gt_diff[selected].sum())
                    r["excess"] += float(positive[selected].sum())
        prev_fsr = fsr

    run_scene(scene, model, device, max_frames, hook)
    return region_summary(totals)


def print_tables(report):
    for source, regions in report.items():
        print(f"\n{source}")
        print("region          fraction        L1      PSNR     instab  GT instab        dev")
        for name, r in regions.items():
            print(f"{name:14s} {r['pixel_fraction']:9.6f} {r['l1']:9.5f} {r['psnr']:9.3f} "
                  f"{r['ti']:10.5f} {r['gt_ti']:10.5f} {r['dev']:+10.5f}")
    print("\nshare of total (positive excess = sum of per-pixel max(instab - GT instab, 0))")
    print("region         model sq error  FSR sq error  model excess  FSR excess")
    for name in REGIONS:
        m, f = report["model"][name], report["FSR"][name]
        print(f"{name:14s} {m['squared_error_share']:14.6f} {f['squared_error_share']:13.6f} "
              f"{m['positive_excess_share']:13.6f} {f['positive_excess_share']:11.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = ap.parse_args()
    if args.frames is not None and args.frames < 2:
        ap.error("Region diagnosis requires at least two frames")
    names = args.scenes.split(",") if args.scenes else list_scenes(args.engine_data)[:3]
    if not names:
        ap.error("No capture scenes found")
    models, totals = {}, new_totals()
    for name in names:
        scene = load_engine_scene(str(Path(args.engine_data) / name), half=True, mmap=True)
        if len(scene) < 2:
            raise ValueError(f"Region diagnosis requires at least two frames: {name}")
        geometry = (tuple(scene[0]["lr"].shape[:2]), tuple(scene[0]["gt"].shape[:2]))
        if geometry not in models:
            models[geometry], _ = load_checkpoint(args.ckpt, *geometry, device=args.device)
        accumulate(scene, models[geometry], args.device, args.frames, totals)
    print_tables(region_summary(totals))


if __name__ == "__main__":
    main()
