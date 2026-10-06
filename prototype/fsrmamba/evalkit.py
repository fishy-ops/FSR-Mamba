"""Streaming full-frame scoring and paired temporal block confidence intervals."""

from __future__ import annotations

from contextlib import nullcontext
from itertools import islice

import torch
import torch.nn.functional as F

from .engine_data import _upsample_to
from .config import new_state
from .fast import FastAccumulator
from .fused import FusedFast, state_rgb
from .mamba import MambaAccumulator
from .metrics import psnr, ssim, temporal_deviation, temporal_instability
from .phase import PhaseAccumulator


def full_frame(f, device):
    h, w = f["gt"].shape[:2]
    result = {key: f[key].float().to(device) for key in ("lr", "gt", "fsr_out")}
    result["mv"] = _upsample_to(f["mv"].float().to(device), h, w, "bilinear")
    result["depth"] = _upsample_to(f["depth"].float().to(device), h, w, "nearest")
    jitter = f["jitter"]
    result["jitter"] = jitter.float().to(device) if torch.is_tensor(jitter) else jitter
    return result


def temporal_diff_map(curr, prev, mv):
    curr, prev, mv = curr.float(), prev.float(), mv.float()
    h, w, _ = curr.shape
    ys = (torch.arange(h, device=curr.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=curr.device, dtype=torch.float32) + 0.5) / w
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    uv = torch.stack((gx, gy), dim=-1) + mv
    valid = (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
    warped = F.grid_sample(prev.permute(2, 0, 1).unsqueeze(0),
                           (uv * 2 - 1).unsqueeze(0), mode="bilinear",
                           padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
    return (curr - warped).abs().mean(dim=-1), valid


@torch.no_grad()
def run_scene(scene, model=None, device="cpu", max_frames=None, on_frame=None, amp=False,
              native_lr=False, fused_backend=None):
    """Keep every frame's metrics; reference fusion is an internal CPU test backend."""
    if fused_backend not in (None, "cuda", "reference"):
        raise ValueError("Unknown fused backend")
    engine = FusedFast(model) if fused_backend is not None else None
    native_lr = native_lr or engine is not None
    if native_lr and isinstance(model, MambaAccumulator):
        print("--native-lr ignored for MambaAccumulator; using output-resolution auxiliaries.")
    use_lr = native_lr and isinstance(model, (FastAccumulator, PhaseAccumulator))
    scores = {key: [] for key in ("psnr", "ssim", "ti", "gt_ti", "dev")}
    prev_out, prev_depth, prev_gt, state = None, None, None, None
    frames = scene if max_frames is None else islice(scene, max_frames)
    for i, raw in enumerate(frames):
        frame = full_frame(raw, device)
        if model is None:
            out = frame["fsr_out"]
        else:
            if state is None:
                state = engine.init_state() if engine is not None else new_state(model, device)
            mv = raw["mv"].float().to(device) if use_lr else frame["mv"]
            depth = raw["depth"].float().to(device) if use_lr else frame["depth"]
            ctx = (torch.autocast(torch.device(device).type, dtype=torch.float16)
                   if amp else nullcontext())
            if engine is not None:
                step = engine.step_reference if fused_backend == "reference" else engine.step
                _, state = step(state, frame["lr"], mv, depth, raw["jitter"])
                out = state_rgb(state)
            else:
                with ctx:
                    out, state = model(state, frame["lr"], mv, depth,
                                       frame["jitter"], prev_depth_hr=prev_depth)
        out = out.float()
        scores["psnr"].append(float(psnr(out, frame["gt"])))
        scores["ssim"].append(float(ssim(out, frame["gt"])))
        if prev_out is not None:
            dev, gt_ti = temporal_deviation(out, prev_out, frame["gt"], prev_gt, frame["mv"])
            scores["ti"].append(float(temporal_instability(out, prev_out, frame["mv"])))
            scores["gt_ti"].append(float(gt_ti))
            scores["dev"].append(float(dev))
        if on_frame is not None:
            on_frame(i, frame, out, prev_out, prev_gt, model)
        prev_out, prev_gt, prev_depth = out, frame["gt"], frame["depth"]
    return scores


def _bootstrap_means(diff, block, n_boot, generator):
    n = diff.numel()
    block = min(block, n)
    if bool((diff == diff[0]).all()):
        return diff[0].expand(n_boot)
    offsets = torch.arange(block)
    counts = (n + block - 1) // block
    batches = []
    for start in range(0, n_boot, 128):
        starts = torch.randint(n - block + 1, (min(128, n_boot - start), counts),
                               generator=generator)
        indices = (starts[..., None] + offsets).flatten(1)[:, :n]
        batches.append(diff[indices].mean(dim=1))
    return torch.cat(batches)


def paired_scene_bootstrap(a_scenes, b_scenes, block=8, n_boot=2000, seed=0):
    """Resample each scene independently and combine with frame-count weights."""
    if block < 1 or n_boot < 1:
        raise ValueError("block and n_boot must be positive")
    if len(a_scenes) != len(b_scenes):
        raise ValueError("Scene counts differ")
    generator = torch.Generator().manual_seed(seed)
    total, observed = 0, 0.0
    samples = torch.zeros(n_boot, dtype=torch.float64)
    for a, b in zip(a_scenes, b_scenes):
        if len(a) != len(b):
            raise ValueError("Paired series must have equal lengths")
        if not a:
            continue
        diff = torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)
        if not bool(torch.isfinite(diff).all()):
            raise ValueError("Bootstrap requires finite paired values")
        n = len(a)
        observed += diff.sum().item()
        total += n
        samples += _bootstrap_means(diff, block, n_boot, generator) * n
    if not total:
        return float("nan"), float("nan"), float("nan")
    samples /= total
    lo, hi = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return observed / total, lo, hi


def paired_block_bootstrap(a, b, block=8, n_boot=2000, seed=0):
    return paired_scene_bootstrap([a], [b], block, n_boot, seed)


def scored_frames(per_frame, skip=0):
    """Temporal entry zero is the pair ending at frame one."""
    if skip < 0:
        raise ValueError("skip must be nonnegative")
    return {key: values[max(skip - 1, 0) if key in ("ti", "gt_ti", "dev") else skip:]
            for key, values in per_frame.items()}


def summarise(per_frame, skip=0):
    per_frame = scored_frames(per_frame, skip)
    result = {key: sum(values) / len(values) if values else 0.0
              for key, values in per_frame.items()}
    dev = per_frame["dev"]
    result["abs_dev"] = sum(abs(x) for x in dev) / len(dev) if dev else 0.0
    return result
