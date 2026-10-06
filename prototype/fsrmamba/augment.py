"""Clip-wide exposure and dihedral transforms for images and sampling geometry."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


IDENTITY = (False, False, False)


def random_op(rng):
    """Choose each of the eight transforms with equal probability."""
    bits = rng.randrange(8)
    return (bool(bits & 4), bool(bits & 2), bool(bits & 1))


def augment_sequence(frames, op):
    """Transpose first, then flip in the resulting x/y axes; never write inputs."""
    transpose, flip_x, flip_y = op

    def spatial(x):
        if transpose:
            x = x.transpose(0, 1)
        dims = ([1] if flip_x else []) + ([0] if flip_y else [])
        return x.flip(dims) if dims else x.clone()

    result = []
    for frame in frames:
        out = dict(frame)
        for key in ("lr", "gt", "fsr_out", "depth", "mv"):
            if key in frame:
                out[key] = spatial(frame[key])
        mv = out["mv"]
        # UV is normalised per axis: transposing even a non-square crop only
        # swaps components, because each component carries its axis's divisor.
        if transpose:
            mv = mv[..., [1, 0]]
        out["mv"] = mv * mv.new_tensor([-1 if flip_x else 1, -1 if flip_y else 1])
        jitter = frame["jitter"]
        jx, jy = (jitter[1], jitter[0]) if transpose else jitter
        values = (-jx if flip_x else jx, -jy if flip_y else jy)
        if torch.is_tensor(jitter):
            out["jitter"] = torch.stack(values)
        else:
            out["jitter"] = type(jitter)(values)
        result.append(out)
    return result


def exposure_range(value):
    """Parse positive, finite log-uniform gain bounds."""
    try:
        lo, hi = (float(v) for v in value.split(","))
    except ValueError as exc:
        raise ValueError("exposure augmentation must be LO,HI") from exc
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo <= hi:
        raise ValueError("exposure augmentation requires finite 0 < LO <= HI")
    return lo, hi


def exposure_gain(image, gain):
    """Apply a linear-light gain to per-channel Reinhard RGB without writing inputs."""
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("exposure gain must be positive and finite")
    x = image.clamp(max=1 - 1 / 1024)
    linear = x / (1 - x) * gain
    return linear / (1 + linear)


def augment_exposure(frames, bounds, rng):
    """Draw one gain for the whole clip; disabled augmentation consumes no randomness."""
    if bounds is None:
        return frames
    lo, hi = bounds
    gain = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    result = []
    for frame in frames:
        out = dict(frame)
        for key in ("lr", "gt", "fsr_out"):
            if key in frame:
                out[key] = exposure_gain(frame[key], gain)
        result.append(out)
    return result


def luma_range(image, footprint=1):
    """Local 3x3 luma range, optionally including every sub-pixel of a footprint."""
    rgb = image.float()
    luma = (.25 * rgb[..., 0] + .5 * rgb[..., 1] + .25 * rgb[..., 2])[None, None]
    hi = F.max_pool2d(luma, footprint, footprint)
    lo = -F.max_pool2d(-luma, footprint, footprint)
    return (F.max_pool2d(hi, 3, 1, 1) + F.max_pool2d(-lo, 3, 1, 1))[0, 0]


def dither_zone(frames, rng, threshold=.08, region_fraction=.6):
    """Select connected edge regions once over the clip's swept edge footprint."""
    h, w, _ = frames[0]["lr"].shape
    edges = torch.zeros((h, w), dtype=torch.bool, device=frames[0]["lr"].device)
    for frame in frames:
        if frame["lr"].shape != (h, w, 3) or frame["gt"].shape != (2*h, 2*w, 3):
            raise ValueError("dither augmentation requires RGB GT at 2x render resolution")
        edges |= luma_range(frame["gt"], 2) > threshold
    edges = F.max_pool2d(edges.float()[None, None], 3, 1, 1)[0, 0].bool()
    pending = edges.cpu().flatten().tolist()
    selected = [False] * (h*w)
    for start in range(h*w):
        if not pending[start]:
            continue
        keep = rng.random() < region_fraction
        pending[start] = False
        stack = [start]
        while stack:
            at = stack.pop()
            selected[at] = keep
            y, x = divmod(at, w)
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < h and 0 <= nx < w and pending[ny*w+nx]:
                    pending[ny*w+nx] = False
                    stack.append(ny*w+nx)
    return torch.tensor(selected, dtype=torch.bool, device=edges.device).reshape(h, w)


def augment_dither(frames, probability, rng, threshold=.08, region_fraction=.6):
    """Independent uniform GT sub-pixel draws; targets and sampling geometry stay intact."""
    if not 0 <= probability <= 1 or not 0 <= region_fraction <= 1:
        raise ValueError("dither probabilities must be in [0, 1]")
    if probability == 0 or rng.random() >= probability or not frames:
        return frames
    zone = dither_zone(frames, rng, threshold, region_fraction)
    h, w = zone.shape
    generator = torch.Generator(device=zone.device).manual_seed(rng.getrandbits(63))
    result = []
    for frame in frames:
        taps = frame["gt"].reshape(h, 2, w, 2, 3).permute(0, 2, 1, 3, 4).reshape(h, w, 4, 3)
        choice = torch.randint(4, (h, w, 1, 1), generator=generator, device=zone.device)
        sample = taps.gather(2, choice.expand(h, w, 1, 3)).squeeze(2).to(frame["lr"].dtype)
        result.append(dict(frame, lr=torch.where(zone[..., None], sample, frame["lr"])))
    return result
