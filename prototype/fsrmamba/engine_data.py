"""Turn engine captures into training sequences for the accumulator.

Pairs each scene's FSR pass (low-res color + motion + depth) with its
ground-truth pass (native full-res render) frame-by-frame, producing the same
per-frame dicts `train.py` already consumes from the synthetic pipeline:

    lr     (rh, rw, 3)  low-res input color, tonemapped to [0,1)
    mv     (rh, rw, 2)  motion vectors -- engine UV convention, verified by
                        warp-alignment to match the synthetic one exactly
                        (scale 1.0, +x/+y), so used as-is
    depth  (rh, rw)
    gt     (uh, uw, 3)  native ground truth, tonemapped
    jitter (2,)

HDR note: engine color is linear HDR (values > 1). Both lr and gt are tonemapped
with the same Reinhard curve so the model trains and is scored in one consistent
[0,1) space -- the same space the FSR-vs-GT baseline (32.5 dB) was measured in.
"""

from __future__ import annotations

import glob
import os
import random

import torch
import torch.nn.functional as F

from .capture import load_frame


def _upsample_to(x: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    """Resize an (H,W) or (H,W,C) field to (h,w[,C])."""
    if x.dim() == 2:
        return F.interpolate(x[None, None], size=(h, w), mode=mode)[0, 0]
    return F.interpolate(x.permute(2, 0, 1)[None], size=(h, w), mode=mode)[0].permute(1, 2, 0)


def _tonemap(x: torch.Tensor) -> torch.Tensor:
    """Reinhard: linear HDR -> [0,1). Cheap, invertible, and the same curve used
    to report the FSR-vs-GT baseline, so numbers are comparable."""
    x = x.clamp(min=0.0)
    return x / (1.0 + x)


def load_engine_scene(scene_dir: str, device: str = "cpu", tonemap: bool = True,
                      cache: bool = True) -> list[dict]:
    """Load one scene's paired capture (``<scene>/fsr`` + ``<scene>/gt``).

    Decoding the packed rg11b10 HDR color is CPU-heavy (~17 s/scene), so the
    decoded frames are cached as float16 next to the capture and reused. Delete
    the ``_cache_*.pt`` files to force a re-decode.
    """
    cache_path = os.path.join(scene_dir, f"_cache_tm{int(tonemap)}.pt")
    if cache and os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return [{k: (v.float().to(device) if torch.is_tensor(v) else v) for k, v in f.items()}
                for f in data]

    fsr = sorted(glob.glob(os.path.join(scene_dir, "fsr", "frame_*.json")))
    gt = sorted(glob.glob(os.path.join(scene_dir, "gt", "frame_*.json")))
    if not fsr or not gt:
        raise FileNotFoundError(f"{scene_dir}: need both fsr/ and gt/ captures")
    n = min(len(fsr), len(gt))

    frames = []
    for i in range(n):
        f = load_frame(fsr[i])
        g = load_frame(gt[i])
        lr = f["lr"]
        gt_img = g["hr"]  # native full-res render = ground truth
        if tonemap:
            lr = _tonemap(lr)
            gt_img = _tonemap(gt_img)
        fsr_out = _tonemap(f["hr"]) if tonemap else f["hr"]  # captured FSR output = baseline

        # Keep mv/depth at *render* resolution here (small). The model wants them
        # at output res, but upsampling every full frame and holding it in RAM is
        # ~4x the memory and slow -- crop_sequence upsamples only the small crop.
        frames.append({
            "lr": lr,                 # render resolution
            "mv": f["mv"],            # render resolution
            "depth": f["depth"],      # render resolution
            "gt": gt_img,             # output resolution
            "fsr_out": fsr_out,       # actual FSR 3.1.4 output, the number to beat
            "jitter": f["jitter"],
        })

    if cache:
        half = [{k: (v.half() if torch.is_tensor(v) else v) for k, v in f.items()} for f in frames]
        torch.save(half, cache_path)

    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in f.items()} for f in frames]


def crop_sequence(frames: list[dict], crop_render: int, scale: int = 2,
                  rng: random.Random | None = None) -> list[dict]:
    """One fixed random crop applied to the whole sequence.

    The crop location is constant across frames so the per-pixel recurrent state
    stays registered frame-to-frame. ``lr``/``mv``/``depth`` are cropped at render
    res, then mv+depth are upsampled to the output-res crop (the model wants them
    at output res); ``gt``/``fsr_out`` are cropped at the ``scale``x region.

    Motion vectors are rescaled: they are stored as *full-frame* UV offsets, but
    the model builds its sampling grid in [0,1] over the *crop*, so a full-frame
    offset must be multiplied by (full_size / crop_size) to stay physically the
    same displacement.
    """
    rng = rng or random
    rh, rw, _ = frames[0]["lr"].shape
    ch = cw = crop_render
    if rh < ch or rw < cw:
        return frames

    y = rng.randint(0, rh - ch)
    x = rng.randint(0, rw - cw)
    Y, X, CH, CW = y * scale, x * scale, ch * scale, cw * scale
    mv_rescale = torch.tensor([rw / cw, rh / ch], device=frames[0]["mv"].device)

    out = []
    for f in frames:
        mv_c = f["mv"][y:y + ch, x:x + cw] * mv_rescale
        depth_c = f["depth"][y:y + ch, x:x + cw]
        c = {
            "lr": f["lr"][y:y + ch, x:x + cw],
            "mv": _upsample_to(mv_c, CH, CW, "bilinear"),      # -> output-res crop
            "depth": _upsample_to(depth_c, CH, CW, "nearest"),  # -> output-res crop
            "gt": f["gt"][Y:Y + CH, X:X + CW],
            "jitter": f["jitter"],
        }
        if "fsr_out" in f:
            c["fsr_out"] = f["fsr_out"][Y:Y + CH, X:X + CW]
        out.append(c)
    return out


def list_scenes(captures_dir: str) -> list[str]:
    """Scene names under ``captures_dir`` that have both passes captured."""
    names = []
    for d in sorted(glob.glob(os.path.join(captures_dir, "*"))):
        if os.path.isdir(os.path.join(d, "fsr")) and os.path.isdir(os.path.join(d, "gt")):
            names.append(os.path.basename(d))
    return names
