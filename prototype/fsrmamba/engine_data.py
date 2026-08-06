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
                      cache: bool = True, half: bool = False,
                      mmap: bool | None = None) -> list[dict]:
    """Load one scene's paired capture (``<scene>/fsr`` + ``<scene>/gt``).

    Decoding the packed rg11b10 HDR color is CPU-heavy (~17 s/scene), so the
    decoded frames are cached as float16 next to the capture and reused. Delete
    the ``_cache_*.pt`` files to force a re-decode.

    ``half=True`` keeps the frames in the cache's native float16 instead of
    expanding to float32. A 1080p frame is ~62 MB in float32 across all buffers, so
    ten 96-frame scenes is ~59 GB and does not fit in 64 GB of RAM -- fp16 halves
    that to ~30 GB. Callers that ask for half MUST cast to float before doing real
    arithmetic (`crop_sequence` does this for the crop it returns); fp16 CPU math is
    both slow and, for the metrics, not precise enough to trust.

    ``mmap`` memory-maps the cache instead of reading it into anonymous RAM, and
    defaults on whenever ``half`` is set and the frames stay on the CPU. fp16 alone
    stopped being enough once the dataset grew to 24 sequences (~70 GB of cache
    against 64 GB of RAM) -- eager loading thrashes. Mapped pages are backed by the
    file, so the OS evicts cold scenes under pressure instead of swapping, and the
    load itself drops from ~40 s to ~6 s per scene because nothing is read until it
    is touched. Every consumer copies out with ``.float()`` before doing arithmetic,
    so the read-only mapping is never written through.
    """
    cache_path = os.path.join(scene_dir, f"_cache_tm{int(tonemap)}.pt")
    if cache and os.path.exists(cache_path):
        if mmap is None:
            mmap = half and str(device) == "cpu"
        data = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=mmap)

        def conv(v):
            if not torch.is_tensor(v):
                return v
            return v.to(device) if half else v.float().to(device)

        return [{k: conv(v) for k, v in f.items()} for f in data]

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

        # SUPERSAMPLED GROUND TRUTH.
        # The GT pass may be rendered at a higher resolution than the FSR pass (see
        # capture_scenes.py --gt-res). Area-downsampling it to the FSR output size is
        # what turns those extra samples into anti-aliasing: a 2x GT gives 4 samples
        # per output pixel. This matters more than it sounds -- the default 1-sample
        # GT is itself aliased, and a model cannot learn to be cleaner than its own
        # target. Thin geometry is where it shows: at 1 spp overhead wires break into
        # dotted fragments, at 4 spp they are continuous.
        # Done here rather than at use time so the cache stores the final target and
        # every downstream consumer sees a consistent gt/lr ratio.
        gh, gw = gt_img.shape[:2]
        oh, ow = fsr_out.shape[:2]
        if (gh, gw) != (oh, ow):
            if gh % oh or gw % ow:
                raise ValueError(
                    f"{scene_dir}: GT {gw}x{gh} is not an integer multiple of the FSR "
                    f"pass {ow}x{oh}. The passes were rendered at mismatched aspect "
                    f"ratios and their frames do not correspond -- recapture with "
                    f"matching aspect (capture_scenes.py enforces this).")
            gt_img = F.interpolate(gt_img.permute(2, 0, 1)[None], size=(oh, ow),
                                   mode="area")[0].permute(1, 2, 0)

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
                  rng: random.Random | None = None, edge_bias: float = 0.0) -> list[dict]:
    """One fixed random crop applied to the whole sequence.

    The crop location is constant across frames so the per-pixel recurrent state
    stays registered frame-to-frame. ``lr``/``mv``/``depth`` are cropped at render
    res, then mv+depth are upsampled to the output-res crop (the model wants them
    at output res); ``gt``/``fsr_out`` are cropped at the ``scale``x region.

    Motion vectors are rescaled: they are stored as *full-frame* UV offsets, but
    the model builds its sampling grid in [0,1] over the *crop*, so a full-frame
    offset must be multiplied by (full_size / crop_size) to stay physically the
    same displacement.

    ``edge_bias``: with this probability, snap the crop to touch a random frame
    edge. A uniform interior crop almost never covers the true frame border, yet
    that border (where camera motion pushes in new, history-less content) is
    exactly where the full-frame model is weakest and loses PSNR to FSR. Training
    on edge-anchored crops teaches real disocclusion handling; unlike an interior
    crop -- where a pixel leaving the crop still has valid content just outside it
    -- a pixel leaving a true frame edge IS a genuine disocclusion, so the crop's
    off-edge = disocclusion assumption is physically correct here.
    """
    rng = rng or random
    rh, rw, _ = frames[0]["lr"].shape
    ch = cw = crop_render
    if rh < ch or rw < cw:
        return frames

    y = rng.randint(0, rh - ch)
    x = rng.randint(0, rw - cw)
    if edge_bias > 0.0 and rng.random() < edge_bias:
        side = rng.choice(("top", "bottom", "left", "right"))
        if side == "top":
            y = 0
        elif side == "bottom":
            y = rh - ch
        elif side == "left":
            x = 0
        else:
            x = rw - cw
    Y, X, CH, CW = y * scale, x * scale, ch * scale, cw * scale
    mv_rescale = torch.tensor([rw / cw, rh / ch], device=frames[0]["mv"].device)

    out = []
    for f in frames:
        # Cast to float32 here, not upstream: the full sequences may be held in fp16
        # to fit in RAM (see load_engine_scene's `half`), but every consumer of a crop
        # -- the resolve, the metrics, the warp -- needs float32. The crop is tiny, so
        # converting at this boundary costs nothing and keeps fp16 out of the maths.
        mv_c = f["mv"][y:y + ch, x:x + cw].float() * mv_rescale
        depth_c = f["depth"][y:y + ch, x:x + cw].float()
        c = {
            "lr": f["lr"][y:y + ch, x:x + cw].float(),
            "mv": _upsample_to(mv_c, CH, CW, "bilinear"),      # -> output-res crop
            "depth": _upsample_to(depth_c, CH, CW, "nearest"),  # -> output-res crop
            "gt": f["gt"][Y:Y + CH, X:X + CW].float(),
            "jitter": f["jitter"],
        }
        if "fsr_out" in f:
            c["fsr_out"] = f["fsr_out"][Y:Y + CH, X:X + CW].float()
        out.append(c)
    return out


def list_scenes(captures_dir: str) -> list[str]:
    """Scene names under ``captures_dir`` that are loadable.

    A scene counts as loadable if it has **either** both raw passes (``fsr/`` and
    ``gt/``) **or** a decoded ``_cache_tm1.pt``. The cache alone is enough because
    `load_engine_scene` checks it first and never touches the raw dumps when it is
    present -- and the raw dumps are ~2x the size of the cache, so they get deleted
    once a scene is cached. Requiring the raw directories here made every
    cached-and-cleaned scene invisible to training (found the hard way: a
    seven-scene capture set reported exactly one usable scene).
    """
    names = []
    for d in sorted(glob.glob(os.path.join(captures_dir, "*"))):
        if not os.path.isdir(d):
            continue
        has_raw = os.path.isdir(os.path.join(d, "fsr")) and os.path.isdir(os.path.join(d, "gt"))
        has_cache = bool(glob.glob(os.path.join(d, "_cache_tm*.pt")))
        if has_raw or has_cache:
            names.append(os.path.basename(d))
    return names
