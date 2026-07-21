"""Load engine-captured frames dumped by the instrumented FSR sample.

The FSR sample (samples/fsrapi/fsrapirendermodule.cpp, FSR-Mamba capture block)
writes, per frame, four raw buffers plus a JSON sidecar:

    frame_NNNN_color_lr.bin   low-res input color   (render-res sub-rect)
    frame_NNNN_color_hr.bin   FSR upscaled output   (upscale-res sub-rect)
    frame_NNNN_depth.bin      depth                 (render-res sub-rect)
    frame_NNNN_motion.bin     motion vectors        (render-res sub-rect)
    frame_NNNN.json           { renderWidth/Height, upscaleWidth/Height,
                                jitterX/Y, and per-buffer {w,h,channels,dtype,
                                stride} }

Each .bin is tight-packed (row padding already removed engine-side) at the
buffer's FULL texture size w x h. FSR renders into display-sized targets and
uses only a top-left sub-rect, so this crops each buffer to the valid region:
render-res for LR/depth/motion, upscale-res for HR.

This mirrors the synthetic `Scene.render` outputs (see synth.py) closely enough
that train.py can consume either: color in [0,1]-ish linear RGB, motion as
per-pixel UV-space offsets, depth as a single channel.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

_NP_DTYPE = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "float16": np.float16,
    "float32": np.float32,
}


def _load_buffer(bin_path: str, desc: dict) -> np.ndarray:
    """Read one tight-packed raw buffer into (H, W, C) float32.

    `desc` is the per-buffer metadata block from the JSON sidecar.
    """
    w, h, ch, dtype = desc["w"], desc["h"], desc["channels"], desc["dtype"]

    if dtype in _NP_DTYPE:
        raw = np.fromfile(bin_path, dtype=_NP_DTYPE[dtype])
        arr = raw.reshape(h, w, ch).astype(np.float32)
        if dtype == "uint8":
            arr /= 255.0
        elif dtype == "uint16":
            arr /= 65535.0
        return arr

    # Packed 32-bit formats: read as u32 and unpack the bit fields.
    if dtype == "rg11b10":
        u = np.fromfile(bin_path, dtype=np.uint32).reshape(h, w)
        r = _float11(u & 0x7FF)
        g = _float11((u >> 11) & 0x7FF)
        b = _float10((u >> 22) & 0x3FF)
        return np.stack([r, g, b], axis=-1).astype(np.float32)
    if dtype == "rgb10a2":
        u = np.fromfile(bin_path, dtype=np.uint32).reshape(h, w)
        r = ((u >> 0) & 0x3FF) / 1023.0
        g = ((u >> 10) & 0x3FF) / 1023.0
        b = ((u >> 20) & 0x3FF) / 1023.0
        a = ((u >> 30) & 0x3) / 3.0
        return np.stack([r, g, b, a], axis=-1).astype(np.float32)

    raise ValueError(f"{bin_path}: unsupported dtype {dtype!r} (extend capture.py)")


def _float10(bits: np.ndarray) -> np.ndarray:
    """Unpack unsigned 10-bit float (5-bit exp, 5-bit mantissa, no sign)."""
    return _unpack_small_float(bits, exp_bits=5, mant_bits=5)


def _float11(bits: np.ndarray) -> np.ndarray:
    """Unpack unsigned 11-bit float (5-bit exp, 6-bit mantissa, no sign)."""
    return _unpack_small_float(bits, exp_bits=5, mant_bits=6)


def _unpack_small_float(bits: np.ndarray, exp_bits: int, mant_bits: int) -> np.ndarray:
    bits = bits.astype(np.uint32)
    exp = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mant = bits & ((1 << mant_bits) - 1)
    bias = (1 << (exp_bits - 1)) - 1
    out = np.zeros(bits.shape, dtype=np.float32)
    m = mant.astype(np.float32) / float(1 << mant_bits)
    normal = exp > 0
    out[normal] = (1.0 + m[normal]) * np.exp2((exp[normal].astype(np.float32) - bias))
    sub = (exp == 0) & (mant > 0)
    out[sub] = m[sub] * np.exp2(float(1 - bias))
    return out


def load_frame(json_path: str) -> dict:
    """Load one captured frame, cropped to valid regions, as torch tensors.

    Returns a dict shaped like a `render_sequence` frame from train.py:
        lr     (rh, rw, 3)  low-res color, RGB
        gt-ish: there is no supersampled GT from the engine -- `hr` is FSR's own
                output, which is the *baseline to beat*, not ground truth. Engine
                captures have no reference; see README note on how to evaluate.
        hr     (uh, uw, 3)  FSR upscaled output, RGB
        depth  (rh, rw)     depth
        mv     (rh, rw, 2)  motion vectors (raw engine units; scale at use site)
        jitter (2,)         subpixel jitter for this frame
    """
    with open(json_path) as f:
        meta = json.load(f)

    root = os.path.splitext(json_path)[0]
    rw, rh = meta["renderWidth"], meta["renderHeight"]
    uw, uh = meta["upscaleWidth"], meta["upscaleHeight"]

    color_lr = _load_buffer(root + "_color_lr.bin", meta["color_lr"])[:rh, :rw, :3]
    color_hr = _load_buffer(root + "_color_hr.bin", meta["color_hr"])[:uh, :uw, :3]
    depth = _load_buffer(root + "_depth.bin", meta["depth"])[:rh, :rw, 0]
    motion = _load_buffer(root + "_motion.bin", meta["motion"])[:rh, :rw, :2]

    return {
        "lr": torch.from_numpy(np.ascontiguousarray(color_lr)),
        "hr": torch.from_numpy(np.ascontiguousarray(color_hr)),
        "depth": torch.from_numpy(np.ascontiguousarray(depth)),
        "mv": torch.from_numpy(np.ascontiguousarray(motion)),
        "jitter": (float(meta["jitterX"]), float(meta["jitterY"])),
        "meta": meta,
    }


def load_sequence(capture_dir: str) -> list[dict]:
    """Load every captured frame in a directory, ordered by frame index."""
    jsons = sorted(glob.glob(os.path.join(capture_dir, "frame_*.json")))
    if not jsons:
        raise FileNotFoundError(f"no frame_*.json under {capture_dir}")
    return [load_frame(j) for j in jsons]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect an engine capture directory.")
    ap.add_argument("dir")
    args = ap.parse_args()

    seq = load_sequence(args.dir)
    print(f"loaded {len(seq)} frames from {args.dir}\n")
    f0 = seq[0]
    for k in ("lr", "hr", "depth", "mv"):
        t = f0[k]
        print(f"  {k:6s} shape={tuple(t.shape)}  "
              f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.float().mean().item():.4f}")
    print(f"  jitter {f0['jitter']}")
    print(f"  render {f0['meta']['renderWidth']}x{f0['meta']['renderHeight']}  "
          f"-> upscale {f0['meta']['upscaleWidth']}x{f0['meta']['upscaleHeight']}")
