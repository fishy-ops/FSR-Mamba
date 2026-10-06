"""Analyse FSMDUMP1 captures against a FastAccumulator checkpoint.

Plane offsets are relative to the raw payload following the UTF-8 JSON. Rows
include D3D12 pitch padding. dxgi_format describes the original texture and the
optional footprint_format describes the physical layout copied for plane zero.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
import numpy as np
import torch
import torch.nn.functional as F
from fsrmamba.config import build_model, load_checkpoint
from fsrmamba.fast import FastAccumulator
from fsrmamba.metrics import psnr, ssim, temporal_instability


_FLOAT_FORMATS = {10: ("<f2", 4), 2: ("<f4", 4), 34: ("<f2", 2),
                  16: ("<f4", 2), 41: ("<f4", 1), 39: ("<f4", 1),
                  40: ("<f4", 1), 54: ("<f2", 1)}


def decode_plane(raw, plane):
    resource_format = int(plane["dxgi_format"])
    if resource_format not in _FLOAT_FORMATS and resource_format not in (19, 20, 21, 26, 24):
        raise ValueError(f"plane {plane['name']}: unsupported DXGI format {resource_format}")
    fmt = int(plane.get("footprint_format", resource_format))
    w, h, pitch = (int(plane[k]) for k in ("width", "height", "row_pitch"))
    if fmt in _FLOAT_FORMATS:
        dtype, channels = _FLOAT_FORMATS[fmt]
        bpp = np.dtype(dtype).itemsize * channels
    elif fmt in (19, 20, 21, 26, 24):
        # Legacy files without footprint_format use the DXGI interleaved layout.
        # Proxy captures carry the planar footprint format (e.g. 39/41) instead.
        bpp = 8 if fmt in (19, 20, 21) else 4
    else:
        raise ValueError(f"plane {plane['name']}: unsupported DXGI format {fmt}")
    offset, size = int(plane["offset"]), int(plane["bytes"])
    if min(w, h) <= 0 or pitch < w * bpp or size != pitch * h or offset < 0 or offset + size > len(raw):
        raise ValueError(f"plane {plane['name']}: invalid dimensions, pitch, or payload bounds")
    rows = np.frombuffer(raw, np.uint8, count=size, offset=offset).reshape(h, pitch)
    pixels = rows[:, :w * bpp].copy()
    if fmt in _FLOAT_FORMATS:
        array = pixels.view(dtype).reshape(h, w, channels).astype(np.float32)
    elif fmt in (19, 20, 21):
        array = pixels.reshape(h, w, 8)[..., :4].copy().view("<f4").reshape(h, w, 1)
    else:
        packed = pixels.view("<u4").reshape(h, w)
        if fmt == 24:
            array = np.stack([(packed >> shift) & mask for shift, mask in
                              ((0, 1023), (10, 1023), (20, 1023), (30, 3))], -1).astype(np.float32)
            array /= np.array([1023, 1023, 1023, 3], np.float32)
        else:
            decoded = []
            for shift, mantissa_bits in ((0, 6), (11, 6), (22, 5)):
                bits = (packed >> shift) & ((1 << (mantissa_bits + 5)) - 1)
                exponent = bits >> mantissa_bits
                mantissa = (bits & ((1 << mantissa_bits) - 1)).astype(np.float32)
                normal = np.ldexp(1 + mantissa / (1 << mantissa_bits), exponent.astype(np.int32) - 15)
                subnormal = np.ldexp(mantissa, -14 - mantissa_bits)
                value = np.where(exponent == 0, subnormal, normal)
                value = np.where(exponent == 31, np.where(mantissa == 0, np.inf, np.nan), value)
                decoded.append(value)
            array = np.stack(decoded, -1).astype(np.float32)
    return torch.from_numpy(array.copy())


def load_frame(path):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != b"FSMDUMP1":
            raise ValueError(f"{path}: invalid FSMDUMP1 magic")
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise ValueError(f"{path}: truncated JSON length")
        length, = struct.unpack("<I", length_bytes)
        if not 0 < length <= 1024 * 1024:
            raise ValueError(f"{path}: invalid JSON length {length}")
        metadata = json.loads(stream.read(length).decode("utf-8"))
        raw = stream.read()
    w, h = metadata["renderSize"]
    dw, dh = metadata["displaySize"]
    planes = {}
    for plane in metadata["planes"]:
        name = plane["name"]
        if name in planes:
            raise ValueError(f"{path}: duplicate plane {name}")
        value = decode_plane(raw, plane)
        # Display-resolution vectors must retain their display extent until the
        # pack shader's centre-texel sampling has been applied.
        if name in ("color", "depth", "motionVectors"):
            cw, ch = (dw, dh) if name == "motionVectors" and metadata["context_flags"] & 2 else (w, h)
        elif name == "output":
            cw, ch = dw, dh
        else:
            cw, ch = value.shape[1], value.shape[0]
        if value.shape[0] < ch or value.shape[1] < cw:
            raise ValueError(f"{path}: {name} smaller than declared active region")
        planes[name] = value[:ch, :cw].contiguous()
    if not {"color", "depth", "motionVectors", "output"} <= planes.keys():
        raise ValueError(f"{path}: missing required plane")
    return {"metadata": metadata, "planes": planes, "path": path}


class DumpSequence:
    """Load one frame at a time so a full 60 GB run need not fit in RAM."""
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_frame(self.paths[index])


def load_dump(directory, frame_slice=slice(None)):
    paths = sorted(Path(directory).glob("frame_*.bin"), key=lambda path: int(path.stem[6:]))[frame_slice]
    if not paths:
        raise ValueError(f"no frame_*.bin files selected under {directory}")
    return DumpSequence(paths)


@dataclass(frozen=True)
class Configuration:
    jitter: str
    motion: str
    gain: str

    def __str__(self):
        return f"jitter={self.jitter} mv={self.motion} gain={self.gain}"


def signs(flip, device="cpu"):
    return torch.tensor([-1 if axis in flip else 1 for axis in "xy"], device=device)


def log_average(color):
    # Log-average linear luminance; callers remove engine pre-exposure.
    luminance = (color.clamp(min=0) * color.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    return torch.exp(torch.log(luminance.clamp(min=1e-6)).mean())


def tonemap(color, exposure):
    x = color.clamp(min=0) * exposure
    return x / (1 + x)


def convert(frame, config, previous_jitter=None, device="cpu"):
    meta, planes = frame["metadata"], frame["planes"]
    w, h = meta["renderSize"]
    dw, dh = meta["displaySize"]
    if (dw, dh) != (2 * w, 2 * h):
        raise ValueError("DX12 learned path requires exactly 2x display/render size")
    color = planes["color"][..., :3].to(device)
    output = planes["output"][..., :3].to(device)
    if color.shape[-1] != 3 or output.shape[-1] != 3 or planes["motionVectors"].shape[-1] < 2:
        raise ValueError("colour/output need RGB and motionVectors need two channels")
    pre = float(meta["preExposure"]) or 1.0
    if not math.isfinite(pre) or pre <= 0:
        raise ValueError("invalid preExposure")
    gain = float(1 / (9.6 * log_average(color / pre))) if config.gain == "auto" else float(config.gain)
    exposure = float(planes["exposure"][0, 0, 0]) if "exposure" in planes else 1.0
    exposure = exposure or 1.0
    scale = exposure * gain / pre
    depth = planes["depth"][..., 0].to(device)
    if not meta["context_flags"] & 8:
        depth = 1 - depth
    mv = planes["motionVectors"][..., :2].to(device)
    target_size = (dw, dh) if meta["context_flags"] & 2 else (w, h)
    mv = mv * torch.tensor(meta["motionVectorScale"], device=device) / mv.new_tensor(target_size)
    mv = mv * signs(config.motion, device)
    previous_jitter = meta.get("previousJitterOffset", previous_jitter)
    if meta["context_flags"] & 4 and previous_jitter is not None:
        # Runtime flips affect the model jitter; cancellation uses raw dispatch jitter.
        cancellation = (mv.new_tensor(previous_jitter) - mv.new_tensor(meta["jitterOffset"])) / mv.new_tensor(target_size)
        mv = mv - cancellation
    if meta["context_flags"] & 2:
        mv = mv[1:dh:2, 1:dw:2].contiguous()
    lr = tonemap(color, scale).half().float()
    depth = depth.half().float()
    reference = tonemap(output, scale)
    jitter = torch.tensor(meta["jitterOffset"], device=device) * signs(config.jitter, device)
    for name, value in (("lr", lr), ("depth", depth), ("mv", mv), ("reference", reference), ("jitter", jitter)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{frame.get('path', 'frame')}: nonfinite {name}")
    return dict(lr=lr, mv=mv, depth=depth, gt=reference, fsr_out=reference, jitter=jitter)


def continuous(previous, current):
    if previous is None or current["reset"]:
        return False
    if previous["renderSize"] != current["renderSize"] or previous["displaySize"] != current["displaySize"] or previous["context_flags"] != current["context_flags"]:
        return False
    if previous.get("context_id") != current.get("context_id"):
        return False
    if "dispatch_index" in previous and "dispatch_index" in current:
        return current["dispatch_index"] == previous["dispatch_index"] + 1
    return True


def motion_for_metric(model, converted):
    mv = converted["mv"]
    if model.mv_dilate:
        from fsrmamba.fast import _nearest_depth
        _, field = _nearest_depth(converted["depth"][None, None], mv[None])
        mv = field[0]
    return F.interpolate(mv.permute(2, 0, 1)[None], size=model.output_size,
                         mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


@torch.inference_mode()
def evaluate(model, frames, config, warmup=8, cfg=None, callback=None):
    state = None
    previous_meta = previous_output = None
    quality, structure, temporal = [], [], []
    device = next(model.parameters()).device
    for index, frame in enumerate(frames):
        meta = frame["metadata"]
        h, w = reversed(meta["renderSize"])
        dh, dw = reversed(meta["displaySize"])
        if model.render_size != (h, w) or model.output_size != (dh, dw):
            if cfg is None:
                raise ValueError("changing resolution requires checkpoint configuration")
            resized = build_model(cfg, (h, w), (dh, dw), device)
            resized.load_state_dict(model.state_dict())
            model = resized
        follows = continuous(previous_meta, meta)
        converted = convert(frame, config, previous_meta["jitterOffset"] if previous_meta else None, device)
        if state is None or not follows:
            state = model.init_state()
        prediction, state = model(state, converted["lr"], converted["mv"], converted["depth"], converted["jitter"])
        prediction = prediction.clamp(0, 1 - 1 / 1024)
        if index >= warmup:
            quality.append(psnr(prediction, converted["gt"]))
            structure.append(ssim(prediction, converted["gt"]))
            if follows:
                temporal.append(temporal_instability(prediction, previous_output, motion_for_metric(model, converted)))
        if callback:
            callback(index, prediction, converted)
        previous_meta, previous_output = meta, prediction
    mean = lambda values: sum(values) / len(values) if values else math.nan
    return dict(config=config, psnr=mean(quality), ssim=mean(structure), temporal=mean(temporal),
                scored=len(quality), temporal_frames=len(temporal))


def sweep(model, frames, warmup=8, cfg=None, progress=False):
    if len(frames) <= warmup:
        raise ValueError(f"need more than {warmup} frames to score after warmup")
    results = []
    import os
    pick = lambda name, default: tuple(os.environ[name].split(",")) if os.environ.get(name) else default  # narrow the sweep
    grid = list(itertools.product(pick("FSRM_JITTERS", ("none", "x", "y", "xy")), pick("FSRM_MOTIONS", ("none", "x", "y", "xy")),
                                  pick("FSRM_GAINS", ("1", "0.3", "0.1", "0.03", "auto"))))
    for jitter, motion, gain in grid:
        config = Configuration(jitter, motion, gain)
        result = evaluate(model, frames, config, warmup, cfg)
        results.append(result)
        if progress:
            print(f"{len(results):2d}/{len(grid)} {config}: PSNR {result['psnr']:.4f}", flush=True)
    return sorted(results, key=lambda row: (row["psnr"], row["ssim"]), reverse=True)


def percentiles(value):
    return ", ".join(f"{x:.6g}" for x in torch.quantile(value.float().flatten(), torch.tensor([0., .01, .5, .95, .99, 1.])).tolist())


def print_statistics(frames):
    print("Input percentiles [min, p1, p50, p95, p99, max] (one row per frame):")
    previous = None
    for i, frame in enumerate(frames):
        meta, planes = frame["metadata"], frame["planes"]
        color = planes["color"][..., :3]
        luminance = (color.clamp(min=0) * torch.tensor([.2126, .7152, .0722])).sum(-1)
        depth = planes["depth"][..., 0]
        converted = convert(frame, Configuration("none", "none", "1"), previous)
        w, h = meta["renderSize"]
        magnitude = (converted["mv"] * torch.tensor([w, h])).norm(dim=-1)
        inverted = bool(meta["context_flags"] & 8)
        # Far-background concentration is a clue, not proof, for arbitrary views.
        low, high = float((depth < .1).float().mean()), float((depth > .9).float().mean())
        looks = "inverted" if low > high else "forward" if high > low else "ambiguous"
        print(f"{i}: colour RGB [{percentiles(color)}]; luminance log-average={float(log_average(color)):.6g} [{percentiles(luminance)}]")
        print(f"   depth [{percentiles(depth)}], looks {looks} (background heuristic), inverted flag={inverted}; motion pixels [{percentiles(magnitude)}]")
        print(f"   jitter={meta['jitterOffset']} exposure={float(planes['exposure'][0, 0, 0]) if 'exposure' in planes else 'null (1)'} preExposure={meta['preExposure']}")
        previous = meta["jitterOffset"]


def write_png(path, image):
    # Standard-library PNG writer; no Pillow/torchvision dependency.
    pixels = (image.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    h, w, _ = pixels.shape
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def save_best(model, frames, best, cfg, png_dir=None, cache_path=None):
    start = 8 if len(frames) >= 11 else 0
    chosen = {start, start + (len(frames) - 1 - start) // 2, len(frames) - 1}
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
    cache = []
    def save(index, prediction, converted):
        if png_dir and index in chosen:
            write_png(Path(png_dir) / f"frame_{index:05d}.png", torch.cat((prediction, converted["gt"]), dim=1))
        if cache_path:
            cache.append({key: value.cpu() for key, value in converted.items()})
    evaluate(model, frames, best, cfg=cfg, callback=save)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)


def parse_slice(value):
    try:
        a, b = value.split(":")
        start, stop = int(a) if a else None, int(b) if b else None
        if (start is not None and start < 0) or (stop is not None and stop < 0):
            raise ValueError
        return slice(start, stop)
    except ValueError:
        raise argparse.ArgumentTypeError("--frames must be A:B (zero-based, B exclusive)") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=parse_slice, default=slice(None))
    parser.add_argument("--save-png", type=Path)
    parser.add_argument("--export-cache", type=Path)
    args = parser.parse_args()
    try:
        frames = load_dump(args.dump_dir, args.frames)
        if len(frames) <= 8:
            raise ValueError("select at least 9 frames; first 8 are warmup")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        first = frames[0]["metadata"]
        model, cfg = load_checkpoint(args.ckpt, tuple(reversed(first["renderSize"])), tuple(reversed(first["displaySize"])), args.device)
        import os
        for name in filter(None, os.environ.get("FSRM_MODEL_FLAGS", "").split(",")):  # e.g. mv_dilate,depth_dilate (no weights involved)
            if not hasattr(model, name): raise ValueError(f"unknown model flag {name}")
            setattr(model, name, True); cfg[name] = True if isinstance(cfg, dict) else None
            print(f"model flag forced on: {name}")
        if not isinstance(model, FastAccumulator):
            raise ValueError("checkpoint must use arch=fast (FastAccumulator)")
        print(f"Checkpoint jitter_sign={model.jitter_sign}; sweep flips multiply the raw dispatch jitter before this model sign.")
        print("Dump gaps/reset/resolution changes restart history; temporal scores exclude these boundaries. Use dump_every=1 for recurrence diagnosis.")
        print_statistics(frames)
        results = sweep(model, frames, cfg=cfg, progress=True)
        print("\nSorted by PSNR, then SSIM; agreement with FSR 2 is not native-reference quality.")
        print(f"{'jitter':8} {'mv':8} {'gain':6} {'PSNR dB':>10} {'SSIM':>10} {'temporal MAE':>14} {'frames':>7} {'pairs':>7}")
        for row in results:
            config = row["config"]
            print(f"{config.jitter:8} {config.motion:8} {config.gain:6} {row['psnr']:10.4f} {row['ssim']:10.6f} {row['temporal']:14.6f} {row['scored']:7d} {row['temporal_frames']:7d}")
        print(f"Best: {results[0]['config']}")
        print("Exposure changes metric contrast; inspect the PNGs and compare within each gain as well as the overall ranking.")
        if args.save_png or args.export_cache:
            save_best(model, frames, results[0]["config"], cfg, args.save_png, args.export_cache)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
