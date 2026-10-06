#!/usr/bin/env python3
"""Convert consecutive FSMDUMP1 gameplay renders into a 2x training cache."""
from __future__ import annotations

import argparse
from collections import deque
import errno
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_dump as dump
import torch
import torch.nn.functional as F


# The shipped RDR2 ini flips both raw dispatch axes. accumulate_gt uses +1
# for the resulting cache jitter (centre + jitter); model jitter_sign is separate.
CONFIG = dump.Configuration("xy", "none", "auto")
PHASES = ((0, 0), (1, 1), (1, 0), (0, 1))
HISTORY = 8


def load_frame(path):
    """Read only training planes; optional R8 reactive masks need no decoder."""
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"FSMDUMP1":
            raise ValueError(f"{path}: invalid FSMDUMP1 magic")
        encoded = stream.read(4)
        if len(encoded) != 4:
            raise ValueError(f"{path}: truncated JSON length")
        length, = struct.unpack("<I", encoded)
        if not 0 < length <= 1024 * 1024:
            raise ValueError(f"{path}: invalid JSON length")
        metadata = json.loads(stream.read(length).decode("utf-8"))
        start = stream.tell()
        stream.seek(0, 2)
        size = stream.tell() - start
        w, h = metadata["renderSize"]
        dw, dh = metadata["displaySize"]
        if min(w, h) <= 0 or w % 2 or h % 2 or (dw, dh) != (2 * w, 2 * h):
            raise ValueError("need even render dimensions and exactly 2x display/render size")
        planes, seen = {}, set()
        for plane in metadata["planes"]:
            name = plane["name"]
            if name in seen:
                raise ValueError(f"{path}: duplicate plane {name}")
            seen.add(name)
            offset, count = int(plane["offset"]), int(plane["bytes"])
            if offset < 0 or count < 0 or offset + count > size:
                raise ValueError(f"{path}: invalid payload bounds for {name}")
            if name not in ("color", "depth", "motionVectors", "output", "exposure"):
                continue
            stream.seek(start + offset)
            value = dump.decode_plane(stream.read(count), dict(plane, offset=0))
            cw, ch = value.shape[1], value.shape[0]
            if name in ("color", "depth", "motionVectors"):
                cw, ch = (dw, dh) if name == "motionVectors" and metadata["context_flags"] & 2 else (w, h)
            elif name == "output":
                cw, ch = dw, dh
            if value.shape[0] < ch or value.shape[1] < cw:
                raise ValueError(f"{path}: {name} smaller than active region")
            planes[name] = value[:ch, :cw].contiguous()
    if not {"color", "depth", "motionVectors", "output"} <= planes.keys():
        raise ValueError(f"{path}: missing required plane")
    return dict(metadata=metadata, planes=planes, path=Path(path))


def synthetic_jitter(jitter, phase):
    # (2*x+a+0.5+j_hr)/2 = x+0.5 + (a-0.5+j_hr)/2.
    return (jitter + jitter.new_tensor(phase) - .5) / 2


class PhaseSequence:
    """Use all four phases per block, balancing the combined jitter's 4x4 bins."""
    def __init__(self):
        self.remaining = []
        self.bins = torch.zeros(4, 4, dtype=torch.int64)

    def choose(self, jitter):
        if not self.remaining:
            self.remaining = list(PHASES)
        def bin_for(phase):
            xy = ((synthetic_jitter(jitter, phase) + .5) * 4).floor().long().clamp(0, 3)
            return int(xy[1]), int(xy[0])
        phase = min(self.remaining, key=lambda p: int(self.bins[bin_for(p)]))
        self.remaining.remove(phase)
        self.bins[bin_for(phase)] += 1
        return phase


def box2(value):
    return F.avg_pool2d(value.permute(2, 0, 1)[None], 2, 2)[0].permute(1, 2, 0).contiguous()


def prepare(frame, previous_jitter=None):
    converted = dump.convert(frame, CONFIG, previous_jitter)
    meta, planes = frame["metadata"], frame["planes"]
    pre = float(meta["preExposure"]) or 1.0
    linear = planes["color"][..., :3] / pre
    exposure = float(planes["exposure"][0, 0, 0]) if "exposure" in planes else 1.0
    gain = float(1 / (9.6 * dump.log_average(linear)))
    scale = (exposure or 1.0) * gain
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid exposure scale")
    return dict(linear=linear, color=converted["lr"], depth=converted["depth"],
                mv=converted["mv"], jitter=converted["jitter"], scale=scale,
                fsr_out=box2(converted["fsr_out"]))


def sample(value, coordinates):
    h, w = value.shape[:2]
    grid = (coordinates + .5) * coordinates.new_tensor([2 / w, 2 / h]) - 1
    source = value[None, None] if value.ndim == 2 else value.permute(2, 0, 1)[None]
    result = F.grid_sample(source, grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0]
    return result[0] if value.ndim == 2 else result.permute(1, 2, 0)


def in_bounds(coordinates, h, w):
    return ((coordinates >= 0) & (coordinates <= coordinates.new_tensor([w - 1, h - 1]))).all(-1)


def render_accum(current, history):
    """Compose backward UV motion, sampling each source once on the unjittered grid."""
    h, w = current["depth"].shape
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    position = torch.stack((xs, ys), -1).float()
    coordinates = position - current["jitter"]
    color = sample(current["color"], coordinates)
    depth = sample(current["depth"], coordinates)
    chw = color.permute(2, 0, 1)[None]
    high = F.max_pool2d(chw, 3, 1, 1)[0].permute(1, 2, 0) + .05
    low = -F.max_pool2d(-chw, 3, 1, 1)[0].permute(1, 2, 0) - .05
    total = color.clone()
    count = torch.ones(h, w, dtype=torch.int32)
    valid = in_bounds(coordinates, h, w)
    previous = current
    for old in reversed(history):
        motion = sample(previous["mv"], coordinates)
        position = position + motion * motion.new_tensor([w, h])
        coordinates = position - old["jitter"]
        valid = valid & in_bounds(coordinates, h, w)
        old_depth = sample(old["depth"], coordinates)
        # Gate every link as well as agreement with the target, so occluded
        # paths cannot become valid again when they reach older matching depths.
        valid = valid & ((old_depth - depth).abs() <= .02 * depth.abs().clamp(min=1e-6))
        old_color = sample(dump.tonemap(old["linear"], current["scale"]), coordinates)
        survives = valid & ((old_color >= low) & (old_color <= high)).all(-1)
        total += old_color * survives[..., None]
        count += survives.int()
        previous = old
    return total / count[..., None], (count >= 4).to(torch.uint8)


def print_jitter_statistics(jitters, phases):
    values = torch.stack(jitters).float()
    bins = torch.zeros(4, 4, dtype=torch.int64)
    for x, y in ((values + .5) * 4).floor().long().clamp(0, 3).tolist():
        bins[y, x] += 1
    print(f"Jitter LR pixels: min={values.amin(0).tolist()} max={values.amax(0).tolist()} "
          f"mean={values.mean(0).tolist()} std={values.std(0, correction=0).tolist()}")
    print(f"Phase counts {[phases.count(p) for p in PHASES]} in order {PHASES}; "
          f"4x4 jitter bins (rows=y): {bins.tolist()}")


@torch.inference_mode()
def convert_dump(directory, output, gt="render-accum", frame_slice=slice(None)):
    if gt not in ("render-accum", "fsr-down"):
        raise ValueError(f"unknown target {gt}")
    paths = sorted(Path(directory).glob("frame_*.bin"), key=lambda p: int(p.stem[6:]))[frame_slice]
    if not paths:
        raise ValueError("no frame_*.bin files selected")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = deque(maxlen=HISTORY)
    sequence = PhaseSequence()
    previous_meta = None
    jitters, phases, coverage = [], [], []
    with tempfile.TemporaryDirectory(prefix="hires_cache_") as temporary:
        shards = []
        for index, path in enumerate(paths):
            frame = load_frame(path)
            meta = frame["metadata"]
            follows = dump.continuous(previous_meta, meta)
            if not follows:
                history.clear()
            if int(meta.get("dump_every", 1)) != 1:
                raise ValueError("use dump_every=1; skipped dispatches cannot be reprojected")
            current = prepare(frame, previous_meta["jitterOffset"] if follows else None)
            del frame
            phase = sequence.choose(current["jitter"])
            a, b = phase
            jitter = synthetic_jitter(current["jitter"], phase)
            if gt == "render-accum":
                target, mask = render_accum(current, history)
            else:
                target = current["fsr_out"]
                mask = torch.ones(target.shape[:2], dtype=torch.uint8)
            record = dict(lr=current["color"][b::2, a::2].contiguous(),
                          depth=current["depth"][b::2, a::2].contiguous(), mv=box2(current["mv"]),
                          gt=target, fsr_out=current["fsr_out"], jitter=jitter, mask=mask)
            record = {key: value if key == "mask" else value.half().contiguous() for key, value in record.items()}
            shard = Path(temporary) / f"{index:06d}.pt"
            torch.save(record, shard)
            shards.append(shard)
            coverage.append(float(mask.float().mean()))
            jitters.append(jitter)
            phases.append(phase)
            print(f"{index + 1}/{len(paths)} {path.name}: phase={phase} jitter={jitter.tolist()} "
                  f"mask={coverage[-1]:.2%}{' history reset' if not follows else ''}", flush=True)
            if gt == "render-accum":
                history.append({k: current[k] for k in ("linear", "depth", "mv", "jitter")})
            previous_meta = meta
            del current, record, target, mask
        history.clear()
        # File-backed storages let torch.save stream the final ordinary list cache
        # without retaining all decoded frames in anonymous RAM.
        cache = [torch.load(path, weights_only=True, mmap=True) for path in shards]
        staging = Path(temporary) / "cache.pt"
        torch.save(cache, staging)
        del cache
        try:
            staging.replace(output)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # Cross-volume exports need one additional temporary output copy.
            pending = output.with_name(output.name + ".tmp")
            try:
                shutil.copyfile(staging, pending)
                pending.replace(output)
            finally:
                pending.unlink(missing_ok=True)
    print_jitter_statistics(jitters, phases)
    print(f"Mean mask coverage: {sum(coverage) / len(coverage):.2%}; wrote {output} ({len(paths)} frames, gt={gt})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--gt", choices=("render-accum", "fsr-down"), default="render-accum")
    parser.add_argument("--frames", type=dump.parse_slice, default=slice(None))
    args = parser.parse_args(argv)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    try:
        convert_dump(args.dump_dir, args.out, args.gt, args.frames)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"hires_to_cache: {exc}\n")


if __name__ == "__main__":
    main()
