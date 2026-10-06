"""Inspect optional per-pixel masks in FSMDUMP1 captures (NumPy only).

Each file is magic, little-endian u32 JSON length, UTF-8 JSON, then raw planes.
Plane offsets are payload-relative; row_pitch includes D3D12 copy padding.
Only color and the two masks are decoded; unrelated planes can use any format.
"""
import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np


MASKS = ("reactive", "transparencyAndComposition")
PERCENTILES = (0, 1, 5, 50, 95, 99, 100)
# DXGI float and UNORM layouts, including common typeless resource aliases.
FORMATS = {2: ("<f4", 4, 1), 9: ("<f2", 4, 1), 10: ("<f2", 4, 1),
           16: ("<f4", 2, 1), 34: ("<f2", 2, 1), 39: ("<f4", 1, 1),
           41: ("<f4", 1, 1), 54: ("<f2", 1, 1),
           53: ("<u2", 1, 65535), 56: ("<u2", 1, 65535),
           60: ("u1", 1, 255), 61: ("u1", 1, 255),
           27: ("u1", 4, 255), 28: ("u1", 4, 255), 29: ("u1", 4, 255),
           87: ("u1", 4, 255), 90: ("u1", 4, 255), 91: ("u1", 4, 255)}


def decode_plane(raw, plane):
    fmt = int(plane.get("footprint_format", plane["dxgi_format"]))
    if fmt not in FORMATS and fmt not in (24, 26):
        raise ValueError(f"{plane['name']}: unsupported DXGI format {fmt}")
    dtype, channels, scale = FORMATS.get(fmt, ("<u4", 1, 1))
    bpp = np.dtype(dtype).itemsize * channels
    w, h, pitch, offset, size = (int(plane[k]) for k in
                                ("width", "height", "row_pitch", "offset", "bytes"))
    if min(w, h) <= 0 or pitch < w * bpp or size != pitch * h or offset < 0 or offset + size > len(raw):
        raise ValueError(f"{plane['name']}: invalid dimensions, pitch, or payload bounds")
    rows = np.frombuffer(raw, np.uint8, count=size, offset=offset).reshape(h, pitch)
    pixels = rows[:, :w * bpp].copy().view(dtype).reshape(h, w, channels)
    if fmt == 24:
        packed = pixels[..., 0]
        value = np.stack([(packed >> shift) & mask for shift, mask in
                          ((0, 1023), (10, 1023), (20, 1023), (30, 3))], -1)
        return value.astype(np.float32) / np.array([1023, 1023, 1023, 3], np.float32)
    if fmt == 26:
        packed = pixels[..., 0]
        values = []
        for shift, bits in ((0, 6), (11, 6), (22, 5)):
            channel = (packed >> shift) & ((1 << (bits + 5)) - 1)
            exponent = (channel >> bits).astype(np.int32)
            mantissa = (channel & ((1 << bits) - 1)).astype(np.float32)
            value = np.where(exponent == 0, np.ldexp(mantissa, -14 - bits),
                             np.ldexp(1 + mantissa / (1 << bits), exponent - 15))
            values.append(np.where(exponent == 31, np.where(mantissa == 0, np.inf, np.nan), value))
        return np.stack(values, -1)
    value = pixels.astype(np.float32) / scale
    if fmt in (87, 90, 91):
        value = value[..., [2, 1, 0, 3]]
    if fmt in (29, 91):
        rgb = value[..., :3]
        value[..., :3] = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return value


def load_frame(path):
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"FSMDUMP1":
            raise ValueError(f"{path}: invalid FSMDUMP1 magic")
        length = stream.read(4)
        if len(length) != 4:
            raise ValueError(f"{path}: truncated JSON length")
        length, = struct.unpack("<I", length)
        if not 0 < length <= 1024 * 1024:
            raise ValueError(f"{path}: invalid JSON length")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError(f"{path}: truncated JSON")
        metadata = json.loads(encoded)
        raw = stream.read()
    w, h = (int(v) for v in metadata["renderSize"])
    if min(w, h) <= 0:
        raise ValueError(f"{path}: invalid renderSize")
    planes, seen = {}, set()
    for plane in metadata["planes"]:
        name = plane["name"]
        if name in seen:
            raise ValueError(f"{path}: duplicate plane {name}")
        seen.add(name)
        if name not in ("color", *MASKS):
            continue
        value = decode_plane(raw, plane)
        if value.shape[0] < h or value.shape[1] < w:
            raise ValueError(f"{path}: {name} smaller than renderSize")
        if name in MASKS and value.shape[2] != 1:
            raise ValueError(f"{path}: {name} must have one channel")
        if name == "color" and value.shape[2] < 3:
            raise ValueError(f"{path}: color must have at least three channels")
        planes[name] = value[:h, :w].copy()
    if "color" not in planes:
        raise ValueError(f"{path}: missing color plane")
    return {"metadata": metadata, "planes": planes}


def mask_statistics(mask):
    finite = np.isfinite(mask)
    values = mask[finite]
    return {"finite": int(finite.sum()), "pixels": int(mask.size),
            "gt0": float(np.mean(values > 0)) if values.size else float("nan"),
            "gt05": float(np.mean(values > .5)) if values.size else float("nan"),
            "percentiles": np.percentile(values, PERCENTILES) if values.size else np.full(len(PERCENTILES), np.nan)}


def color_preview(color):
    rgb = np.maximum(color[..., :3].astype(np.float64), 0)
    if not np.isfinite(rgb).all():
        raise ValueError("color contains nonfinite values")
    luminance = rgb @ np.array([.2126, .7152, .0722])
    average = np.exp(np.log(np.maximum(luminance, 1e-6)).mean())
    x = rgb / (9.6 * average)
    return x / (1 + x)


def write_png(path, rgb):
    rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h, w, _ = rgb.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + row.tobytes() for row in rgb)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" +
                          chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                          chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def inspect_frame(path, png=None):
    frame = load_frame(path)
    meta, planes = frame["metadata"], frame["planes"]
    print(f"{Path(path).name}: enableAutoReactive={meta.get('enableAutoReactive', 'unknown')} "
          f"colorOpaqueOnly_present={meta.get('colorOpaqueOnly_present', 'unknown')}")
    preview = color_preview(planes["color"]) if png is not None else None
    for name in MASKS:
        if name not in planes:
            print(f"  {name}: absent")
            continue
        mask = planes[name]
        stats = mask_statistics(mask)
        percentiles = " ".join(f"p{p}={v:.6g}" for p, v in zip(PERCENTILES, stats["percentiles"]))
        print(f"  {name}: finite={stats['finite']}/{stats['pixels']} "
              f">0={stats['gt0']:.6f} >0.5={stats['gt05']:.6f} {percentiles}")
        if png is not None:
            gray = np.repeat(np.clip(mask, 0, 1), 3, axis=2)
            gray[~np.isfinite(mask[..., 0])] = [1, 0, 1]
            Path(png).mkdir(parents=True, exist_ok=True)
            write_png(Path(png) / f"{Path(path).stem}_{name}.png", np.concatenate([preview, gray], axis=1))


def frame_range(value):
    try:
        start, end = value.split(":")
        start, end = int(start) if start else 0, int(end) if end else None
        if start < 0 or (end is not None and end <= start):
            raise ValueError
        return slice(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("expected nonnegative A:B with B > A (B is exclusive)") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--frames", type=frame_range, default=slice(None), help="slice of sorted captures, A:B (B exclusive)")
    parser.add_argument("--png", type=Path, help="output directory for color-left/mask-right PNGs")
    args = parser.parse_args(argv)
    try:
        paths = sorted(args.dump_dir.glob("frame_*.bin"), key=lambda p: int(p.stem[6:]))[args.frames]
        if not paths:
            raise ValueError("no frame_*.bin files selected")
        for path in paths:
            inspect_frame(path, args.png)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"inspect_masks: {exc}\n")


if __name__ == "__main__":
    main()
