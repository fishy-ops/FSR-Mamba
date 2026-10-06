"""Plain-assert FSMDUMP1 loader, conversion, and CPU sweep regression tests."""
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_dump as dump
import numpy as np
import torch
from fsrmamba.config import load_checkpoint, save_sidecar
from fsrmamba.fast import FastAccumulator


def write_frame(path, index, color, depth, motion, output, exposure=None, flags=8):
    h, w = depth.shape
    payload = bytearray()
    planes = []
    sources = [("color", 10, color.astype("<f2")), ("depth", 39, depth.astype("<f4")[..., None]),
               ("motionVectors", 34, motion.astype("<f2")), ("output", 2, output.astype("<f4"))]
    if exposure is not None:
        sources.append(("exposure", 54, np.array([[[exposure]]], dtype="<f2")))
    for name, fmt, data in sources:
        ph, pw, channels = data.shape
        row_bytes = pw * channels * data.dtype.itemsize
        pitch = (row_bytes // 256 + 1) * 256
        planes.append(dict(name=name, dxgi_format=fmt, width=pw, height=ph, row_pitch=pitch,
                           offset=len(payload), bytes=pitch * ph))
        for row in data:
            payload.extend(row.tobytes())
            payload.extend(b'\xa5' * (pitch - row_bytes))
    meta = dict(renderSize=[w - 1, h - 1], displaySize=[2 * (w - 1), 2 * (h - 1)],
                jitterOffset=[index / 100, -index / 200], motionVectorScale=[w - 1, h - 1],
                preExposure=2, frameTimeDelta=16.7, reset=index == 0, cameraNear=.1,
                cameraFar=1000, cameraFovAngleVertical=1.2, enableSharpening=False, sharpness=0,
                context_flags=flags, dispatch_index=index + 1, dump_every=1, planes=planes)
    metadata = json.dumps(meta).encode("utf-8")
    path.write_bytes(b"FSMDUMP1" + struct.pack("<I", len(metadata)) + metadata + payload)
    return meta


def packed_plane(fmt, packed, bpp=4):
    raw = packed + b'padding!'
    plane = dict(name="test", dxgi_format=fmt, width=1, height=1, row_pitch=bpp + 8,
                 offset=0, bytes=len(raw))
    return dump.decode_plane(raw, plane)


def test_formats():
    formats = {10: ("<f2", 4), 2: ("<f4", 4), 34: ("<f2", 2), 16: ("<f4", 2),
               41: ("<f4", 1), 39: ("<f4", 1), 40: ("<f4", 1), 54: ("<f2", 1)}
    for fmt, (dtype, channels) in formats.items():
        values = np.arange(1, channels + 1, dtype=dtype) / 4
        actual = packed_plane(fmt, values.tobytes(), values.nbytes)
        assert torch.equal(actual, torch.from_numpy(values.astype(np.float32).reshape(1, 1, channels)))
    for fmt in (19, 20, 21):
        assert packed_plane(fmt, struct.pack("<fI", .375, 0xdeadbeef), 8).item() == .375
    for resource_format in (19, 20, 21):
        planar = dict(name="depth", dxgi_format=resource_format, footprint_format=39,
                      width=2, height=1, row_pitch=256, offset=0, bytes=256)
        actual = dump.decode_plane(struct.pack("<ff", .375, .625) + bytes(248), planar)
        assert torch.equal(actual, torch.tensor([[[.375], [.625]]]))
    assert torch.equal(packed_plane(24, struct.pack("<I", 1023 | (1023 << 20) | (3 << 30))), torch.tensor([[[1., 0., 1., 1.]]]))
    # R=1, G=2, B=.5, then the smallest subnormals and nonfinite encodings.
    packed = (15 << 6) | ((16 << 6) << 11) | ((14 << 5) << 22)
    assert torch.equal(packed_plane(26, struct.pack("<I", packed)), torch.tensor([[[1., 2., .5]]]))
    subnormal = packed_plane(26, struct.pack("<I", 1 | (1 << 11) | (1 << 22)))
    assert torch.equal(subnormal, torch.tensor([[[2. ** -20, 2. ** -20, 2. ** -19]]]))
    special = packed_plane(26, struct.pack("<I", (31 << 6) | (((31 << 6) | 1) << 11)))
    assert torch.isinf(special[0, 0, 0]) and torch.isnan(special[0, 0, 1])
    try:
        packed_plane(28, bytes(4))
        assert False, "unsupported format accepted"
    except ValueError as exc:
        assert "unsupported DXGI format 28" in str(exc)
    try:
        dump.decode_plane(bytes(4), dict(name="bad", dxgi_format=41, width=2, height=1, row_pitch=4, offset=0, bytes=4))
        assert False, "invalid pitch accepted"
    except ValueError:
        pass


def test_sequence(directory):
    rng = np.random.default_rng(71)
    color = rng.random((9, 9, 4), dtype=np.float32) * 4
    depth = rng.random((9, 9), dtype=np.float32)
    motion = np.zeros((9, 9, 2), np.float32)
    output = rng.random((18, 18, 4), dtype=np.float32) * 4
    for i in range(11):
        write_frame(directory / f"frame_{i:05d}.bin", i, color, depth, motion, output, exposure=.5 if i else None)
    frames = dump.load_dump(directory)
    frame = frames[1]
    assert len(frames) == 11
    assert torch.equal(frame["planes"]["color"], torch.from_numpy(color[:8, :8].astype(np.float16).astype(np.float32)))
    assert torch.equal(frame["planes"]["depth"][..., 0], torch.from_numpy(depth[:8, :8]))
    assert torch.equal(frame["planes"]["motionVectors"], torch.from_numpy(motion[:8, :8]))
    assert torch.equal(frame["planes"]["output"], torch.from_numpy(output[:16, :16]))
    assert frame["planes"]["exposure"].item() == .5
    assert "exposure" not in frames[0]["planes"]
    assert len(dump.load_dump(directory, slice(2, 5))) == 3
    converted = dump.convert(frame, dump.Configuration("x", "y", "0.3"))
    expected = dump.tonemap(frame["planes"]["color"][..., :3], .5 * .3 / 2).half().float()
    assert torch.equal(converted["lr"], expected)
    assert torch.equal(converted["depth"], torch.from_numpy(depth[:8, :8]).half().float())
    assert torch.equal(converted["jitter"], torch.tensor([-.01, -.005]))
    auto = dump.convert(frame, dump.Configuration("none", "none", "auto"))
    gain = 1 / (9.6 * float(dump.log_average(frame["planes"]["color"][..., :3] / 2)))
    assert torch.equal(auto["lr"], dump.tonemap(frame["planes"]["color"][..., :3], .5 * gain / 2).half().float())
    # Display centre sampling, inverted-depth conversion, and unflipped raw-jitter cancellation.
    display_motion = np.zeros((18, 18, 2), np.float32)
    display_motion[1::2, 1::2] = [.25, -.125]
    write_frame(directory / "display.bin", 1, color, depth, display_motion, output, exposure=0, flags=2 | 4)
    special = dump.load_frame(directory / "display.bin")
    special["metadata"]["motionVectorScale"] = [16, 16]
    special["metadata"]["previousJitterOffset"] = [.03, -.04]
    converted = dump.convert(special, dump.Configuration("xy", "x", "1"), [100, 100])
    expected_mv = torch.tensor([-.25, -.125]) - (torch.tensor([.03, -.04]) - torch.tensor([.01, -.005])) / 16
    assert torch.allclose(converted["mv"], expected_mv.expand(8, 8, 2))
    assert torch.equal(converted["depth"], (1 - torch.from_numpy(depth[:8, :8])).half().float())
    assert torch.equal(converted["lr"], dump.tonemap(special["planes"]["color"][..., :3], .5).half().float())
    previous, current = frames[1]["metadata"], frames[2]["metadata"]
    assert dump.continuous(previous, current)
    current["context_id"] = 2
    assert not dump.continuous(previous, current)
    current.pop("context_id")
    current["dispatch_index"] += 1
    assert not dump.continuous(previous, current)
    model = FastAccumulator((8, 8), (16, 16), widths=(4,), depths=(0,), n_state=0,
                            accum=True, film=True, depth_test=True, nearest_sample=True,
                            conf_consistent=True, carry_raw=True, base_gate=True, mv_dilate=True)
    checkpoint = directory / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    save_sidecar(checkpoint, dict(arch="fast", widths=[4], depths=[0], n_state=0,
                                accum=True, depth_test=True, nearest_sample=True,
                                conf_consistent=True, carry_raw=True, base_gate=True, mv_dilate=True))
    loaded, cfg = load_checkpoint(checkpoint, (8, 8), (16, 16))
    results = dump.sweep(loaded, frames, cfg=cfg)
    assert len(results) == 80 and len({row["config"] for row in results}) == 80
    assert all(row["scored"] == 3 and row["temporal_frames"] == 3 for row in results)
    assert all(math.isfinite(row[key]) for row in results for key in ("psnr", "ssim", "temporal"))
    assert results[0]["psnr"] >= results[-1]["psnr"]
    again = dump.evaluate(loaded, frames, results[0]["config"], cfg=cfg)
    assert again["psnr"] == results[0]["psnr"]
    png_dir, cache = directory / "png", directory / "cache.pt"
    dump.save_best(loaded, frames, results[0]["config"], cfg, png_dir, cache)
    assert len(list(png_dir.glob("*.png"))) == 3
    for path in png_dir.glob("*.png"):
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", data[16:24]) == (32, 16)
        assert zlib.crc32(data[12:29]) == struct.unpack(">I", data[29:33])[0]
    exported = torch.load(cache, weights_only=True)
    assert len(exported) == 11
    assert set(exported[0]) == {"lr", "mv", "depth", "gt", "fsr_out", "jitter"}
    assert torch.equal(exported[1]["gt"], exported[1]["fsr_out"])
    with contextlib.redirect_stdout(io.StringIO()) as stats:
        dump.print_statistics(frames)
    assert "luminance log-average" in stats.getvalue() and "jitter=" in stats.getvalue()
    argv = sys.argv
    try:
        short_png = directory / "png_short"
        sys.argv = ["analyze_dump.py", str(directory), "--ckpt", str(checkpoint),
                    "--frames", "1:11", "--save-png", str(short_png)]
        with contextlib.redirect_stdout(io.StringIO()) as cli:
            dump.main()
        assert "Sorted by PSNR" in cli.getvalue() and "Best:" in cli.getvalue()
        assert len(list(short_png.glob("*.png"))) == 3
    finally:
        sys.argv = argv


def main():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    test_formats()
    scratch = Path(__file__).resolve().parents[2] / ".work"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dump_test_", dir=scratch) as directory:
        test_sequence(Path(directory))
    print("PASS: padded planes, DXGI formats, proxy conventions, 80-way CPU sweep, PNG and cache")


if __name__ == "__main__":
    main()
