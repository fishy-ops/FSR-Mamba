"""Plain-assert mask capture inspection tests; no GPU or checkpoint required."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_masks as masks
import numpy as np


def write_frame(path, with_masks):
    color = np.full((3, 4, 4), 2, dtype="<f2")
    color[:2, :3, :3] = np.arange(18).reshape(2, 3, 3) / 4
    reactive = np.array([[0, 1, 128, 255], [127, 255, 0, 255], [255, 255, 255, 255]], dtype="u1")[..., None]
    composition = np.array([[0, .5, 1, 1], [.25, .75, 0, 1], [1, 1, 1, 1]], dtype="<f2")[..., None]
    sources = [("color", 10, color)]
    if with_masks:
        sources += [("reactive", 61, reactive), ("transparencyAndComposition", 54, composition)]
    planes, raw = [], bytearray()
    for name, fmt, data in sources:
        h, w, c = data.shape
        pitch = 256
        planes.append(dict(name=name, dxgi_format=fmt, footprint_format=fmt,
                           width=w, height=h, row_pitch=pitch, offset=len(raw), bytes=pitch * h))
        for row in data:
            raw.extend(row.tobytes())
            raw.extend(b"\xa5" * (pitch - w * c * data.dtype.itemsize))
    # The inspector must not need to decode depth or future unrelated planes.
    planes.append(dict(name="unused", dxgi_format=999, width=1, height=1,
                       row_pitch=256, offset=len(raw), bytes=256))
    raw.extend(bytes(256))
    metadata = dict(renderSize=[3, 2], displaySize=[6, 4], planes=planes)
    if with_masks:
        metadata.update(enableAutoReactive=False, colorOpaqueOnly_present=True)
    encoded = json.dumps(metadata).encode()
    path.write_bytes(b"FSMDUMP1" + struct.pack("<I", len(encoded)) + encoded + raw)
    return metadata, raw, color[:2, :3].astype(np.float32), reactive[:2, :3].astype(np.float32) / 255, composition[:2, :3].astype(np.float32)


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, compressed = 8, bytearray()
    while at < len(data):
        length, = struct.unpack_from(">I", data, at)
        kind, payload = data[at + 4:at + 8], data[at + 8:at + 8 + length]
        crc, = struct.unpack_from(">I", data, at + 8 + length)
        assert crc == zlib.crc32(kind + payload)
        if kind == b"IHDR":
            w, h, bits, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            assert (bits, color_type, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            assert at + 12 == len(data)
        at += length + 12
    rows = np.frombuffer(zlib.decompress(compressed), np.uint8).reshape(h, 1 + w * 3)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(h, w, 3)


def test_capture(directory):
    absent = directory / "frame_00000.bin"
    present = directory / "frame_00001.bin"
    write_frame(absent, False)
    metadata, raw, color, reactive, composition = write_frame(present, True)
    assert set(masks.load_frame(absent)["planes"]) == {"color"}
    loaded = masks.load_frame(present)
    assert loaded["metadata"]["enableAutoReactive"] is False
    for name, expected in (("color", color), ("reactive", reactive), ("transparencyAndComposition", composition)):
        np.testing.assert_array_equal(loaded["planes"][name], expected)
    # Older dumps need no footprint_format key.
    plane = dict(metadata["planes"][1])
    del plane["footprint_format"]
    np.testing.assert_array_equal(masks.decode_plane(raw, plane)[:2, :3], reactive)
    stats = masks.mask_statistics(reactive)
    assert stats["finite"] == stats["pixels"] == 6
    assert stats["gt0"] == 4 / 6 and stats["gt05"] == 2 / 6
    np.testing.assert_allclose(stats["percentiles"], np.percentile(reactive, masks.PERCENTILES))
    stats = masks.mask_statistics(composition)
    assert stats["gt0"] == 4 / 6 and stats["gt05"] == 2 / 6
    assert stats["percentiles"][3] == .375
    assert masks.mask_statistics(np.zeros((2, 3, 1)))["gt0"] == 0
    invalid = masks.mask_statistics(np.array([0, 1, np.nan, np.inf]))
    assert invalid["finite"] == 2 and invalid["pixels"] == 4 and invalid["gt0"] == .5

    png = directory / "png"
    result = subprocess.run([sys.executable, str(Path(masks.__file__)), str(directory),
                             "--frames", "0:2", "--png", str(png)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "reactive: absent" in result.stdout and "transparencyAndComposition: absent" in result.stdout
    assert "enableAutoReactive=unknown" in result.stdout and "enableAutoReactive=False" in result.stdout
    assert ">0=0.666667 >0.5=0.333333" in result.stdout
    assert "p50=0.375" in result.stdout
    assert len(list(png.glob("*.png"))) == 2
    linear = color[..., :3].astype(np.float64)
    log_average = np.exp(np.log(np.maximum((linear * [.2126, .7152, .0722]).sum(-1), 1e-6)).mean())
    x = linear / (9.6 * log_average)
    expected_color = np.rint(x / (1 + x) * 255).astype(np.uint8)
    np.testing.assert_allclose(masks.color_preview(color * 8), masks.color_preview(color))
    for name, expected in zip(masks.MASKS, (reactive, composition)):
        image = read_png(png / f"frame_00001_{name}.png")
        assert image.shape == (2, 6, 3)
        np.testing.assert_array_equal(image[:, :3], expected_color)
        np.testing.assert_array_equal(image[:, 3:], np.repeat(np.rint(expected * 255).astype(np.uint8), 3, axis=2))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        masks.main([str(directory), "--frames", "1:"])
    assert "frame_00000" not in out.getvalue() and "frame_00001" in out.getvalue()

    for change in ({"row_pitch": 1}, {"offset": len(raw)}, {"bytes": 1}, {"dxgi_format": 999, "footprint_format": 999}):
        bad = dict(plane, **change)
        try:
            masks.decode_plane(raw, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid plane accepted: {change}")
    broken = directory / "broken.bin"
    for data in (b"wrong", b"FSMDUMP1\1", b"FSMDUMP1" + struct.pack("<I", 100) + b"{}", present.read_bytes()[:-300]):
        broken.write_bytes(data)
        try:
            masks.load_frame(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed capture accepted")


def test_formats():
    def decode(fmt, data):
        return masks.decode_plane(data, dict(name="color", dxgi_format=fmt, width=1, height=1,
                                            row_pitch=len(data), offset=0, bytes=len(data)))
    np.testing.assert_array_equal(decode(2, struct.pack("<ffff", 1, 2, 3, 4)), [[[1, 2, 3, 4]]])
    np.testing.assert_array_equal(decode(28, bytes([0, 255, 0, 255])), [[[0, 1, 0, 1]]])
    np.testing.assert_array_equal(decode(87, bytes([0, 0, 255, 255])), [[[1, 0, 0, 1]]])
    packed = (15 << 6) | ((16 << 6) << 11) | ((14 << 5) << 22)
    np.testing.assert_array_equal(decode(26, struct.pack("<I", packed)), [[[1, 2, .5]]])
    np.testing.assert_array_equal(decode(24, struct.pack("<I", 1023 | (3 << 30))), [[[1, 0, 0, 1]]])


if __name__ == "__main__":
    test_formats()
    with tempfile.TemporaryDirectory() as directory:
        test_capture(Path(directory))
    print("mask inspection tests passed")
