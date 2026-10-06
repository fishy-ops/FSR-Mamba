"""Plain-assert tests for jittered gameplay dump conversion and temporal targets."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hires_to_cache as hires
import numpy as np
import torch
from fsrmamba.engine_data import load_engine_scene


def halton(index, base):
    value, fraction = 0., 1.
    while index:
        fraction /= base
        index, digit = divmod(index, base)
        value += digit * fraction
    return value


def scene(x, y):
    return torch.stack((1 + .35 * torch.sin(x * .91) * torch.cos(y * .63),
                        .8 + .25 * torch.cos(x * .77 + y * .49),
                        .6 + .2 * torch.sin(y * .88)), -1)


def grid(h=96, w=128):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return x.float() + .5, y.float() + .5


def write_frame(path, index, color, jitter, motion=None, depth=None, flags=8, exposure=.75,
                pre=2., reset=False, dispatch=None, previous_jitter=None, output=None, **extra):
    h, w = color.shape[:2]
    if motion is None:
        motion = torch.zeros(h, w, 2)
    if depth is None:
        depth = torch.full((h, w), .5)
    if output is None:
        output = color.repeat_interleave(2, 0).repeat_interleave(2, 1)
    def rgba(rgb):
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), -1).numpy().astype("<f4")
    sources = [("color", 2, rgba(color * pre)), ("depth", 39, depth[..., None].numpy().astype("<f4")),
               ("motionVectors", 16, motion.numpy().astype("<f4")), ("output", 2, rgba(output * pre)),
               ("exposure", 54, np.array([[[exposure]]], dtype="<f2")),
               ("reactive", 61, np.zeros((h, w, 1), dtype=np.uint8))]
    payload, planes = bytearray(), []
    for name, fmt, array in sources:
        ph, pw, channels = array.shape
        row_bytes = pw * channels * array.dtype.itemsize
        pitch = ((row_bytes + 255) // 256) * 256
        planes.append(dict(name=name, dxgi_format=fmt, footprint_format=fmt, width=pw, height=ph,
                           row_pitch=pitch, offset=len(payload), bytes=pitch * ph))
        for row in array:
            payload.extend(row.tobytes())
            payload.extend(bytes(pitch - row_bytes))
    meta = dict(renderSize=[w, h], displaySize=[2*w, 2*h], jitterOffset=jitter,
                previousJitterOffset=previous_jitter if previous_jitter is not None else jitter,
                motionVectorScale=[w, h], preExposure=pre, reset=reset,
                context_flags=flags, dispatch_index=index if dispatch is None else dispatch,
                context_id=1, dump_every=1, planes=planes)
    meta.update(extra)
    encoded = json.dumps(meta).encode("utf-8")
    path.write_bytes(b"FSMDUMP1" + struct.pack("<I", len(encoded)) + encoded + payload)


def make_sequence(directory, count=24, velocity=(0., 0.)):
    directory.mkdir()
    x, y = grid()
    for index in range(count):
        raw = [halton(index % 32 + 1, 2) - .5, halton(index % 32 + 1, 3) - .5]
        # Raw FSR dispatch jitter is negated to get physical sample displacement.
        color = scene(x - raw[0] - index * velocity[0], y - raw[1] - index * velocity[1])
        motion = torch.tensor([-velocity[0] / x.shape[1], -velocity[1] / x.shape[0]]).expand(*x.shape, 2)
        write_frame(directory / f"frame_{index:05d}.bin", index, color, raw, motion=motion,
                    reset=index == 0)


def run_converter(directory, output, *arguments):
    with contextlib.redirect_stdout(io.StringIO()) as progress:
        hires.main([str(directory), str(output), *arguments])
    text = progress.getvalue()
    assert "Mean mask coverage:" in text and "4x4 jitter bins" in text
    return torch.load(output, weights_only=True, mmap=True), text


def test_subsamples(directory):
    moving = directory / "moving"
    make_sequence(moving, 12, (.75, -.5))
    cache, progress = run_converter(moving, directory / "moving.pt")
    assert len(cache) == 12 and progress.count("phase=") == 12
    x, y = grid(48, 64)
    phase_sequence = hires.PhaseSequence()
    errors = []
    for index, record in enumerate(cache):
        frame = hires.load_frame(moving / f"frame_{index:05d}.bin")
        current = hires.prepare(frame)
        phase = phase_sequence.choose(current["jitter"])
        a, b = phase
        derived = hires.synthetic_jitter(current["jitter"], phase)
        assert torch.allclose(record["jitter"].float(), derived, atol=2e-4)
        analytic = scene(2 * (x + derived[0]) - index * .75, 2 * (y + derived[1]) + index * .5)
        expected = hires.dump.tonemap(analytic, current["scale"])
        error = float((record["lr"].float() - expected).abs().max())
        errors.append(error)
        assert error < 1e-3, error
        assert torch.equal(record["depth"], current["depth"][b::2, a::2].half())
        assert torch.allclose(record["mv"].float(), torch.tensor([-.75/128, .5/96]), atol=2e-6)
        assert set(record) == {"lr", "mv", "depth", "gt", "fsr_out", "jitter", "mask"}
        assert all(t.dtype == (torch.uint8 if k == "mask" else torch.float16) for k, t in record.items())
        assert record["gt"].shape == (96, 128, 3) and record["lr"].shape == (48, 64, 3)
    for block in range(0, 12, 4):
        phases = []
        for index in range(block, block + 4):
            raw = hires.load_frame(moving / f"frame_{index:05d}.bin")["metadata"]["jitterOffset"]
            phases.append(tuple((2*cache[index]["jitter"].float() + torch.tensor(raw) + .5).round().int().tolist()))
        assert set(phases) == set(hires.PHASES)
    # Motion composition tracks a translating textured scene, including jitter.
    current = hires.prepare(hires.load_frame(moving / "frame_00011.bin"))
    x, y = grid()
    truth = hires.dump.tonemap(scene(x - 11*.75, y + 11*.5), current["scale"])
    history = [hires.prepare(hires.load_frame(moving / f"frame_{i:05d}.bin")) for i in range(3, 11)]
    correct, mask = hires.render_accum(current, history)
    wrong, _ = hires.render_accum(dict(current, mv=torch.zeros_like(current["mv"])),
                                  [dict(f, mv=torch.zeros_like(f["mv"])) for f in history])
    region = (slice(8, -8), slice(8, -8))
    assert float(mask[region].float().mean()) > .95
    assert float((correct[region] - truth[region]).square().mean()) < float((wrong[region] - truth[region]).square().mean())
    print(f"PASS: analytic moving subsamples, all four phases, chained motion (max LR error {max(errors):.6g})")


def test_conversion(directory):
    x, y = grid()
    depth = .1 + x / 1000 + y / 2000
    motion = torch.stack((x / 1280, -y / 960), -1)
    raw, previous = [.25, -.125], [-.25, .375]
    path = directory / "conversion.bin"
    # Display-resolution vectors: centre texel selection, then jitter cancellation.
    display = torch.full((192, 256, 2), 42.)
    display[1::2, 1::2] = motion
    write_frame(path, 1, scene(x, y), raw, motion=display, depth=1-depth,
                flags=2 | 4, previous_jitter=previous, motionVectorScale=[256, 192])
    frame = hires.load_frame(path)
    current = hires.prepare(frame)
    expected = motion - (torch.tensor(previous) - torch.tensor(raw)) / torch.tensor([256, 192])
    assert torch.allclose(current["mv"], expected)
    assert torch.allclose(hires.box2(current["mv"]), expected.reshape(48, 2, 64, 2, 2).mean((1, 3)), atol=1e-7)
    assert torch.equal(current["depth"], depth.half().float())
    assert torch.equal(current["jitter"], -torch.tensor(raw))
    scale = .75 / (9.6 * float(hires.dump.log_average(scene(x, y))))
    assert torch.equal(current["color"], hires.dump.tonemap(scene(x, y), scale).half().float())
    assert torch.allclose(current["fsr_out"], hires.box2(hires.dump.tonemap(frame["planes"]["output"][..., :3], scale / 2)))
    print("PASS: display MV centre sampling, UV pooling, cancellation, depth inversion, exposure")


def test_static(directory):
    static = directory / "static"
    make_sequence(static, 32)
    output = directory / "scene" / "_cache_tm1.pt"
    cache, _ = run_converter(static, output)
    x, y = grid()
    # Dense quadrature of the analytic image over each unjittered HR pixel.
    samples = 24
    offsets = (torch.arange(samples).float() + .5) / samples - .5
    linear = sum(scene(x + dx, y + dy) for dx in offsets for dy in offsets) / samples**2
    current = hires.prepare(hires.load_frame(static / "frame_00031.bin"))
    reference = hires.dump.tonemap(linear, current["scale"])
    region = (slice(4, -4), slice(4, -4))
    error = float((cache[-1]["gt"].float()[region] - reference[region]).square().mean())
    single_errors = []
    for index in range(32):
        frame = hires.prepare(hires.load_frame(static / f"frame_{index:05d}.bin"))
        single = hires.dump.tonemap(frame["linear"], current["scale"])
        single_errors.append(float((single[region] - reference[region]).square().mean()))
    assert error < min(single_errors), (error, min(single_errors))
    assert cache[0]["mask"].sum() == 0
    assert cache[2]["mask"].sum() == 0
    assert float(cache[-1]["mask"][region].float().mean()) > .99
    loaded = load_engine_scene(str(output.parent), half=True)
    assert len(loaded) == 32 and torch.equal(loaded[-1]["gt"], cache[-1]["gt"])
    assert torch.equal(loaded[-1]["mask"], cache[-1]["mask"])
    baseline, _ = run_converter(static, directory / "baseline.pt", "--gt", "fsr-down", "--frames", "4:7")
    assert len(baseline) == 3
    assert all(torch.equal(f["gt"], f["fsr_out"]) and bool(f["mask"].all()) for f in baseline)
    print(f"PASS: static box reference MSE {error:.6g} < best single {min(single_errors):.6g}; cache loader and FSR baseline")


def test_rejection(directory):
    captures = directory / "rejection"
    captures.mkdir()
    color = torch.ones(96, 128, 3)
    for index in range(11):
        depth = torch.full((96, 128), .5)
        changed = color.clone()
        if index >= 8:
            depth[20:40, 20:40] = .2  # newly uncovered background, same colour
            changed[50:70, 50:70] = 4  # colour-only disagreement
        write_frame(captures / f"frame_{index:05d}.bin", index, changed, [0., 0.], depth=depth,
                    reset=index in (0, 9), dispatch=index if index < 10 else 12)
    cache, progress = run_converter(captures, directory / "rejection.pt")
    assert bool(cache[8]["mask"][:16, :16].all())
    assert not bool(cache[8]["mask"][22:38, 22:38].any())
    assert not bool(cache[8]["mask"][52:68, 52:68].any())
    assert not bool(cache[9]["mask"].any()) and not bool(cache[10]["mask"].any())
    assert progress.count("history reset") == 3
    # Offscreen history must not inherit the grid sampler's border padding.
    current = hires.prepare(hires.load_frame(captures / "frame_00007.bin"))
    current["mv"] = torch.ones_like(current["mv"])
    _, mask = hires.render_accum(current, [current] * 8)
    assert not bool(mask.any())
    # Historical radiance is normalized using the target frame's exposure.
    current["mv"].zero_()
    history = [dict(current, color=current["color"] * .01, scale=current["scale"] * .01)] * 8
    target, mask = hires.render_accum(current, history)
    assert bool(mask.all()) and torch.allclose(target, current["color"], atol=3e-5)
    path = captures / "frame_00010.bin"
    write_frame(path, 10, color, [0., 0.], dump_every=2)
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            run_converter(captures, directory / "bad.pt", "--frames", "10:11")
            assert False, "skipped dispatches accepted"
        except SystemExit as exc:
            assert exc.code == 1
    assert not (directory / "bad.pt").exists()
    print("PASS: depth/colour disocclusion, warmup, resets/gaps, bounds, exposure normalization")


def test_streaming(directory):
    captures = directory / "streaming"
    captures.mkdir()
    color = torch.ones(8, 12, 3)
    for index in range(205):
        raw = [halton(index + 1, 2) - .5, halton(index + 1, 3) - .5]
        write_frame(captures / f"frame_{index:05d}.bin", index, color, raw, reset=index == 0)
    cache, progress = run_converter(captures, directory / "streaming.pt")
    assert len(cache) == 205 and progress.count("phase=") == 205
    assert cache[-1]["gt"].shape == (8, 12, 3)
    print("PASS: 205-frame streaming export")


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="hires_test_") as temporary:
        directory = Path(temporary)
        test_subsamples(directory)
        test_conversion(directory)
        test_static(directory)
        test_rejection(directory)
        test_streaming(directory)
    print("PASS: hires_to_cache")


if __name__ == "__main__":
    main()
