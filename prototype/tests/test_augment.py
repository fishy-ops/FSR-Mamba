"""Geometry, sampling, recurrent warm-up, and engine-training smoke tests."""

import itertools
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.augment import IDENTITY, augment_sequence, random_op
from fsrmamba.baseline import FSRAccumulator
from fsrmamba.cropbank import sample_from_window
from fsrmamba.engine_data import _upsample_to, crop_sequence, load_engine_scene
from fsrmamba.metrics import temporal_instability
from tools.make_fake_engine import make_fake_engine
from train import balance_core, loss_core, warm_sequence, warmup_length, warp_prev


OPS = list(itertools.product((False, True), repeat=3))
ROOT = Path(__file__).resolve().parents[1]


def spatial(x, op):
    transpose, flip_x, flip_y = op
    if transpose:
        x = x.transpose(0, 1)
    if flip_x:
        x = x.flip(1)
    if flip_y:
        x = x.flip(0)
    return x


def equal_frames(a, b):
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.keys() == fb.keys()
        for key in fa:
            if torch.is_tensor(fa[key]):
                assert torch.equal(fa[key], fb[key]), key
            else:
                assert fa[key] == fb[key], key


def legacy_crop(frames, size, rng):
    """The original uniform crop calculation, independent of new options."""
    rh, rw = frames[0]["lr"].shape[:2]
    y, x = rng.randint(0, rh - size), rng.randint(0, rw - size)
    result = []
    for f in frames:
        mv = f["mv"][y:y + size, x:x + size].float() * torch.tensor([rw / size, rh / size])
        result.append(dict(
            lr=f["lr"][y:y + size, x:x + size].float(),
            mv=_upsample_to(mv, size * 2, size * 2, "bilinear"),
            depth=_upsample_to(f["depth"][y:y + size, x:x + size].float(), size * 2, size * 2, "nearest"),
            gt=f["gt"][y * 2:(y + size) * 2, x * 2:(x + size) * 2].float(),
            fsr_out=f["fsr_out"][y * 2:(y + size) * 2, x * 2:(x + size) * 2].float(),
            jitter=f["jitter"]))
    return result


def geometry(raw):
    # A non-square subcrop of the 40x56 capture, including crop-local UV rescale.
    frames = []
    for f in raw:
        frames.append(dict(
            lr=f["lr"][4:36, 6:50].float(),
            gt=f["gt"][8:72, 12:100].float(),
            fsr_out=f["fsr_out"][8:72, 12:100].float(),
            depth=_upsample_to(f["depth"][4:36, 6:50].float(), 64, 88, "nearest"),
            mv=_upsample_to(f["mv"][4:36, 6:50].float() * torch.tensor([56 / 44, 40 / 32]),
                            64, 88, "bilinear"), jitter=f["jitter"]))
    snapshot = [{k: v.clone() if torch.is_tensor(v) else v for k, v in f.items()} for f in frames]
    assert any(abs(f["jitter"][0]) > 0 and abs(f["jitter"][1]) > 0 for f in frames)
    assert any(f["mv"].abs().max() > 0 for f in frames)
    for op in OPS:
        augmented = augment_sequence(frames, op)
        transpose, fx, fy = op
        inverse = (transpose, fy, fx) if transpose else op
        equal_frames(augment_sequence(augmented, inverse), frames)
        assert all(a is not b for a, b in zip(frames, augmented))
        # Each UV component's pixel displacement changes axis together with its divisor.
        h, w = augmented[1]["mv"].shape[:2]
        pixels = spatial(frames[1]["mv"] * torch.tensor([88, 64]), op)
        if transpose:
            pixels = pixels[..., [1, 0]]
        pixels = pixels * torch.tensor([-1 if fx else 1, -1 if fy else 1])
        torch.testing.assert_close(augmented[1]["mv"] * torch.tensor([w, h]), pixels)
        for i, f in enumerate(augmented):
            if i:
                original = temporal_instability(frames[i]["gt"], frames[i - 1]["gt"], frames[i]["mv"])
                transformed = temporal_instability(f["gt"], augmented[i - 1]["gt"], f["mv"])
                assert abs(original - transformed) <= 1e-5, (op, original, transformed)
                warped = warp_prev(frames[i - 1]["gt"], frames[i]["mv"])
                torch.testing.assert_close(warp_prev(augmented[i - 1]["gt"], f["mv"]),
                                           spatial(warped, op), atol=1e-5, rtol=0)
        tensor_frames = [dict(f, jitter=torch.tensor(f["jitter"], dtype=torch.float64)) for f in frames]
        transformed = augment_sequence(tensor_frames, op)
        assert transformed[0]["jitter"].dtype == torch.float64
        assert isinstance(augmented[0]["jitter"], tuple)
        equal_frames(augment_sequence(transformed, inverse), tensor_frames)
    equal_frames(snapshot, frames)
    assert IDENTITY == (False, False, False)
    rng = random.Random(51)
    counts = {op: 0 for op in OPS}
    for _ in range(8000):
        counts[random_op(rng)] += 1
    assert all(850 < n < 1150 for n in counts.values()), counts
    return frames


def jitter_geometry(frames):
    baseline = FSRAccumulator(frames[0]["lr"].shape[:2], frames[0]["gt"].shape[:2])
    resolves = [baseline._upsample(f["lr"], f["jitter"])[0] for f in frames]
    for op in OPS:
        augmented = augment_sequence(frames, op)
        acc = FSRAccumulator(augmented[0]["lr"].shape[:2], augmented[0]["gt"].shape[:2])
        for i, f in enumerate(augmented):
            resolved = acc._upsample(f["lr"], f["jitter"])[0]
            expected = spatial(resolves[i], op)[4:-4, 4:-4]
            # Not exact: the ported resolve picks its 3x3 window with a strict '>' (upsample.h:307),
            # so a mirrored footprint differs on ties. The check that matters is that the
            # transformed jitter is right, so compare against the wrong-signed jitter.
            err = (resolved[4:-4, 4:-4] - expected).abs().mean().item()
            jx, jy = float(f["jitter"][0]), float(f["jitter"][1])
            wrong = acc._upsample(f["lr"], (-jx, -jy))[0][4:-4, 4:-4]
            err_wrong = (wrong - expected).abs().mean().item()
            assert err < 1.5e-2, (op, i, err)
            if max(abs(jx), abs(jy)) > 0.15:
                assert err < 0.5 * err_wrong, (op, i, f["jitter"], err, err_wrong)
            print(f"  jitter op={op} frame={i} err={err:.5f} wrong-sign={err_wrong:.5f}")


def sampling(raw):
    for seed in range(10):
        before, after = random.Random(seed), random.Random(seed)
        expected = legacy_crop(raw, 32, before)
        actual = crop_sequence(raw, 32, rng=after, augment_rng=None, edge_prob=0)
        equal_frames(expected, actual)
        assert before.getstate() == after.getstate()
    a = crop_sequence(raw, 32, rng=random.Random(8), augment_rng=random.Random(2))
    b = crop_sequence(raw, 32, rng=random.Random(8), augment_rng=random.Random(2))
    equal_frames(a, b)
    expected = augment_sequence(legacy_crop(raw, 32, random.Random(8)), random_op(random.Random(2)))
    equal_frames(a, expected)
    entry = dict(ch=40, cw=56, full_rh=40, full_rw=56, x=0, y=0,
                 jitter=[f["jitter"] for f in raw],
                 **{key: torch.stack([f[key] for f in raw]) for key in ("lr", "mv", "depth", "gt")})
    rng = random.Random(9)
    rng.randint(0, 0)  # Bank's existing temporal-start draw.
    expected = legacy_crop(raw, 32, rng)
    for f in expected:
        del f["fsr_out"]
    equal_frames(sample_from_window(entry, 32, 2, random.Random(9)), expected)
    equal_frames(sample_from_window(entry, 32, 2, random.Random(9), augment_rng=random.Random(4)),
                 augment_sequence(expected, random_op(random.Random(4))))
    a = sample_from_window(entry, 32, 2, random.Random(6), frames=4, augment_rng=random.Random(7))
    b = sample_from_window(entry, 32, 2, random.Random(6), frames=4, augment_rng=random.Random(7))
    equal_frames(a, b)
    assert len(a) == 4

    # A single strong GT edge must attract centres; identify origins from LR coordinates.
    lr = torch.arange(40 * 56).reshape(40, 56, 1).expand(-1, -1, 3).float()
    gt = torch.zeros(80, 112, 3)
    gt[:, 64:] = 1
    f = dict(raw[0], lr=lr, gt=gt)
    origins = []
    for seed in range(20):
        c = crop_sequence([f], 16, rng=random.Random(seed), edge_prob=1)[0]
        origin = int(c["lr"][0, 0, 0]) % 56
        origins.append(origin)
        assert 23 <= origin <= 25, origin
    assert len(set(origins)) > 1
    a = crop_sequence(raw, 32, rng=random.Random(3), edge_prob=1)
    b = crop_sequence(raw, 32, rng=random.Random(3), edge_prob=1)
    equal_frames(a, b)
    blank = dict(f, gt=torch.zeros_like(gt))
    equal_frames(crop_sequence([blank], 16, rng=random.Random(4), edge_prob=1),
                 crop_sequence([blank], 16, rng=random.Random(4)))


def halo():
    image = torch.arange(20 * 30 * 3).reshape(20, 30, 3).float() / (20 * 30 * 3)
    assert loss_core(image, 0) is image
    assert loss_core(image, 4).shape == (12, 22, 3)
    assert loss_core(image[..., 0], 4).shape == (12, 22)
    for bad in (-1, 10):
        try:
            loss_core(image, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid halo accepted")
    mv = torch.zeros(20, 30, 2)
    mv[..., 0], mv[..., 1] = 5 / 30, -2 / 20
    full = warp_prev(image, mv)
    core = loss_core(full, 4)
    torch.testing.assert_close(core, full[4:16, 4:26], atol=0, rtol=0)
    # Reproject against the full image, including history outside the loss core.
    torch.testing.assert_close(core, image[2:14, torch.arange(9, 31).clamp(max=29)],
                               atol=1e-6, rtol=0)
    assert not torch.allclose(core, warp_prev(loss_core(image, 4), loss_core(mv, 4)))
    class Router:
        def __init__(self):
            self._last_weights = torch.zeros(1, 2, 20, 30, requires_grad=True)
            with torch.no_grad():
                self._last_weights[:, 0] = 0.75
                self._last_weights[:, 1] = 0.25

        def load_balance_loss(self):
            return self._last_weights.new_tensor(123.0)

    router = Router()
    assert balance_core(router, 0).item() == 123
    balance_core(router, 4).backward()
    grad = router._last_weights.grad[0].permute(1, 2, 0)
    assert grad[:4].count_nonzero() == grad[-4:].count_nonzero() == 0
    assert grad[:, :4].count_nonzero() == grad[:, -4:].count_nonzero() == 0
    assert loss_core(grad, 4).count_nonzero() > 0


def recurrent(raw):
    rng = random.Random(0)
    original_rng_state = rng.getstate()
    assert warmup_length(6, 0, 0.2, rng) == 0
    assert rng.getstate() == original_rng_state
    assert warmup_length(6, 2, 0, rng) == 2
    assert warmup_length(6, 2, 1, rng) == 0
    rng = random.Random(12)
    decisions = [warmup_length(6, 2, 0.2, rng) for _ in range(1000)]
    assert 150 < decisions.count(0) < 250
    for count in (-1, 6, 7):
        try:
            warmup_length(6, count, 0.2, rng)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid warm-up accepted")

    class Recurrence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def init_state(self, dev):
            return torch.zeros((), device=dev)

        def forward(self, state, lr, mv, depth, jitter, prev_depth_hr=None):
            self.calls.append((torch.is_grad_enabled(), prev_depth_hr))
            state = state + self.weight
            return lr * state, state

    seq = crop_sequence(raw, 32, rng=random.Random(0))
    student, teacher = Recurrence(), Recurrence()
    state, ts, prev, depth, gt = warm_sequence(student, teacher, seq, 2, "cpu")
    assert state.item() == ts.item() == 2
    assert not state.requires_grad and not ts.requires_grad and not prev.requires_grad
    assert depth is seq[1]["depth"] and gt is seq[1]["gt"]
    for model in (student, teacher):
        assert model.calls[0] == (False, None)
        assert model.calls[1][0] is False and model.calls[1][1] is seq[0]["depth"]
        assert model.weight.grad is None
    out, state = student(state, **{k: seq[2][k] for k in ("lr", "mv", "depth", "jitter")},
                         prev_depth_hr=depth)
    out.mean().backward()
    torch.testing.assert_close(student.weight.grad, seq[2]["lr"].mean())
    fresh = Recurrence()
    state, ts, prev, depth, gt = warm_sequence(fresh, None, seq, 0, "cpu")
    assert state.item() == 0 and ts is prev is depth is gt is None
    assert not fresh.calls


def smoke(data):
    command = [sys.executable, "train.py", "--engine-data", str(data), "--arch", "phase",
               "--epochs", "1", "--crop", "32", "--bptt", "3", "--augment",
               "--warmup-frames", "2", "--loss-halo", "8", "--edge-crop-prob", "0.5",
               "--val-scenes", "1", "--eval-crops", "1", "--device", "cpu", "--no-full-eval"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    teacher = Path(data) / "teacher.pt"
    for extra in (["--save", str(teacher)],
                  ["--exposure-aug", "0.05,20", "--cold-start-prob", "0", "--ssim-weight", "0.1", "--grad-weight", "0.1",
                   "--edge-grad-weight", "0.1", "--freq-weight", "0.1", "--temporal-through"],
                  ["--exposure-aug", "0.05,20", "--cold-start-prob", "0",
                   "--distill-from", str(teacher.with_name("teacher_last.pt"))]):
        result = subprocess.run(command + extra, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "epoch   0" in result.stdout and "learned (last)" in result.stdout
        print("Engine training smoke passed: " + " ".join(extra))


def static_pan():
    rh, rw, n = 30, 44, 6
    gt = torch.rand(2 * rh, 2 * rw, 3)
    frames = [dict(lr=gt[::2, ::2].clone(), gt=gt, mv=torch.zeros(rh, rw, 2), depth=torch.ones(rh, rw),
                   jitter=(0., 0.), static=True, mask=torch.ones(2 * rh, 2 * rw)) for _ in range(n)]
    moved = 0
    for seed in range(20):
        crops = crop_sequence(frames, 16, 2, rng=random.Random(seed), static_pan=3)
        v = (crops[1]["mv"][0, 0] * torch.tensor([16., 16.])).round().long() if n > 1 else torch.zeros(2)
        vx, vy = int(v[0]), int(v[1])
        assert torch.allclose(crops[0]["mv"], torch.zeros_like(crops[0]["mv"]))
        moved += (vx, vy) != (0, 0)
        for t in range(1, n):
            assert torch.allclose(crops[t]["mv"], crops[1]["mv"])
            a, b = crops[t]["lr"], crops[t - 1]["lr"]
            # crop t at p equals crop t-1 at p + v (the motion points at the previous position)
            ys, xs = slice(max(0, -vy), 16 - max(0, vy)), slice(max(0, -vx), 16 - max(0, vx))
            yp, xp = slice(ys.start + vy, ys.stop + vy), slice(xs.start + vx, xs.stop + vx)
            assert torch.equal(a[ys, xs], b[yp, xp])
            g, h = crops[t]["gt"], crops[t - 1]["gt"]
            Ys, Xs = slice(2 * ys.start, 2 * ys.stop), slice(2 * xs.start, 2 * xs.stop)
            assert torch.equal(g[Ys, Xs], h[Ys.start + 2 * vy:Ys.stop + 2 * vy, Xs.start + 2 * vx:Xs.stop + 2 * vx])
    assert moved >= 15
    still = [dict(f, static=False) for f in frames]
    crops = crop_sequence(still, 16, 2, rng=random.Random(0), static_pan=3)
    assert all(torch.equal(c["lr"], crops[0]["lr"]) for c in crops)
    print("Static-scene synthetic pan: exact registration and motion, non-static untouched")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as directory:
        make_fake_engine(directory, scenes=2, frames=6, render=(40, 56))
        raw = load_engine_scene(str(Path(directory) / "scene_00"), half=True)
        frames = geometry(raw)
        print("Involution, motion, and non-square UV geometry passed")
        sampling(raw)
        halo()
        recurrent(raw)
        smoke(directory)
        jitter_geometry(frames)
        static_pan()
    print("test_augment passed")


if __name__ == "__main__":
    main()
