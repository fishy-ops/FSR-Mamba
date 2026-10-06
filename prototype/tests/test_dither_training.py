"""Stochastic coverage sampling, edge supervision, and coverage update scaling."""
import copy
import random
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.augment import augment_dither, dither_zone, luma_range
from fsrmamba.config import infer_config, load_checkpoint, save_sidecar
from fsrmamba.fast import FastAccumulator, _coverage_coefficient
from train import edge_l1, training_optimizer


def clip(count=256, dtype=torch.float32):
    gt = torch.full((24, 40, 3), .3, dtype=dtype)
    # Separate sub-pixel structures, each exceeding the .08 edge threshold.
    gt[10:12, 6:8] = torch.tensor([.3, .33, .36, .4], dtype=dtype).reshape(2, 2, 1)
    gt[10:12, 30:32] = torch.tensor([.4, .36, .33, .3], dtype=dtype).reshape(2, 2, 1)
    frame = dict(lr=torch.full((12, 20, 3), .2, dtype=dtype), gt=gt,
                 mv=torch.zeros(24, 40, 2), depth=torch.ones(24, 40), jitter=(.25, -.25))
    return [dict(frame) for _ in range(count)]


def test_dither():
    frames = clip()
    rng = random.Random(14)
    state = rng.getstate()
    assert augment_dither(frames, 0, rng) is frames and rng.getstate() == state
    zone = dither_zone(frames, random.Random(5), region_fraction=1)
    assert zone.any() and (~zone).any()
    out = augment_dither(frames, 1, random.Random(14), region_fraction=1)
    replay = augment_dither(frames, 1, random.Random(14), region_fraction=1)
    avg = torch.stack([f["lr"] for f in out]).mean(0)
    box = F.avg_pool2d(frames[0]["gt"].permute(2, 0, 1)[None], 2, 2)[0].permute(1, 2, 0)
    error = (avg[zone] - box[zone]).abs().max().item()
    assert error < 1e-2, error
    assert not torch.equal(out[0]["lr"], out[1]["lr"])
    for original, sample, again in zip(frames, out, replay):
        assert torch.equal(sample["lr"], again["lr"])
        assert torch.equal(sample["lr"][~zone], original["lr"][~zone])
        assert sample["lr"].shape == original["lr"].shape
        assert sample["lr"].dtype == original["lr"].dtype
        assert torch.equal(original["lr"], torch.full_like(original["lr"], .2))
        for key in ("gt", "mv", "depth", "jitter"):
            assert sample[key] is original[key]
    # A connected component is kept or discarded whole, never as isolated pixels.
    partial = dither_zone(frames, random.Random(1))
    assert torch.equal(partial[:, :10], zone[:, :10]) and not partial[:, 10:].any()
    assert not dither_zone(frames, random.Random(1), region_fraction=0).any()
    for dtype in (torch.float16, torch.float64):
        assert augment_dither(clip(1, dtype), 1, random.Random(1))[0]["lr"].dtype == dtype
    # Motion changes the footprint's colours: always sample this frame's GT.
    moving = clip(8)
    for i, frame in enumerate(moving):
        frame["gt"] = frame["gt"].roll(2*i, 1) + i * .01
        frame["mv"] = torch.full_like(frame["mv"], .02*i)
    out = augment_dither(moving, 1, random.Random(3), region_fraction=1)
    zone = dither_zone(moving, random.Random(3), region_fraction=1)
    for f, sample in zip(moving, out):
        taps = f["gt"].reshape(12, 2, 20, 2, 3).permute(0, 2, 1, 3, 4).reshape(12, 20, 4, 3)
        assert ((sample["lr"][..., None, :] == taps).all(-1).any(-1) | ~zone).all()
        assert sample["mv"] is f["mv"]
    print(f"dither: identity, seeded regions, per-frame footprints, dtypes; 256-frame max mean error {error:.6g}")


def test_edges():
    gt = torch.full((10, 12, 3), .2)
    gt[:, 6:] = .7
    mask = luma_range(gt) > .08
    assert mask[:, 5:7].all() and mask.sum() == 20
    out = (gt + .1).requires_grad_()
    assert torch.allclose(edge_l1(out, gt), torch.tensor(.1))
    edge_l1(out, gt).backward()
    assert out.grad[~mask].eq(0).all() and out.grad[mask].gt(0).all()
    # Edge normalisation ignores flat area and invalid references.
    valid = torch.zeros_like(mask, dtype=torch.float32)
    valid[:2] = 1
    assert torch.allclose(edge_l1(out, gt, mask=valid), torch.tensor(.1))
    assert torch.allclose(edge_l1(out, gt, halo=2), torch.tensor(.1))
    for target in (torch.full_like(gt, .2), gt):
        value = edge_l1(out, target, mask=torch.zeros_like(valid))
        assert value.item() == 0 and torch.isfinite(value)
    assert edge_l1(out, torch.full_like(gt, .2)).item() == 0
    print("edge loss: local mask, edge normalisation, halo, validity, gradients and empty edges passed")


def test_optimizer():
    for detail in (0, 4):
        model = FastAccumulator((8, 12), (16, 24), widths=(8,), depths=(0,),
                                n_state=2, coverage=True, detail_ch=detail)
        base, neutral, boosted = (copy.deepcopy(model) for _ in range(3))
        optimizers = (torch.optim.Adam(base.parameters(), lr=.001),
                      training_optimizer(neutral, .001), training_optimizer(boosted, .001, 5))
        scaled = {id(p): index for p, index in optimizers[2].slices}
        for step in range(3):
            # Identical varying gradients isolate LR scaling from recurrent forward changes.
            before = [[p.detach().clone() for p in m.parameters()] for m in (base, neutral, boosted)]
            for params in zip(base.parameters(), neutral.parameters(), boosted.parameters()):
                grad = torch.randn_like(params[0])
                for p in params:
                    p.grad = grad.clone()
            for opt in optimizers:
                opt.param_groups[0]["lr"] = .001 / (step+1)
                opt.step()
            for i, (a, b, c) in enumerate(zip(base.parameters(), neutral.parameters(), boosted.parameters())):
                assert torch.equal(a, b)
                factor = torch.ones_like(c)
                if id(c) in scaled:
                    factor[scaled[id(c)]] = 5
                assert torch.allclose(c-before[2][i], (a-before[0][i])*factor, atol=2e-7, rtol=1e-4)
                unchanged = factor == 1
                assert torch.equal(a[unchanged], c[unchanged])
    print("coverage LR: Adam bit identity at K=1, 5x head/input updates, unchanged other weights, scheduled LR passed")


@torch.no_grad()
def test_calibration():
    for dtype in (torch.float16, torch.float32, torch.float64):
        for bias in (-6., -3.):
            assert _coverage_coefficient(torch.full((16,), bias, dtype=dtype), bias).eq(0).all()
    old = FastAccumulator((8, 12), (16, 24), coverage=True, coverage_bias=-6)
    old.out.bias[old._cov_offset*4:(old._cov_offset+old.p)*4] = -4
    sd = old.state_dict()
    del sd["coverage_bias"]
    fresh = FastAccumulator((8, 12), (16, 24), coverage=True)
    fresh.load_state_dict(sd)
    assert float(fresh.coverage_bias) == -6
    lr, mv, depth = torch.rand(8, 12, 3), torch.zeros(8, 12, 2), torch.ones(8, 12)
    expected, _ = old(old.init_state(), lr, mv, depth, (0, 0))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "legacy.pt"
        torch.save(sd, path)
        for sidecar in (False, True):
            if sidecar:
                cfg = infer_config(sd)
                save_sidecar(path, cfg)
                import json
                cfg = json.loads(Path(str(path)+".json").read_text())
                del cfg["coverage_bias"]
                Path(str(path)+".json").write_text(json.dumps(cfg))
            restored, cfg = load_checkpoint(path, (8, 12), (16, 24))
            assert cfg["coverage_bias"] == -6 and float(restored.coverage_bias) == -6
            actual, _ = restored(restored.init_state(), lr, mv, depth, (0, 0))
            assert torch.equal(actual, expected)
    print("calibration: exact neutral -3/-6 in three dtypes; trained legacy checkpoint with/without sidecar passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(43)
    test_dither()
    test_edges()
    test_optimizer()
    test_calibration()
