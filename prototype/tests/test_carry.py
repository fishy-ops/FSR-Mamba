"""Raw sample recurrence, display residuals, and training/checkpoint integration."""

import argparse
import copy
import itertools
import math
from pathlib import Path
import re
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, build_model, infer_config,
                             load_checkpoint, load_sidecar, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator
from fsrmamba.fused import _validate
from fsrmamba.synth import halton_jitter
from test_scripts import run
from train import loss_core


def make(**kwargs):
    opts = dict(widths=(16, 32), depths=(1, 1), n_state=0, accum=True, carry_raw=True)
    opts.update(kwargs)
    return FastAccumulator((8, 12), (16, 24), **opts).eval()


def sample_base(lr, jitter, nearest):
    phases = []
    h, w = lr.shape[:2]
    for py, px in itertools.product((0.25, 0.75), repeat=2):
        ny = round(py - 0.5 + jitter[1]) if nearest else 0
        nx = round(px - 0.5 + jitter[0]) if nearest else 0
        ys = (torch.arange(h) + ny).clamp(0, h - 1)
        xs = (torch.arange(w) + nx).clamp(0, w - 1)
        phases.append(lr[ys[:, None], xs[None, :]].permute(2, 0, 1))
    return torch.stack(phases, dim=1)[None]


@torch.no_grad()
def test_identities():
    lr, mv, depth = torch.rand(8, 12, 3), torch.zeros(8, 12, 2), torch.ones(8, 12)
    for nearest, consistent, detail in itertools.product((False, True), (False, True), (0, 8)):
        model = make(nearest_sample=nearest, conf_consistent=consistent, detail_ch=detail)
        plain = make(carry_raw=False, nearest_sample=nearest, conf_consistent=consistent,
                     detail_ch=detail)
        for jitter in ((0.4, -0.4), (-0.4, 0.4), (0.0, 0.0)):
            out, state = model(model.init_state(), lr, mv, depth, jitter)
            expected = F.pixel_shuffle(sample_base(lr, jitter, nearest).reshape(1, 12, 8, 12), 2)
            assert torch.equal(out, expected[0].permute(1, 2, 0))
            assert torch.equal(state.color, expected)
            assert torch.equal(model._last_carry, state.color)
            assert model._last_alpha.eq(1).all() and model._last_reset.eq(1).all()
        state, other = model.init_state(), plain.init_state()
        for i in range(12):
            jitter = halton_jitter(i)
            _, state = model(state, lr, mv, depth, jitter)
            _, other = plain(other, lr, mv, depth, jitter)
            assert torch.equal(model._last_alpha, plain._last_alpha)
            assert torch.equal(state.conf, other.conf)


@torch.no_grad()
def test_residual_and_bounds():
    for nearest, detail in itertools.product((False, True), (0, 8)):
        model = make(nearest_sample=nearest, detail_ch=detail, conf_consistent=True,
                     depth_test=True)
        other = copy.deepcopy(model)
        last = model.detail_out if detail else model.out
        rep = 1 if detail else 4
        last.bias[:3 * model.p * rep].fill_(0.3)
        state, clean = model.init_state(), other.init_state()
        for i in range(4):
            lr = torch.rand(8, 12, 3)
            mv = (torch.rand(8, 12, 2) - 0.5) / 12
            depth = torch.ones(8, 12)
            jitter = halton_jitter(i)
            out, state = model(state, lr, mv, depth, jitter)
            zero, clean = other(clean, lr, mv, depth, jitter)
            assert torch.equal(state.color, clean.color)
            torch.testing.assert_close(out - zero, torch.full_like(out, 0.3), atol=6e-8, rtol=0)

        for parameter in model.parameters():
            parameter.uniform_(-0.2, 0.2)
        state = model.init_state()
        lo, hi = torch.ones(3), torch.zeros(3)
        for i in range(12):
            lr = 0.1 + torch.rand(8, 12, 3) * 0.8
            lo = torch.minimum(lo, lr.amin(dim=(0, 1)))
            hi = torch.maximum(hi, lr.amax(dim=(0, 1)))
            mv = (torch.rand(8, 12, 2) - 0.5) * 0.2
            _, state = model(state, lr, mv, torch.ones(8, 12), halton_jitter(i))
            assert (state.color >= lo[None, :, None, None] - 1e-7).all()
            assert (state.color <= hi[None, :, None, None] + 1e-7).all()


@torch.no_grad()
def test_innovation_and_beta():
    for nearest, detail in itertools.product((False, True), (0, 8)):
        model = make(nearest_sample=nearest, detail_ch=detail, depth_test=True)
        state = model.init_state()
        state.frame_index = 2
        state.color.uniform_(-2, 3)
        lr, mv = torch.rand(8, 12, 3), torch.zeros(8, 12, 2)
        depth = torch.ones(8, 12)
        depth[:2] = 2
        jitter = (0.4, -0.4)
        packed = []
        hook = model.stem.register_forward_pre_hook(lambda module, args: packed.append(args[0]))
        out, new = model(state, lr, mv, depth, jitter)
        hook.remove()
        rgb = lr.permute(2, 0, 1)[None]
        mx, mn = F.max_pool2d(rgb, 3, 1, 1), -F.max_pool2d(-rgb, 3, 1, 1)
        rng = mx - mn
        hist = model.reproject(state, mv.repeat_interleave(2, 0).repeat_interleave(2, 1)[None])[:, :3]
        raw = F.pixel_unshuffle(hist, 2).reshape(1, 3, 4, 8, 12)
        base = sample_base(lr, jitter, nearest)
        luma = lambda t: 0.25 * t[:, 0] + 0.5 * t[:, 1] + 0.25 * t[:, 2]
        expected = ((luma(raw) - luma(base)) / (luma(rng)[:, None] + 0.02)).clamp(-4, 4)
        expected *= 1 - model._last_reset
        inputs = F.pixel_shuffle(packed[0], 2)
        assert torch.equal(inputs[:, -model.p:], expected)
        clamped = torch.minimum(torch.maximum(raw, (mn - rng * model.box_slack.abs())[:, :, None]),
                                (mx + rng * model.box_slack.abs())[:, :, None])
        mix = torch.lerp(raw, clamped, torch.sigmoid(torch.tensor(3.0)))
        carry = torch.lerp(mix, base, model._last_alpha)
        expected_carry = F.pixel_shuffle(carry.reshape(1, 12, 8, 12), 2)
        assert torch.equal(new.color, expected_carry)
        assert torch.equal(out, expected_carry[0].permute(1, 2, 0).clamp(min=0))


def test_config(tmp):
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-accum", "--fast-carry-raw"])
    assert model_kwargs(args)["carry_raw"]
    cases = itertools.product(((16, 32), (32, 64)), (False, True), (False, True), (0, 8))
    for i, (widths, depth, nearest, detail) in enumerate(cases):
        cfg = dict(FAST_DEFAULTS, arch="fast", widths=list(widths), n_state=0,
                   accum=True, carry_raw=True, depth_test=depth, nearest_sample=nearest,
                   detail_ch=detail)
        model = build_model(cfg, (8, 12), (16, 24))
        sd = model.state_dict()
        assert "_carry_marker" in sd and "_carry_marker" not in dict(model.named_parameters())
        inferred = infer_config(sd)
        assert inferred == dict(cfg, depth_test=False, nearest_sample=False)
        assert infer_config(sd, {"depth_test": depth, "nearest_sample": nearest}) == cfg
        build_model(inferred, (8, 12), (16, 24)).load_state_dict(sd, strict=True)
        path = tmp / f"carry_{i}.pt"
        torch.save(sd, path)
        loaded, loaded_cfg = load_checkpoint(path, (8, 12), (16, 24))
        assert loaded_cfg == inferred and loaded.carry_raw
        save_sidecar(path, cfg)
        assert load_sidecar(path) == cfg
        loaded, loaded_cfg = load_checkpoint(path, (8, 12), (16, 24))
        assert loaded_cfg == cfg
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in sd.items())
    for opts in (dict(accum=False), dict(learned_clamp=True), dict(hist_residual=True)):
        try:
            make(**opts)
        except ValueError as exc:
            assert "carry_raw" in str(exc)
        else:
            raise AssertionError("Invalid carry_raw combination accepted")
    _validate(make())   # the fused path supports carry_raw


@torch.no_grad()
def test_default_off():
    torch.manual_seed(31)
    default = FastAccumulator((8, 12), (16, 24), widths=(16, 32), n_state=0, accum=True)
    torch.manual_seed(31)
    explicit = make(carry_raw=False, depths=(1, 2))
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in default.state_dict().items())
    assert not hasattr(default, "_carry_marker")
    assert not hasattr(default, "_last_carry")
    assert not any("carry" in k for k, _ in default.named_parameters())
    state, other = default.init_state(), explicit.init_state()
    for i in range(4):
        lr, mv = torch.rand(8, 12, 3), torch.rand(8, 12, 2) * 0.02
        inputs = (lr, mv, torch.ones(8, 12), halton_jitter(i))
        out, state = default(state, *inputs)
        same, other = explicit(other, *inputs)
        assert torch.equal(out, same) and torch.equal(state.color, other.color)
    assert not hasattr(default, "_last_carry")


def test_backward():
    model = make(nearest_sample=True, conf_consistent=True)
    state = model.init_state()
    loss = 0
    for i in range(3):
        out, state = model(state, torch.rand(8, 12, 3), torch.zeros(8, 12, 2),
                           torch.ones(8, 12), halton_jitter(i))
        gt = torch.rand_like(out)
        carry = model._last_carry[0].permute(1, 2, 0)
        loss = loss + F.l1_loss(loss_core(out, 2), loss_core(gt, 2))
        loss = loss + 0.25 * F.l1_loss(loss_core(carry, 2), loss_core(gt, 2))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    grad = model.out.weight.grad
    assert grad[3 * model.p * 4:4 * model.p * 4].abs().sum() > 0
    assert grad[4 * model.p * 4:5 * model.p * 4].abs().sum() > 0


def test_train(tmp):
    data = tmp / "engine"
    run("tools/make_fake_engine.py", "--out", data, "--scenes", 2, "--frames", 4,
        "--render", "32x32")
    path = tmp / "trained.pt"
    text = run("train.py", "--engine-data", data, "--arch", "fast", "--fast-accum",
               "--fast-carry-raw", "--fast-nearest-sample", "--fast-conf-consistent",
               "--epochs", 1, "--crop", 32, "--bptt", 3, "--val-scenes", 1,
               "--eval-crops", 1, "--device", "cpu", "--no-full-eval", "--save", path,
               "--loss-halo", 2)
    losses = re.findall(r"epoch\s+\d+\s+loss\s+(\S+)", text)
    assert losses and all(math.isfinite(float(value)) for value in losses), text
    assert load_sidecar(path)["carry_raw"]
    loaded, cfg = load_checkpoint(path, (32, 32), (64, 64))
    assert loaded.carry_raw and cfg["nearest_sample"] and cfg["conf_consistent"]
    print(f"carry_raw training smoke: loss {losses[0]}")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    test_identities()
    test_residual_and_bounds()
    test_innovation_and_beta()
    test_default_off()
    test_backward()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_config(tmp)
        test_train(tmp)
    print("test_carry passed")


if __name__ == "__main__":
    main()
