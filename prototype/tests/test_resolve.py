"""Phase resolve compatibility and sample-constrained Fast current frames."""

import argparse
import copy
import itertools
import math
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.baseline import FSRAccumulator, _lanczos2
from fsrmamba.colorspace import ycocg_to_rgb
from fsrmamba.config import (FAST_DEFAULTS, PHASE_DEFAULTS, add_arch_args, build_model,
                             infer_config, load_checkpoint, load_sidecar, model_kwargs,
                             save_sidecar)
from fsrmamba.fast import FastAccumulator
from fsrmamba.phase import PhaseAccumulator
from fsrmamba.resolve import phase_kernels, window_bounds
from fsrmamba.synth import halton_jitter
from test_carry import sample_base
from test_scripts import run

_BOX_CURVE = -2.3


class _OldPhase(PhaseAccumulator):
    # Frozen pre-refactor kernel calculation.
    def _kernels(self, jitter, device):
        j = jitter if torch.is_tensor(jitter) else torch.tensor(jitter, device=device)
        j = j.float()
        ox = self._tap_x + 0.5 - j[0] - self._ph_x
        oy = self._tap_y + 0.5 - j[1] - self._ph_y
        # The 3x3 window starts at -2 when the base sample lies beyond the output point.
        sx = torch.where(0.5 - j[0] - self._ph_x > 0, -2.0, -1.0)
        sy = torch.where(0.5 - j[1] - self._ph_y > 0, -2.0, -1.0)
        inwin = ((self._tap_x >= sx) & (self._tap_x <= sx + 2)
                 & (self._tap_y >= sy) & (self._tap_y <= sy + 2)).float()
        r2 = ox * ox + oy * oy
        r = torch.sqrt(r2)
        k0 = _lanczos2(r) * inwin
        k1 = _lanczos2(r * self.sharp_bias) * inwin
        g = torch.exp(_BOX_CURVE * r2) * inwin
        s0 = k0.sum(dim=(1, 2), keepdim=True)
        s1 = k1.sum(dim=(1, 2), keepdim=True)
        # The sharp kernel can lose all its weight when no tap is near; fall back to k0.
        k1 = torch.where(s1 > 1e-3, k1 / s1.clamp(min=1e-3), k0 / s0.clamp(min=1e-4))
        k0 = k0 / s0.clamp(min=1e-4)
        g = g / g.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        prior = torch.log(s0.clamp(min=1e-4) / s0.mean()).reshape(1, self.p, 1, 1)
        win = [(int(sy[i].item()) + 2, int(sx[i].item()) + 2) for i in range(self.p)]
        return j, k0.unsqueeze(1), k1.unsqueeze(1), g.unsqueeze(1), prior, win


def _old_bounds(taps, win, h, w):
    mxp = F.max_pool2d(taps, 3, stride=1)
    mnp = -F.max_pool2d(-taps, 3, stride=1)
    mx = torch.cat([mxp[:, :, a: a + h, b: b + w] for a, b in win], dim=1).unsqueeze(0)
    mn = torch.cat([mnp[:, :, a: a + h, b: b + w] for a, b in win], dim=1).unsqueeze(0)
    return mn, mx


def _assert_state(a, b):
    assert torch.equal(a.color, b.color) and torch.equal(a.feat, b.feat)
    assert a.frame_index == b.frame_index
    if hasattr(a, "conf"):
        assert torch.equal(a.depth, b.depth)
        assert (a.conf is None and b.conf is None) or torch.equal(a.conf, b.conf)


@torch.no_grad()
def test_phase():
    for scale, residual in itertools.product((1, 2, 3), (False, True)):
        opts = dict(widths=(8, 16), depths=(1, 1), n_state=2, residual=residual)
        output = (8 * scale, 12 * scale)
        torch.manual_seed(71)
        model = PhaseAccumulator((8, 12), output, **opts)
        old = _OldPhase((8, 12), output, **opts)
        for parameter in model.parameters():
            parameter.uniform_(-0.08, 0.08)
        old.load_state_dict(model.state_dict(), strict=True)
        negative = PhaseAccumulator((8, 12), output, **opts, jitter_sign=-1)
        negative.load_state_dict(dict(model.state_dict(), _jitter_sign=torch.tensor(-1.0)))
        assert "_jitter_sign" not in model.state_dict()
        assert "_jitter_sign" in negative.state_dict()
        assert "_jitter_sign" not in dict(negative.named_parameters())
        state, legacy, neg_state, flipped = (m.init_state() for m in (model, old, negative, model))
        for i in range(5):
            jitter = halton_jitter(i)
            if i % 2:
                jitter = torch.tensor(jitter)
            kernels, original = model._kernels(jitter, "cpu"), old._kernels(jitter, "cpu")
            assert all(torch.equal(a, b) for a, b in zip(kernels[:-1], original[:-1]))
            assert kernels[-1] == original[-1]
            lr, mv = torch.rand(8, 12, 3), torch.rand(8, 12, 2) * 0.02
            depth = torch.ones(8, 12)
            out, state = model(state, lr, mv, depth, jitter)
            with patch("fsrmamba.phase.window_bounds", _old_bounds):
                previous, legacy = old(legacy, lr, mv, depth, jitter)
            assert torch.equal(out, previous)
            _assert_state(state, legacy)
            neg, neg_state = negative(neg_state, lr, mv, depth, jitter)
            inverse = -jitter if torch.is_tensor(jitter) else tuple(-v for v in jitter)
            pos, flipped = model(flipped, lr, mv, depth, inverse)
            assert torch.equal(neg, pos)
            _assert_state(neg_state, flipped)
    print("Phase sign +1 matches frozen kernels and window bounds bit-for-bit; sign reversal passed")


def _make(**opts):
    cfg = dict(widths=(8, 16), depths=(1, 1), n_state=0, base_gate=True)
    cfg.update(opts)
    return FastAccumulator((16, 20), (32, 40), **cfg).eval()


def _set_gate(model, value):
    head = model.detail_out if model.detail_ch else model.out
    rep = 1 if model.detail_ch else 4
    q = model._base_gate_offset * rep
    head.bias[q: q + model.p * rep].fill_(value)


def _lanczos_rgb(lr, jitter):
    h, w = lr.shape[:2]
    phases = []
    rgb = lr.permute(2, 0, 1)
    # Independent gather over each phase's own 3x3 window.
    for py, px in itertools.product((0.25, 0.75), repeat=2):
        sx = -2 if 0.5 - jitter[0] - px > 0 else -1
        sy = -2 if 0.5 - jitter[1] - py > 0 else -1
        samples, weights = [], []
        for y, x in itertools.product(range(sy, sy + 3), range(sx, sx + 3)):
            ys = (torch.arange(h) + y).clamp(0, h - 1)
            xs = (torch.arange(w) + x).clamp(0, w - 1)
            samples.append(rgb[:, ys[:, None], xs[None, :]])
            ox, oy = x + 0.5 - jitter[0] - px, y + 0.5 - jitter[1] - py
            weights.append(_lanczos2(torch.sqrt(torch.tensor(ox * ox + oy * oy))))
        taps, weights = torch.stack(samples), torch.stack(weights)
        color = (taps * (weights / weights.sum())[:, None, None, None]).sum(0)
        phases.append(torch.minimum(torch.maximum(color, taps.amin(0)), taps.amax(0)))
    return torch.stack(phases, dim=1)[None]


def _shuffle(phases):
    _, _, p, h, w = phases.shape
    return F.pixel_shuffle(phases.reshape(1, 3 * p, h, w), math.isqrt(p))[0].permute(1, 2, 0)


@torch.no_grad()
def test_gate_endpoints():
    torch.manual_seed(73)
    lr = torch.rand(16, 20, 3) * 0.8 + 0.1
    mv, depth = torch.zeros(16, 20, 2), torch.ones(16, 20)
    options = itertools.product((False, True), (False, True), (False, True), (0, 8), (1, -1))
    for accum, nearest, hres, detail, sign in options:
        for carry in ((False, True) if accum and not hres else (False,)):
            model = _make(accum=accum, nearest_sample=nearest, hist_residual=hres,
                          detail_ch=detail, jitter_sign=sign, carry_raw=carry)
            head = model.detail_out if detail else model.out
            rep = 1 if detail else 4
            q = (4 + 3 * hres) * model.p * rep
            assert head.bias[q:q + model.p * rep].eq(2).all()
            if accum:
                assert head.bias[(model.n_px - model.p) * rep:model.n_px * rep].eq(3).all()
            if carry:
                assert head.bias[q + model.p * rep:q + 2 * model.p * rep].eq(3).all()
            for jitter in ((0.4, -0.4), (-0.4, 0.4), (0.0, 0.0)):
                corrected = tuple(v * sign for v in jitter)
                _set_gate(model, -40)
                out, state = model(model.init_state(), lr, mv, depth, jitter)
                kernels = phase_kernels(corrected, "cpu", model._ph_x, model._ph_y,
                                        model._tap_x, model._tap_y)
                taps = F.pad(lr.permute(2, 0, 1).unsqueeze(1), (2, 1, 2, 1), mode="replicate")
                resolved = F.conv2d(taps, kernels[1]).unsqueeze(0)
                lo, hi = window_bounds(taps, kernels[-1], 16, 20)
                resolved = torch.minimum(torch.maximum(resolved, lo), hi)
                assert torch.equal(out, _shuffle(resolved))
                expected = _shuffle(_lanczos_rgb(lr, corrected))
                torch.testing.assert_close(out, expected, atol=3e-7, rtol=0)
                _set_gate(model, 40)
                out, state = model(model.init_state(), lr, mv, depth, jitter)
                expected = _shuffle(sample_base(lr, corrected, nearest))
                assert torch.equal(out, expected)
                if carry:
                    assert torch.equal(state.color[0].permute(1, 2, 0), expected)
    # Window clamp must use this phase's taps even if the centre 3x3 excludes an extreme.
    edge = torch.zeros_like(lr)
    edge[6, 6] = 1
    model = _make()
    _set_gate(model, -40)
    out, _ = model(model.init_state(), edge, mv, depth, (0.4, -0.4))
    torch.testing.assert_close(out, _shuffle(_lanczos_rgb(edge, (0.4, -0.4))), atol=2e-7, rtol=0)
    # Even half-precision model buffers produce float32 kernels.
    half = model.half()
    kernels = phase_kernels(torch.tensor([0.4, -0.4], dtype=torch.float16), "cpu",
                            half._ph_x, half._ph_y, half._tap_x, half._tap_y)
    assert kernels[1].dtype == torch.float32


@torch.no_grad()
def test_baseline():
    y, x = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 20), indexing="ij")
    lr = torch.stack((0.45 + 0.25 * torch.sin(2 * x + y),
                      0.45 + 0.25 * torch.cos(x - 2 * y),
                      0.35 + 0.2 * torch.sin(3 * x - y)), dim=-1)
    model = _make(jitter_sign=1)
    _set_gate(model, -40)
    baseline = FSRAccumulator((16, 20), (32, 40))
    error = 0.0
    for jitter in ((0.4, -0.4), (-0.4, 0.4), (0.125, -0.375), (0.0, 0.0)):
        out, _ = model(model.init_state(), lr, torch.zeros(16, 20, 2), torch.ones(16, 20), jitter)
        expected = ycocg_to_rgb(baseline._upsample(lr, jitter)[0])
        error = max(error, float((out[4:-4, 4:-4] - expected[4:-4, 4:-4]).abs().max()))
    assert error < 5e-3, error
    print(f"RGB vs YCoCg deringing: maximum interior error {error:.9g} (limit 5e-3)")


@torch.no_grad()
def test_carry_and_backward():
    for detail, nearest in itertools.product((0, 8), (False, True)):
        model = _make(accum=True, carry_raw=True, detail_ch=detail, nearest_sample=nearest)
        clean = copy.deepcopy(model)
        head = model.detail_out if detail else model.out
        rep = 1 if detail else 4
        head.bias[:3 * model.p * rep].fill_(0.3)
        state, other = model.init_state(), clean.init_state()
        for i in range(4):
            lr, mv = torch.rand(16, 20, 3), torch.rand(16, 20, 2) * 0.02
            inputs = (lr, mv, torch.ones(16, 20), halton_jitter(i))
            out, state = model(state, *inputs)
            plain, other = clean(other, *inputs)
            _assert_state(state, other)
            torch.testing.assert_close(out - plain, torch.full_like(out, 0.3), atol=1e-7, rtol=0)
            assert torch.equal(state.color, model._last_carry)
    with torch.enable_grad():
        for accum, carry, hres, learned in ((False, False, True, True), (True, False, True, False),
                                            (True, True, False, False)):
            model = _make(accum=accum, carry_raw=carry, hist_residual=hres, learned_clamp=learned)
            state, loss = model.init_state(), 0
            for i in range(3):
                out, state = model(state, torch.rand(16, 20, 3), torch.zeros(16, 20, 2),
                                   torch.ones(16, 20), halton_jitter(i))
                loss = loss + F.l1_loss(out, torch.rand_like(out))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            q = model._base_gate_offset * 4
            assert model.out.weight.grad[q:q + model.p * 4].abs().sum() > 0
            if carry:
                q = model._carry_beta_offset * 4
                assert model.out.weight.grad[q:q + model.p * 4].abs().sum() > 0


@torch.no_grad()
def test_default_off():
    for accum, carry, hres, detail in ((False, False, False, 0), (True, False, True, 8),
                                      (True, True, False, 0)):
        opts = dict(accum=accum, carry_raw=carry, hist_residual=hres, detail_ch=detail)
        torch.manual_seed(79)
        default = FastAccumulator((16, 20), (32, 40), widths=(8, 16), depths=(1, 1),
                                  n_state=0, **opts)
        torch.manual_seed(79)
        explicit = _make(base_gate=False, **opts)
        assert default.state_dict().keys() == explicit.state_dict().keys()
        assert all(torch.equal(v, explicit.state_dict()[k]) for k, v in default.state_dict().items())
        assert not hasattr(default, "_base_gate_marker")
        state, other = default.init_state(), explicit.init_state()
        for i in range(3):
            inputs = (torch.rand(16, 20, 3), torch.rand(16, 20, 2) * 0.02,
                      torch.ones(16, 20), halton_jitter(i))
            out, state = default(state, *inputs)
            same, other = explicit(other, *inputs)
            assert torch.equal(out, same)
            _assert_state(state, other)


def test_config(tmp):
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-base-gate", "--jitter-sign", "-1"])
    assert model_kwargs(args)["base_gate"] and model_kwargs(args)["jitter_sign"] == -1
    cases = itertools.product((False, True), (False, True), (0, 8), (0, 2), (1, 2, 3))
    for i, (accum, carry, detail, n_state, scale) in enumerate(cases):
        if carry and not accum:
            continue
        cfg = dict(FAST_DEFAULTS, arch="fast", widths=[8, 16], depths=[1, 1], n_state=n_state,
                   base_gate=True, accum=accum, carry_raw=carry, detail_ch=detail, jitter_sign=-1)
        output = (8 * scale, 12 * scale)
        model = build_model(cfg, (8, 12), output)
        sd = model.state_dict()
        assert "_base_gate_marker" in sd and "_base_gate_marker" not in dict(model.named_parameters())
        assert infer_config(sd) == cfg
        build_model(infer_config(sd), (8, 12), output).load_state_dict(sd, strict=True)
        path = tmp / f"gate_{i}.pt"
        torch.save(sd, path)
        loaded, inferred = load_checkpoint(path, (8, 12), output)
        assert loaded.base_gate and inferred == cfg
        save_sidecar(path, cfg)
        assert load_sidecar(path) == cfg
        loaded, saved = load_checkpoint(path, (8, 12), output)
        assert saved == cfg and all(torch.equal(v, loaded.state_dict()[k]) for k, v in sd.items())
    for learned, hres, accum in itertools.product((False, True), repeat=3):
        cfg = dict(FAST_DEFAULTS, arch="fast", base_gate=True, learned_clamp=learned,
                   hist_residual=hres, accum=accum, detail_ch=8)
        model = build_model(cfg, (8, 12), (16, 24))
        assert infer_config(model.state_dict()) == cfg
    for sign, residual in itertools.product((1, -1), (False, True)):
        cfg = dict(PHASE_DEFAULTS, arch="phase", jitter_sign=sign, residual=residual)
        model = build_model(cfg, (8, 12), (16, 24))
        assert infer_config(model.state_dict()) == cfg
        path = tmp / f"phase_{sign}_{residual}.pt"
        torch.save(model.state_dict(), path)
        loaded, inferred = load_checkpoint(path, (8, 12), (16, 24))
        assert loaded.jitter_sign == sign and inferred == cfg
        save_sidecar(path, cfg)
        assert load_sidecar(path) == cfg
        assert load_checkpoint(path, (8, 12), (16, 24))[1] == cfg


def test_train(tmp):
    data = tmp / "engine"
    run("tools/make_fake_engine.py", "--out", data, "--scenes", 2, "--frames", 4,
        "--render", "16x16")
    for arch in ("fast", "phase"):
        path = tmp / f"trained_{arch}.pt"
        flags = (["--fast-accum", "--fast-base-gate", "--fast-nearest-sample",
                  "--fast-conf-consistent"] if arch == "fast" else [])
        text = run("train.py", "--engine-data", data, "--arch", arch, *flags,
                   "--jitter-sign", -1, "--epochs", 1, "--crop", 16, "--bptt", 3,
                   "--val-scenes", 1, "--eval-crops", 1, "--device", "cpu", "--no-full-eval",
                   "--save", path, "--loss-halo", 2, "--fast-widths", "8,16",
                   "--fast-depths", "1,1", "--fast-state", 0)
        losses = re.findall(r"epoch\s+\d+\s+loss\s+(\S+)", text)
        assert losses and all(math.isfinite(float(value)) for value in losses), text
        cfg = load_sidecar(path)
        assert cfg["jitter_sign"] == -1
        loaded, saved = load_checkpoint(path, (16, 16), (32, 32))
        assert loaded.jitter_sign == -1 and saved == cfg
        assert infer_config(loaded.state_dict())["jitter_sign"] == -1
        if arch == "fast":
            assert loaded.base_gate and cfg["accum"] and cfg["nearest_sample"] and cfg["conf_consistent"]
        print(f"{arch} sign -1 training smoke: loss {losses[0]}")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(67)
    test_phase()
    test_gate_endpoints()
    test_baseline()
    test_carry_and_backward()
    test_default_off()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_config(tmp)
        test_train(tmp)
    print("test_resolve passed")


if __name__ == "__main__":
    main()
