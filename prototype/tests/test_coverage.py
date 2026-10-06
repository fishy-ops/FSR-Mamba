"""Temporal coverage recurrence, neutral checkpoint expansion, and deployment parity."""
import argparse
import itertools
from pathlib import Path
import sys
import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, infer_config, load_checkpoint,
                            load_model_state, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator, FastState, _coverage_coefficient
from fsrmamba.fused import (FusedFast, _CUDA, _cuda_source, materialize_state,
                           ref_pack, ref_resolve, run_trunk)
from legacy_fast_reference import ref_pack as legacy_pack, ref_resolve as legacy_resolve
from test_fused import close, frames


OPTIONS = dict(widths=(8, 16), depths=(1, 1), n_state=0, accum=True, carry_raw=True,
               base_gate=True, nearest_sample=True, conf_consistent=True,
               hist_filter="bicubic", depth_test=True, depth_soft=True, mv_dilate=True)


def make(render=(64, 96), **opts):
    kwargs = dict(OPTIONS, **opts)
    m = FastAccumulator(render, tuple(2 * v for v in render), **kwargs).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, .015)
        m.out.bias.normal_(0, .1)
        m.film[2].weight.normal_(0, .01)
    return m


@torch.no_grad()
def test_parity(device="cpu"):
    worst = 0
    for render, layout, bias, robust in itertools.product(((64, 96), (63, 95)), ("nhwc", "nchw"),
                                                         (-6., -3.), ({}, dict(depth_dilate=True, thin_lock=True))):
        m = make(render, coverage=True, coverage_bias=bias, **robust).to(device)
        f = FusedFast(m, layout=layout)
        a, b = m.init_state(device), f.init_state()
        for inputs in frames(render, device, 8):
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, a = m(a, *inputs)
                actual, b = f.step(b, *inputs, write_rgb=True)
            else:
                expected, a = m(a, *inputs)
                actual, b = f.step_reference(b, *inputs)
            worst = max(worst, (expected - actual).abs().max().item())
            close(expected, actual, .008 if device == "cuda" else 3e-5)
            close(a.color, b.color, .008 if device == "cuda" else 3e-5)
            close(a.conf, b.conf, .08 if device == "cuda" else 3e-5)
            for key in ("m1", "m2", "osc", "luma"):
                assert torch.equal(getattr(a, key), getattr(b, key)), key
            assert torch.equal(a.depth, b.depth)
        retained = materialize_state(b)
        assert retained.m1.data_ptr() != b.m1.data_ptr()
        detached = a.detach()
        for key in ("m1", "m2", "osc", "luma"):
            assert not getattr(detached, key).requires_grad
    print(f"coverage {device} parity: 16 configurations, 8 frames, max RGB error {worst:.8g}")


@torch.no_grad()
def test_off():
    for render in ((64, 96), (63, 95)):
        m = make(render, depth_soft=False, mv_dilate=False)
        f = FusedFast(m)
        a, b = m.init_state(), f.init_state()
        for lr, mv, depth, jitter in frames(render, "cpu", 8):
            packed = legacy_pack(m, a, lr, mv, depth, jitter)
            current = ref_pack(m, b, lr, mv, depth, jitter)
            for key, value in packed.items():
                if value is not None:
                    assert torch.equal(value, current[key]), key
            expected, color, conf = legacy_resolve(m, run_trunk(m, packed["X"], jitter), lr, packed, jitter)
            actual, b = f.step_reference(b, lr, mv, depth, jitter)
            a = FastState(color, a.feat, depth[None, None], a.frame_index + 1, conf)
            assert torch.equal(expected, actual)
            assert torch.equal(a.color, b.color) and torch.equal(a.conf, b.conf)
            assert b.m1 is None and b.luma is None
    bare = FastAccumulator((8, 12), (16, 24), n_state=0)
    assert _cuda_source(bare) is _CUDA
    assert not any("COVERAGE" in flag for flag in FusedFast(bare)._kernel_options())
    print("coverage off: bit-identical frozen legacy pack/resolve, 8 frames at both sizes")


@torch.no_grad()
def test_warm_start():
    for opts in ({}, dict(n_state=3), dict(detail_ch=4, n_state=2)):
        a = make((16, 24), **opts)
        cfg = dict(infer_config(a.state_dict()), **{k: v for k, v in OPTIONS.items()
                                                   if k not in ("widths", "depths", "n_state")})
        cfg.update(opts)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.pt"
            torch.save(a.state_dict(), path)
            save_sidecar(path, cfg)
            b, actual = load_checkpoint(path, (16, 24), (32, 48), overrides=dict(coverage=True, coverage_bias=-3.0))
            assert actual["coverage"] and b.coverage
            n = b.stem.in_channels - b.n_state
            assert b.stem.weight[:, n-8:n].eq(0).all()
            head = b.detail_out if b.detail_ch else b.out
            rep = 1 if b.detail_ch else 4
            assert head.bias[b._cov_offset*rep:(b._cov_offset+b.p)*rep].eq(-3).all()
            sa, sb = a.init_state(), b.init_state()
            for inputs in frames((16, 24), "cpu", 8):
                old, sa = a(sa, *inputs)
                new, sb = b(sb, *inputs)
                # Expanded convolution shapes can change opmath summation order.
                close(old, new, 2e-6)
                close(sa.color, sb.color, 2e-6)
                close(sa.conf, sb.conf, 2e-6)
            if not opts:
                assert torch.equal(old, new)  # deployed head/stem are exactly neutral
            # Exercise the same loader used by train --init-from.
            fresh = make((16, 24), coverage=True, **opts)
            incompat = load_model_state(fresh, a.state_dict(), strict=False)
            assert incompat.missing_keys == ["_coverage_marker"]
            torch.save(b.state_dict(), path)
            save_sidecar(path, actual)
            restored, _ = load_checkpoint(path, (16, 24), (32, 48))
            assert infer_config(restored.state_dict())["coverage"]
            path.with_suffix(".pt.json").unlink()
            restored, _ = load_checkpoint(path, (16, 24), (32, 48))
            assert restored.coverage
    assert _coverage_coefficient(torch.tensor([-6.])).item() == 0
    assert torch.sigmoid(torch.tensor(-6.)).item() > 0
    ckpt = Path("../ckpt/w_s16_carry_bg_seed2_last.pt")
    if ckpt.exists():
        a, cfg = load_checkpoint(ckpt, (64, 96), (128, 192))
        b, _ = load_checkpoint(ckpt, (64, 96), (128, 192), overrides=dict(coverage=True, coverage_bias=-3.0))
        sa, sb = a.init_state(), b.init_state()
        for inputs in frames((64, 96), "cpu", 8):
            old, sa = a(sa, *inputs)
            new, sb = b(sb, *inputs)
            assert torch.equal(old, new)
        print("target checkpoint: exact neutral warm start passed")
    else:
        print("SKIP: w_s16_carry_bg_seed2_last.pt tensors absent; matching synthetic warm starts tested")
    print("warm-start input/head expansion, state channels, detail branch, sidecars and inference passed")


@torch.no_grad()
def test_evidence():
    render = (16, 36)
    m = make(render, coverage=True, mv_dilate=False)
    state = m.init_state()
    mv = torch.zeros(*render, 2)
    depth = torch.ones(render)
    for i in range(32):
        rgb = torch.full((*render, 3), .2)
        rgb[:, 12:24] = .8 if i % 2 else .2
        rgb[:, 24:] = .8 if i >= 4 else .2
        packed = ref_pack(m, state, rgb, mv, depth, (.25, -.25))
        evidence = F.pixel_shuffle(packed["X"], 2)[:, -2:, :render[0], :render[1]]
        _, state = m(state, rgb, mv, depth, (.25, -.25))
        assert torch.equal(m._last_coverage, evidence)
    alternating = evidence[0, :, 6, 18]
    static = evidence[0, :, 6, 6]
    step = evidence[0, :, 6, 30]
    assert (alternating > static + .1).all() and (alternating > step + .1).all()
    assert state.luma[0, 0, 6, 18] == .8
    # Offscreen reprojection and hard depth rejection seed this frame, even with stale moments.
    mv[..., 0] = 1
    _, reset = m(state, rgb, mv, depth, (0, 0))
    assert m._last_coverage.eq(0).all() and reset.osc.eq(0).all()
    assert torch.equal(reset.m1, reset.luma) and torch.equal(reset.m2, reset.luma.square())
    hard = make(render, coverage=True, depth_soft=False, mv_dilate=False)
    state.depth = torch.zeros_like(state.depth)
    _, reset = hard(state, rgb, torch.zeros_like(mv), depth, (0, 0))
    assert hard._last_coverage.eq(0).all() and reset.osc.eq(0).all()
    state.frame_index = 0
    _, reset = m(state, rgb, torch.zeros_like(mv), depth, (0, 0))
    assert m._last_coverage.eq(0).all() and reset.osc.eq(0).all()
    print(f"synthetic pre-update var/osc: alternating {alternating.tolist()}, static {static.tolist()}, step {step.tolist()}; resets passed")


@torch.no_grad()
def test_resolve_weights():
    for consistent in (False, True):
        m = make((8, 12), coverage=True, conf_consistent=consistent, mv_dilate=False)
        state = m.init_state()
        lr = torch.rand(8, 12, 3)
        motion = torch.zeros(8, 12, 2)
        depth = torch.ones(8, 12)
        packed = ref_pack(m, state, lr, motion, depth, (.25, -.25))
        o = torch.zeros(1, m.n_px * 4, *m._tr_size)
        q = m._cov_offset * 4
        o[:, q:q + m.p*4] = m.coverage_bias
        original, _, conf = ref_resolve(m, o, lr, packed, (.25, -.25))
        o[:, q:q + m.p*4] = 20
        smoothed, _, low_conf = ref_resolve(m, o, lr, packed, (.25, -.25))
        close(low_conf, conf * .1, 1e-6)
        assert not torch.equal(original, smoothed)
        # With cov=1, the base gate is zero, exactly selecting the Lanczos candidate.
        o[:, q:q + m.p*4] = m.coverage_bias
        o[:, m._base_gate_offset*4:(m._base_gate_offset+m.p)*4] = -100
        lanczos, _, _ = ref_resolve(m, o, lr, packed, (.25, -.25))
        assert torch.equal(smoothed, lanczos)
    print("coverage coefficient scales current confidence and selects the smooth base")


def test_training(depth_soft_osc=False, history_age=False, flicker_weight=0):
    from tools.make_fake_engine import make_fake_engine
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        make_fake_engine(root, scenes=2, frames=6, render=(16, 24))
        old = root / "old.pt"
        # Exercise learning inside the clamp interval; its endpoint derivative is
        # zero in PyTorch 2.14. Neutral legacy expansion is checked in test_warm_start.
        m = make((16, 24), coverage=True, coverage_bias=-3.)
        with torch.no_grad():
            q = m._cov_offset*4
            m.out.weight[q:q+m.p*4].zero_()
            m.out.bias[q:q+m.p*4].fill_(-2.5)
        torch.save(m.state_dict(), old)
        cfg = dict(infer_config(m.state_dict()), **{k: v for k, v in OPTIONS.items()
                                                   if k not in ("widths", "depths", "n_state")})
        save_sidecar(old, cfg)
        output = root / "coverage.pt"
        command = [sys.executable, "train.py", "--engine-data", str(root), "--arch", "fast",
                   "--fast-widths", "8,16", "--fast-depths", "1,1", "--fast-state", "0",
                   "--fast-accum", "--fast-carry-raw", "--fast-base-gate", "--fast-nearest-sample",
                   "--fast-conf-consistent", "--fast-hist-filter", "bicubic", "--fast-depth-test",
                   "--fast-depth-soft", "--fast-mv-dilate", "--fast-coverage", "--init-from", str(old),
                   "--save", str(output), "--epochs", "1", "--crop", "12", "--bptt", "3",
                   "--val-scenes", "1", "--eval-crops", "1", "--device", "cpu", "--no-full-eval",
                   "--lr", "0.0001", "--warmup-frames", "2", "--cold-start-prob", "0", "--dither-aug", "1",
                   "--edge-loss-weight", ".2", "--coverage-lr-mult", "4",
                   "--distill-from", str(old)]
        if depth_soft_osc:
            command.append("--fast-depth-soft-osc")
        if history_age:
            command.append("--fast-history-age")
        if flicker_weight:
            command.extend(("--flicker-weight", str(flicker_weight), "--temporal-through"))
        result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                                env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "warm-started from" in result.stdout and "epoch   0" in result.stdout
        trained, actual = load_checkpoint(output, (16, 24), (32, 48))
        assert actual["coverage"] and trained.coverage
        assert actual["depth_soft_osc"] == depth_soft_osc
        assert actual["history_age"] == history_age
        if history_age:
            assert torch.isfinite(trained.alpha_min)
            start = trained.stem.in_channels - 4
            assert trained.stem.weight[:, start:].abs().max() > 0
        if depth_soft_osc:
            assert torch.isfinite(trained.soft_osc_threshold)
        assert actual["coverage_bias"] == -3 and float(trained.coverage_bias) == -3
        cov = trained.out.weight[trained._cov_offset*4:(trained._cov_offset+trained.p)*4]
        assert cov.abs().max().item() > 0 and torch.isfinite(cov).all()
    print(f"coverage train --init-from (depth_soft_osc={depth_soft_osc}, history_age={history_age}, flicker_weight={flicker_weight}): one CPU engine epoch, gradients and saved reload passed")


def test_configuration():
    assert FAST_DEFAULTS["coverage"] is False
    fresh = FastAccumulator((8, 12), (16, 24), coverage=True)
    start = fresh.stem.in_channels - fresh.n_state
    assert fresh.stem.weight[:, start-8:start].eq(0).all()
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-coverage", "--fast-state", "0"])
    assert model_kwargs(args)["coverage"]
    from train import make_model
    from tools.bench_fused import parser
    args = parser().parse_args(["--arch", "fast", "--fast-coverage", "--fast-state", "0"])
    assert make_model(args, (8, 12), (16, 24), "cpu").coverage
    for robust in ({}, dict(depth_soft=True, mv_dilate=True, depth_dilate=True, thin_lock=True)):
        m = make((8, 12), coverage=True, **robust)
        src = _cuda_source(m)
        assert "const float *prev_cov,float *next_cov" in src
        assert "next_cov[3*n+at]=luma" in src
        assert "keep0=cov0+K_COVERAGE*p" in src
        assert "-DK_COVERAGE=1" in FusedFast(m)._kernel_options()
    print("coverage flags, metadata and CUDA specialization passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(712)
    test_configuration()
    test_off()
    test_warm_start()
    test_evidence()
    test_parity()
    test_resolve_weights()
    test_training()
    if torch.cuda.is_available():
        try:
            import cupy
        except ImportError:
            print("SKIP: CUDA coverage parity requires CuPy")
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            test_parity("cuda")
    else:
        print("SKIP: CUDA coverage parity requires CUDA")
    print("test_coverage passed")
