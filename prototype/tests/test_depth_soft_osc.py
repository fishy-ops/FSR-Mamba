"""Oscillation-gated depth resets, checkpoint compatibility, and inference parity."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, build_model, load_checkpoint,
                            load_model_state, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator
from fsrmamba.fused import FusedFast, _cuda_source, ref_pack, materialize_state
from test_coverage import make, test_training
from test_fused import close, frames


def model(render=(16, 24), **opts):
    return make(render, coverage=True, depth_soft_osc=True, **opts)


def test_config():
    assert FAST_DEFAULTS["depth_soft_osc"] is False
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-depth-test", "--fast-depth-soft",
                          "--fast-coverage", "--fast-depth-soft-osc", "--fast-state", "0"])
    assert model_kwargs(args)["depth_soft_osc"]
    from train import make_model
    assert make_model(args, (8, 12), (16, 24), "cpu").depth_soft_osc
    help_text = subprocess.run([sys.executable, "train.py", "--help"], check=True,
                               capture_output=True, text=True).stdout
    assert "--fast-depth-soft-osc" in help_text
    for missing in ("depth_test", "depth_soft", "coverage"):
        opts = dict(depth_test=True, depth_soft=True, coverage=True, depth_soft_osc=True)
        opts[missing] = False
        for create in (lambda: FastAccumulator((8, 12), (16, 24), **opts),
                       lambda: build_model(dict(arch="fast", **opts), (8, 12), (16, 24))):
            try:
                create()
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted missing {missing}")
    for robust in ({}, dict(mv_dilate=False), dict(depth_dilate=True, thin_lock=True)):
        m = model(**robust)
        source = _cuda_source(m)
        assert "#if K_DEPTH_SOFT_OSC" in source
        assert "float *next_cov,float soft_osc_threshold" in source
        assert "departure=prev_d>rh(mx*1.25f)" in source
        assert source.index("bool dither=osc_n>") < source.index("float survive=1-rs;")
        assert source.index("bool dither=osc_n>") < source.index("reset[at]=")
        assert "-DK_DEPTH_SOFT_OSC=1" in FusedFast(m)._kernel_options()
    off = make(coverage=True)
    assert "soft_osc_threshold" not in off.state_dict()
    assert "DEPTH_SOFT_OSC" not in _cuda_source(off)
    assert not any("DEPTH_SOFT_OSC" in f for f in FusedFast(off)._kernel_options())
    print("depth_soft_osc flags, prerequisites and CUDA specialization passed")


@torch.no_grad()
def test_warm_start():
    cfg = dict(arch="fast", widths=[8, 16], depths=[1, 1], n_state=0, accum=True,
               carry_raw=True, base_gate=True, nearest_sample=True, conf_consistent=True,
               hist_filter="bicubic", depth_test=True, depth_soft=True, coverage=True, mv_dilate=True)
    old = make((16, 24), coverage=True)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "old.pt"
        torch.save(old.state_dict(), path)
        save_sidecar(path, cfg)
        sidecar = Path(str(path) + ".json")
        legacy = json.loads(sidecar.read_text())
        del legacy["depth_soft_osc"]
        sidecar.write_text(json.dumps(legacy))
        off, actual = load_checkpoint(path, (16, 24), (32, 48))
        assert not off.depth_soft_osc and not actual["depth_soft_osc"]
        on, actual = load_checkpoint(path, (16, 24), (32, 48), overrides=dict(depth_soft_osc=True))
        assert on.depth_soft_osc and actual["depth_soft_osc"]
        assert on.soft_osc_threshold.item() == .5
        assert load_model_state(on, old.state_dict(), strict=False).missing_keys == ["soft_osc_threshold"]
        for key, value in old.state_dict().items():
            assert torch.equal(value, on.state_dict()[key]), key
        a, b = old.init_state(), off.init_state()
        for inputs in frames((16, 24), "cpu", 8):
            x, a = old(a, *inputs)
            y, b = off(b, *inputs)
            assert torch.equal(x, y) and torch.equal(a.color, b.color)
            assert torch.equal(a.conf, b.conf)
            for key in ("m1", "m2", "osc", "luma"):
                assert torch.equal(getattr(a, key), getattr(b, key))
        on.soft_osc_threshold.fill_(.731)
        torch.save(on.state_dict(), path)
        save_sidecar(path, actual)
        loaded, actual = load_checkpoint(path, (16, 24), (32, 48))
        assert loaded.soft_osc_threshold.item() == on.soft_osc_threshold.item()
        sidecar.unlink()
        inferred, actual = load_checkpoint(path, (16, 24), (32, 48))
        assert actual["depth_test"] and actual["depth_soft"] and actual["depth_soft_osc"]
        assert inferred.soft_osc_threshold.item() == on.soft_osc_threshold.item()
    checkpoint = Path(__file__).resolve().parents[2] / "ckpt/dither_cov_mac_last.pt"
    if checkpoint.exists():
        loaded, _ = load_checkpoint(checkpoint, (16, 24), (32, 48),
                                    overrides=dict(depth_soft_osc=True))
        state = loaded.init_state()
        for inputs in frames((16, 24), "cpu", 8):
            rgb, state = loaded(state, *inputs)
            assert torch.isfinite(rgb).all()
        print("Named dither_cov_mac_last checkpoint warm start: 8 CPU frames passed")
    else:
        print("SKIP: named checkpoint unavailable; legacy coverage checkpoint warm start passed")
    print("depth_soft_osc checkpoint round trip and legacy off defaults passed")


@torch.no_grad()
def test_parity(device="cpu"):
    worst = 0
    for render, layout, robust in itertools.product(((64, 96), (63, 95)), ("nhwc", "nchw"),
                                                   ({}, dict(depth_dilate=True, thin_lock=True))):
        m = model(render, **robust).to(device)
        f = FusedFast(m, layout=layout)
        a, b = m.init_state(device), f.init_state()
        for inputs in frames(render, device, 8):
            packed = ref_pack(m, a, *inputs) if device == "cpu" else None
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, a = m(a, *inputs)
                actual, b = f.step(b, *inputs, write_rgb=True)
            else:
                expected, a = m(a, *inputs)
                actual, b = f.step_reference(b, *inputs)
                assert torch.equal(m._last_reset, packed["reset"])
            worst = max(worst, (expected - actual).abs().max().item())
            close(expected, actual, .008 if device == "cuda" else 3e-5)
            close(a.color, b.color, .008 if device == "cuda" else 3e-5)
            close(a.conf, b.conf, .08 if device == "cuda" else 3e-5)
            assert torch.equal(a.depth, b.depth)
            for key in ("m1", "m2", "osc", "luma"):
                assert torch.equal(getattr(a, key), getattr(b, key))
        if device == "cuda":
            import cupy
            state = materialize_state(b)
            lr, mv, depth, jitter = inputs
            packed = ref_pack(m, state, lr.half(), mv, depth, jitter)
            source = f._stage(state, lr, mv, depth, jitter)
            with cupy.cuda.Device(f._device.index), f._stream_context():
                f._pack(source, False)
            assert torch.equal(f._reset, packed["reset"])
            close(f._X, packed["X"], .008)
    print(f"depth_soft_osc {device} parity: 8 configurations, 8 frames, max RGB error {worst:.8g}")


@torch.no_grad()
def test_synthetic(device="cpu"):
    render = (16, 40)
    for osc in (False, True):
        m = make(render, coverage=True, depth_soft_osc=osc).to(device)
        state = m.init_state(device)
        f = FusedFast(m)
        cuda_state = f.init_state()
        for i in range(8):
            left = 4 + 2 * i
            rgb = torch.full((*render, 3), .05, device=device)
            depth = torch.full(render, .2, device=device)
            mv = torch.zeros(*render, 2, device=device)
            rgb[4:12, left:left+8] = .9
            depth[4:12, left:left+8] = .8
            mv[4:12, left:left+8, 0] = -2 / render[1]
            # High oscillation must never exempt real foreground departure.
            if i:
                state.osc.fill_(4)
                packed = ref_pack(m, state, rgb, mv, depth, (0, 0))
                # The 3x3 depth envelope still sees foreground at left-1.
                # left-2 is the uncovered pixel outside that envelope.
                assert packed["reset"][0, 0, 7, left-2].item() == int(osc)
                feature = F.pixel_shuffle(packed["X"], 2)
                assert feature[0, 20, 7, left-2] == 1
            if device == "cuda":
                if i:
                    cuda_state.osc.fill_(4)
                _, cuda_state = f.step(cuda_state, rgb, mv, depth, (0, 0), write_rgb=True)
                if i:
                    assert f._reset[0, 0, 7, left-2].item() == int(osc)
            _, state = m(state, rgb, mv, depth, (0, 0))
            if i:
                assert m._last_reset[0, 0, 7, left-2].item() == int(osc)
    m = model(render).to(device)
    state = m.init_state(device)
    f = FusedFast(m)
    cuda_state = f.init_state()
    mv = torch.zeros(*render, 2, device=device)
    for i in range(16):
        rgb = torch.full((*render, 3), .2 if i % 2 else .8, device=device)
        # Warm the coverage moments, then alternate by 20% (below departure cutoff).
        depth = torch.full(render, .4 if i < 6 or i % 2 else .48, device=device)
        packed = ref_pack(m, state, rgb, mv, depth, (0, 0))
        if i >= 7:
            assert packed["reset"].eq(0).all()
            assert F.pixel_shuffle(packed["X"], 2)[:, 20].eq(1).all()
        if device == "cuda":
            _, cuda_state = f.step(cuda_state, rgb, mv, depth, (0, 0), write_rgb=True)
            if i >= 7:
                assert f._reset.eq(0).all()
        _, state = m(state, rgb, mv, depth, (0, 0))
        assert torch.equal(m._last_reset, packed["reset"])
    print(f"Synthetic {device}: moving-block departure always resets; warmed dither retains history")


def test_training_gate():
    m = model(mv_dilate=False).train()
    state = m.init_state()
    state.frame_index = 3
    state.depth.fill_(.4)
    state.osc.fill_(.01)  # Uniform RGB gives span=.02, osc_n=.5.
    rgb = torch.full((16, 24, 3), .5)
    mv = torch.zeros(16, 24, 2)
    depth = torch.full((16, 24), .48)
    m(state, rgb, mv, depth, (0, 0))
    close(m._last_reset, torch.full_like(m._last_reset, .5), 1e-6)
    m._last_reset.sum().backward()
    assert m.soft_osc_threshold.grad is not None and m.soft_osc_threshold.grad.item() > 0
    # Fused reference always uses the deployment hard step, even for a train() model.
    assert ref_pack(m, state, rgb, mv, depth, (0, 0))["reset"].eq(1).all()
    m.eval()
    for raw, effective in ((-1, .05), (10, 4), (.5, .5)):
        with torch.no_grad():
            m.soft_osc_threshold.fill_(raw)
            state.osc.fill_((.049 if effective == .05 else effective) * .02)
            m(state, rgb, mv, depth, (0, 0))
            assert m._last_reset.eq(1).all()  # Equality stays hard; test both clamps too.
            state.osc.fill_((effective + .001) * .02)
            m(state, rgb, mv, depth, (0, 0))
            assert m._last_reset.eq(int(effective == 4)).all()
    print("Training sigmoid gradient, inference strict step and threshold clamps passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(839)
    test_config()
    test_warm_start()
    test_training_gate()
    test_synthetic()
    test_parity()
    test_training(depth_soft_osc=True)
    if torch.cuda.is_available():
        try:
            import cupy
        except ImportError:
            print("SKIP: CUDA depth_soft_osc requires CuPy")
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            test_synthetic("cuda")
            test_parity("cuda")
    else:
        print("SKIP: CUDA depth_soft_osc unavailable")
    print("test_depth_soft_osc passed")
