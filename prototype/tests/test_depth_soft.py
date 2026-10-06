"""Depth mismatch as a trunk feature, deployment metadata, and optional CUDA parity."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, build_model, infer_config,
                            load_checkpoint, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator, FastState
from fsrmamba.fused import (FusedFast, _CUDA, _cuda_source, _signed_jitter,
                           materialize_state, ref_pack, run_trunk)
from legacy_fast_reference import ref_pack as legacy_pack, ref_resolve as legacy_resolve
from test_fused import close, frames


OPTIONS = [dict(), dict(mv_dilate=True), dict(accum=True), dict(nearest_sample=True),
           dict(accum=True, conf_consistent=True), dict(base_gate=True),
           dict(hist_filter="bicubic"), dict(accum=True, carry_raw=True),
           dict(accum=True, hist_residual=True), dict(accum=True, conf_motion=True),
           dict(accum=True, carry_raw=True, base_gate=True, nearest_sample=True,
                conf_consistent=True, hist_filter="bicubic", mv_dilate=True),
           dict(accum=True, hist_residual=True, base_gate=True, nearest_sample=True,
                conf_consistent=True, hist_filter="bicubic", mv_dilate=True)]


def make(render=(64, 96), **opts):
    model = FastAccumulator(render, tuple(2 * v for v in render), widths=(8, 16), depths=(1, 1),
                            n_state=0, depth_test=True, **opts).eval()
    with torch.no_grad():
        model.out.weight.normal_(0, .015)
        model.out.bias.normal_(0, .1)
        model.film[2].weight.normal_(0, .01)
        model.film[2].bias.normal_(0, .01)
    return model


def test_config():
    assert FAST_DEFAULTS["depth_soft"] is False
    with patch.dict(os.environ, FSRM_DEPTH_SOFT="1"):
        assert not make().depth_soft
    for soft in (False, True):
        model = make((8, 12), depth_soft=soft)
        cfg = infer_config(model.state_dict())
        assert cfg["depth_soft"] is False
        assert model.state_dict().keys() == make((8, 12)).state_dict().keys()
        cfg.update(depth_test=True, depth_soft=soft)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(model.state_dict(), path)
            save_sidecar(path, cfg)
            loaded, actual = load_checkpoint(path, (8, 12), (16, 24))
            assert actual == cfg and loaded.depth_soft == soft
            sidecar = Path(str(path) + ".json")
            old = json.loads(sidecar.read_text())
            del old["depth_soft"]
            sidecar.write_text(json.dumps(old))
            loaded, actual = load_checkpoint(path, (8, 12), (16, 24))
            assert loaded.depth_soft is False and actual["depth_soft"] is False
            sidecar.unlink()
            loaded, actual = load_checkpoint(path, (8, 12), (16, 24))
            assert loaded.depth_soft is False and actual["depth_soft"] is False
            loaded, actual = load_checkpoint(path, (8, 12), (16, 24),
                                             overrides=dict(depth_test=True, depth_soft=True))
            assert loaded.depth_soft is True and actual["depth_soft"] is True
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-depth-test", "--fast-depth-soft"])
    assert model_kwargs(args)["depth_soft"] is True
    for create in (lambda: FastAccumulator((8, 12), (16, 24), depth_soft=True),
                   lambda: build_model(dict(arch="fast", depth_soft=True), (8, 12), (16, 24))):
        try:
            create()
        except ValueError as exc:
            assert "depth_soft requires depth_test" in str(exc)
        else:
            raise AssertionError("accepted depth_soft without depth_test")
    plain = make()
    assert _cuda_source(plain) is _CUDA
    assert not any("DEPTH_SOFT" in flag for flag in FusedFast(plain)._kernel_options())
    for opts in ({}, dict(mv_dilate=True, depth_dilate=True, thin_lock=True)):
        old = _cuda_source(make(**opts))
        soft = make(depth_soft=True, **opts)
        source = _cuda_source(soft)
        assert "#if K_DEPTH_SOFT" in source and "reset_feature=1.0f" in source
        assert "put(out_x,base,8+3*p,phase,reset_feature,stride)" in source
        assert "float survive=1-rs;" in source and "reset[at]=__float2half_rn(rs)" in source
        assert old != source and "-DK_DEPTH_SOFT=1" in FusedFast(soft)._kernel_options()
    from tools.bench_fused import parser, profile_variants
    from train import make_model
    args = parser().parse_args(["--arch", "fast", "--fast-state", "0", "--fast-depth-test", "--fast-depth-soft"])
    assert make_model(args, (8, 12), (16, 24), "cpu").depth_soft is True
    assert all(not variant.fast_depth_soft or variant.fast_depth_test
               for _, variant in profile_variants(args))
    result = subprocess.run([sys.executable, "train.py", "--help"], check=True, capture_output=True, text=True)
    assert "--fast-depth-soft" in result.stdout
    print("depth_soft configuration: persistence, legacy defaults, validation, CUDA specialization passed")


@torch.no_grad()
def test_recurrence():
    worst = 0
    for render, opts, layout in itertools.product(((64, 96), (63, 95)), OPTIONS, ("nhwc", "nchw")):
        model = make(render, depth_soft=True, **opts)
        fused = FusedFast(model, layout=layout)
        a, b = model.init_state(), fused.init_state()
        for inputs in frames(render, "cpu", 6):
            packed = ref_pack(model, a, *inputs[:-1], _signed_jitter(model, inputs[-1]))
            expected, a = model(a, *inputs)
            actual, b = fused.step_reference(b, *inputs)
            worst = max(worst, (expected - actual).abs().max().item())
            close(expected, actual, 3e-5)
            close(a.color, b.color, 3e-5)
            if model.accum:
                close(a.conf, b.conf, 3e-5)
            assert torch.equal(a.depth, b.depth)
            assert torch.equal(model._last_reset, packed["reset"])
            assert a.frame_index == b.frame_index
    print(f"depth_soft reference parity: {4 * len(OPTIONS)} configurations, 6 frames, max RGB error {worst:.8g}")


@torch.no_grad()
def test_legacy_exact():
    for render in ((64, 96), (63, 95)):
        model = make(render, depth_soft=False, accum=True, carry_raw=True, base_gate=True,
                     nearest_sample=True, conf_consistent=True, hist_filter="bicubic")
        fused = FusedFast(model)
        a, b = model.init_state(), fused.init_state()
        for lr, mv, depth, jitter in frames(render, "cpu", 6):
            signed = _signed_jitter(model, jitter)
            before = legacy_pack(model, a, lr, mv, depth, signed)
            after = ref_pack(model, b, lr, mv, depth, signed)
            for key, value in before.items():
                if value is not None:
                    assert torch.equal(value, after[key]), key
            o = run_trunk(model, before["X"], signed)
            expected, color, conf = legacy_resolve(model, o, lr, before, signed)
            actual, b = fused.step_reference(b, lr, mv, depth, jitter)
            a = FastState(color, a.feat, depth[None, None], a.frame_index + 1, conf)
            assert torch.equal(expected, actual)
            assert torch.equal(a.color, b.color) and torch.equal(a.conf, b.conf)
    print("depth_soft off: bit-identical to frozen legacy pack/resolve, random model, 6 frames at both sizes")


@torch.no_grad()
def test_stipple():
    render = (16, 24)
    for soft, opts in itertools.product((False, True), ({}, dict(mv_dilate=True))):
        model = make(render, depth_soft=soft, accum=True, carry_raw=True, base_gate=True, **opts)
        state = model.init_state()
        rgb = torch.rand(*render, 3)
        mv = torch.zeros(*render, 2)
        mv[0, 0, 0] = -1
        for frame in range(6):
            depth = torch.full(render, .2)
            depth[5:10, 9:14] = .2 if frame % 2 else .8
            packed = ref_pack(model, state, rgb, mv, depth, (0, 0))
            feature = F.pixel_shuffle(packed["X"], 2)[:, 20:21, :render[0], :render[1]]
            assert feature[0, 0, 7, 11] == 1
            assert packed["reset"][0, 0, 7, 11] == (1 if frame == 0 or not soft else 0)
            if frame == 0:
                assert feature.eq(1).all() and packed["reset"].eq(1).all()
            if not opts:
                assert packed["reset"][0, 0, 0, 0] == 1
            if frame:
                conf = packed["conf_w"][0, :, 7, 11]
                assert conf.gt(0).all() if soft else conf.eq(0).all()
            _, state = model(state, rgb, mv, depth, (0, 0))
            assert torch.equal(model._last_reset, packed["reset"])
    print("Stippled depth: trunk reset feature remains 1, hard reset is 0 only with depth_soft after frame 0")


@torch.no_grad()
def test_cuda():
    if not torch.cuda.is_available():
        print("CUDA depth_soft tests skipped: CUDA unavailable")
        return
    try:
        import cupy
    except ImportError:
        print("CUDA depth_soft tests skipped: CuPy unavailable")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for render, opts, layout in itertools.product(((64, 96), (63, 95)), OPTIONS, ("nhwc", "nchw")):
        model = make(render, depth_soft=True, **opts).cuda()
        model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
        fused = FusedFast(model, layout=layout)
        a, b = model.init_state("cuda"), fused.init_state()
        for inputs in frames(render, "cuda", 6):
            with torch.autocast("cuda", dtype=torch.float16):
                expected, a = model(a, *inputs)
            actual, b = fused.step(b, *inputs, write_rgb=True)
            close(expected, actual, .008)
            close(a.color, b.color, .008)
            if model.accum:
                close(a.conf, b.conf, .08)
            assert torch.equal(model._last_reset, fused._reset)
        state = materialize_state(b)
        lr, mv, depth, jitter = inputs
        signed = _signed_jitter(model, jitter)
        packed = ref_pack(model, state, lr.half(), mv, depth, signed)
        source = fused._stage(state, lr, mv, depth, signed)
        with cupy.cuda.Device(fused._device.index), fused._stream_context():
            fused._pack(source, False)
        close(fused._X, packed["X"], .008)
        close(fused._hcl, packed["h_cl"], .003)
        assert torch.equal(fused._reset, packed["reset"])
        if model.accum:
            close(fused._cw, packed["conf_w"], .003)
    print("CUDA depth_soft parity passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(71)
    test_config()
    test_legacy_exact()
    test_stipple()
    test_recurrence()
    test_cuda()
    print("test_depth_soft passed")
