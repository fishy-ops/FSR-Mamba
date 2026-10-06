"""Thin-feature robustness, legacy exactness and optional CUDA parity."""
import argparse
import itertools
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, build_model, infer_config,
                            load_checkpoint, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator, FastState, _nearest_depth, _thin_feature
from fsrmamba.fused import (FusedFast, _CUDA, _cuda_source, _signed_jitter,
                           ref_pack, run_trunk, materialize_state)
from legacy_fast_reference import ref_pack as legacy_pack, ref_resolve as legacy_resolve
from test_fused import frames, close


OPTIONS = [dict(mv_dilate=True), dict(depth_dilate=True), dict(thin_lock=True),
           dict(mv_dilate=True, depth_dilate=True, thin_lock=True)]


def make(render=(63, 95), **opts):
    model = FastAccumulator(render, tuple(2 * v for v in render), widths=(16, 32), depths=(1, 2),
                            n_state=0, accum=True, carry_raw=True, base_gate=True,
                            depth_test=True, nearest_sample=True, conf_consistent=True,
                            conf_motion=True, hist_filter="bicubic", jitter_sign=-1, **opts).eval()
    with torch.no_grad():
        model.out.weight.normal_(0, .008)
        model.film[2].weight.normal_(0, .01)
        model.box_slack.fill_(.2731)
        if model.thin_lock:
            model.thin_slack.fill_(1.0317)
    return model


@torch.no_grad()
def test_recurrence():
    worst = 0
    for render, opts, layout in itertools.product(((64, 96), (63, 95)), OPTIONS, ("nhwc", "nchw")):
        model = make(render, **opts)
        fused = FusedFast(model, layout=layout)
        a, b = model.init_state(), fused.init_state()
        for inputs in frames(render, "cpu", 6):
            packed = ref_pack(model, a, *inputs[:-1], _signed_jitter(model, inputs[-1]))
            expected, a = model(a, *inputs)
            actual, b = fused.step_reference(b, *inputs)
            worst = max(worst, (expected - actual).abs().max().item())
            close(expected, actual, 3e-5)
            close(a.color, b.color, 3e-5)
            close(a.conf, b.conf, 3e-5)
            assert torch.equal(a.depth, b.depth)
            assert torch.equal(model._last_reset, packed["reset"])
        # Explicit reset discards accumulated depth just like a fresh first frame.
        a.frame_index = b.frame_index = 0
        expected, a = model(a, *inputs)
        actual, b = fused.step_reference(b, *inputs)
        close(expected, actual, 3e-5)
        assert model._last_reset.eq(1).all()
    print(f"Thin-feature recurrence: 16 configurations, 6 frames + reset, max RGB error {worst:.8g}")


@torch.no_grad()
def test_legacy_exact():
    for render in ((64, 96), (63, 95)):
        torch.manual_seed(210)
        model = make(render)
        torch.manual_seed(210)
        explicit = make(render, mv_dilate=False, depth_dilate=False, thin_lock=False)
        assert model.state_dict().keys() == explicit.state_dict().keys()
        for key, value in model.state_dict().items():
            assert torch.equal(value, explicit.state_dict()[key]), key
        assert _cuda_source(model) is _CUDA
        options = FusedFast(model)._kernel_options()
        assert not any(flag in " ".join(options) for flag in ("DILATE", "THIN"))
        a, b = model.init_state(), explicit.init_state()
        for inputs in frames(render, "cpu", 6):
            lr, mv, depth, jitter = inputs
            signed = _signed_jitter(model, jitter)
            old = legacy_pack(model, a, lr, mv, depth, signed)
            packed = ref_pack(model, b, lr, mv, depth, signed)
            for key, value in old.items():
                if value is not None:
                    assert torch.equal(value, packed[key]), key
            o = run_trunk(model, old["X"], signed)
            before, color, conf = legacy_resolve(model, o, lr, old, signed)
            a = FastState(color, a.feat, depth[None, None], a.frame_index + 1, conf)
            after, b = FusedFast(explicit).step_reference(b, *inputs)
            assert torch.equal(before, after)
            assert torch.equal(a.color, b.color) and torch.equal(a.conf, b.conf)
            assert torch.equal(a.depth, b.depth)
    print("Legacy pack/resolve recomputation: exact equality, both render sizes, 6 frames")


def test_neighbourhoods():
    h, w = 63, 95
    depth = torch.ones(1, 1, h, w)
    motion = torch.arange(h * w * 2).reshape(1, h, w, 2).float()
    near, selected = _nearest_depth(depth, motion)
    y = (torch.arange(h) - 1).clamp(min=0)
    x = (torch.arange(w) - 1).clamp(min=0)
    assert torch.equal(selected, motion[:, y[:, None], x[None, :]])
    assert torch.equal(near, depth)
    depth[..., -1, -1] = 2
    near, selected = _nearest_depth(depth, motion)
    assert near[..., -2:, -2:].eq(2).all()
    assert selected[:, -2:, -2:].eq(motion[:, -1:, -1:]).all()
    for dtype in (torch.float32, torch.float16):
        rgb = torch.full((1, 3, h, w), .1, dtype=dtype)
        assert _thin_feature(rgb).eq(0).all()
        rgb[..., w // 2] = .9
        thin = _thin_feature(rgb)
        assert thin[..., w // 2].eq(1).all()
        assert thin[..., :w // 2].eq(0).all()
        assert _thin_feature(1 - rgb)[..., w // 2].eq(1).all()
        rgb[..., :w // 2] = .9
        assert _thin_feature(rgb).eq(0).all(), "step edge must not lock"
        # A soft ridge with contrast below half the range stays below the slack threshold.
        rgb.fill_(0)
        rgb[..., 20, 20] = .4
        rgb[..., 19, 19] = 1
        value = _thin_feature(rgb)[0, 0, 20, 20]
        assert 0 < value < .5


@torch.no_grad()
def test_thin_box():
    for dtype in (torch.float32, torch.float16):
        model = make((8, 12), thin_lock=True)
        state = model.init_state()
        state.color.fill_(2)
        rgb = torch.full((8, 12, 3), .125, dtype=dtype)
        rgb[:, 5] = .875
        motion = torch.zeros(8, 12, 2)
        depth = torch.ones(8, 12)
        for scalar in (1., -.25):
            model.thin_slack.fill_(scalar)
            model.box_slack.fill_(.25)
            packed = ref_pack(model, state, rgb, motion, depth, (0, 0))
            # Unpack the extra final channel, including the four 2x2 phases.
            feature = F.pixel_shuffle(packed["X"], 2)[:, -1:]
            assert feature[..., 5].eq(1).all()
            assert feature[..., 4].eq(0).all()
            hist = packed["h_cl"].reshape(1, 3, 4, 8, 12)
            assert hist[..., 5].eq(.875 + .75 * .25 * (1 + scalar)).all()
            assert hist[..., 4].eq(.875 + .75 * .25).all()


@torch.no_grad()
def test_moving_line():
    h, w = 16, 24
    # Exact motion on a continuous line already passes the old depth envelope.
    for opts in ({}, dict(mv_dilate=True, depth_dilate=True)):
        model = make((h, w), **opts)
        state = model.init_state()
        for frame in range(6):
            col = 5 + frame
            rgb = torch.full((h, w, 3), .05)
            rgb[:, col] = .95
            depth = torch.full((h, w), .1)
            depth[:, col] = .8
            mv = torch.zeros(h, w, 2)
            mv[:, col, 0] = -1 / w
            _, state = model(state, rgb, mv, depth, (0, 0))
            assert model._last_reset[0, 0, :, col].eq(1 if frame == 0 else 0).all()
    # A one-pixel coverage gap plus a changing far layer loses the old depth match.
    # Dilation transports the adjacent strand's motion and retains its previous depth.
    for opts in ({}, dict(mv_dilate=True, depth_dilate=True)):
        model = make((h, w), **opts)
        state = model.init_state()
        for frame in range(6):
            col = 5 + frame
            rgb = torch.full((h, w, 3), .05)
            rgb[:, col] = .95
            depth = torch.full((h, w), .02 * (frame + 1))
            depth[:, col] = .8
            depth[h // 2, col] = .02 * (frame + 1)
            mv = torch.zeros(h, w, 2)
            mv[:, col, 0] = -1 / w
            mv[h // 2, col] = 0
            _, state = model(state, rgb, mv, depth, (0, 0))
            if frame:
                assert model._last_reset[0, 0, h // 2, col] == (0 if opts else 1)
    print("Moving line: exact motion retains history in both; coverage gap resets only without dilation")


def test_config():
    for opts in OPTIONS:
        model = make((8, 12), **opts)
        cfg = infer_config(model.state_dict())
        for key in ("mv_dilate", "depth_dilate", "thin_lock"):
            assert cfg[key] == opts.get(key, False)
        rebuilt = build_model(cfg, (8, 12), (16, 24))
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(model.state_dict(), path)
            save_sidecar(path, cfg)
            loaded, actual = load_checkpoint(path, (8, 12), (16, 24))
            assert actual == cfg
            assert loaded.stem.in_channels == model.stem.in_channels
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-mv-dilate", "--fast-depth-dilate", "--fast-thin-lock"])
    assert all(model_kwargs(args)[key] for key in OPTIONS[-1])
    assert all(FAST_DEFAULTS[key] is False for key in OPTIONS[-1])
    # Thin input composes with detail, learned state, and the older clamp option too.
    model = FastAccumulator((9, 11), (18, 22), widths=(8, 16), depths=(1, 1),
                            n_state=2, detail_ch=4, learned_clamp=True, **OPTIONS[-1])
    assert model.thin_slack.item() == 1
    clone = build_model(infer_config(model.state_dict()), (9, 11), (18, 22))
    clone.load_state_dict(model.state_dict(), strict=True)
    out, _ = model(model.init_state(), torch.rand(9, 11, 3), torch.zeros(9, 11, 2), torch.ones(9, 11), (0, 0))
    out.mean().backward()
    assert model.thin_slack.grad is not None


@torch.no_grad()
def test_cuda():
    if not torch.cuda.is_available():
        print("CUDA thin-feature tests skipped: CUDA unavailable")
        return
    try:
        import cupy
    except ImportError:
        print("CUDA thin-feature tests skipped: CuPy unavailable")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for render, opts, layout in itertools.product(((64, 96), (63, 95)), OPTIONS, ("nhwc", "nchw")):
        model = make(render, **opts).cuda()
        memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
        model.to(memory_format=memory_format)
        fused = FusedFast(model, layout=layout)
        a, b = model.init_state("cuda"), model.init_state("cuda")
        for inputs in frames(render, "cuda", 6):
            with torch.autocast("cuda", dtype=torch.float16):
                expected, a = model(a, *inputs)
            actual, b = fused.step(b, *inputs, write_rgb=True)
            close(expected, actual, .008)
            close(a.color, b.color, .008)
            close(a.conf, b.conf, .08)
            assert torch.equal(a.depth, b.depth)
            assert torch.equal(model._last_reset, fused._reset)
        # Isolate pack from convolution error, including fp16 threshold/scalar rounding.
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
        assert torch.equal(fused._stored_depth, packed["depth"])
    print("CUDA thin-feature parity passed")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(71)
    test_config()
    test_neighbourhoods()
    test_thin_box()
    test_legacy_exact()
    test_recurrence()
    test_moving_line()
    test_cuda()
    print("test_thin_features passed")
