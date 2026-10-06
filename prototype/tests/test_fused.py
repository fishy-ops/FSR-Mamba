"""Independent pack/resolve and folded-head equivalence, plus optional CUDA parity."""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_latency import smooth_motion
from fsrmamba.fast import FastAccumulator
from fsrmamba.fused import (FusedFast, TrunkModule, _HostFrameCache, _fold_head, _phase_offsets, _phase_weights,
                           _pack_index, _sample, _sample_nearest, _shuffle_add_reference, _state_views,
                           _TensorRTTrunkModule, _signed_jitter,
                           materialize_state, ref_pack, ref_resolve, run_trunk, state_rgb)
from fsrmamba.resolve import phase_kernels, window_bounds


def model_for(render, widths, accum, hist, film, device="cpu", scale=2,
              depth_test=False, conf_motion=False, hist_residual=False,
              nearest_sample=False, conf_consistent=False, carry_raw=False, jitter_sign=1.0,
              base_gate=False):
    model = FastAccumulator(render, tuple(v * scale for v in render), widths=widths,
                            depths=tuple(1 for _ in widths), n_state=0, film=film,
                            hist_filter=hist, accum=accum, depth_test=depth_test,
                            conf_motion=conf_motion, hist_residual=hist_residual,
                            nearest_sample=nearest_sample, conf_consistent=conf_consistent,
                            carry_raw=carry_raw, jitter_sign=jitter_sign, base_gate=base_gate, device=device).eval()
    with torch.no_grad():
        model.out.weight.normal_(0, 0.015)
        if model.base_gate:
            q = model._base_gate_offset * 4
            model.out.bias[q:q + model.p * 4].uniform_(-1, 1)
        if model.carry_raw:
            model.out.bias[:3 * model.p * 4].uniform_(-0.2, 0.2)
            q = model._carry_beta_offset * 4
            model.out.bias[q:q + model.p * 4].uniform_(-2, 2)
        if model.hist_residual:
            model.hres_gain.fill_(-0.3502)
            model.out.bias[4 * model.p * 4:7 * model.p * 4].uniform_(-0.5, 0.5)
        if model.film is not None:
            model.film[2].weight.normal_(0, 0.03)
            model.film[2].bias.uniform_(-0.08, 0.08)
        if model.conf_motion:
            model.conf_m.fill_(-0.5002)
    return model


JITTERS = ((0.4, -0.4), (-0.4, 0.4), (0.1, 0.2), (0.25, -0.25), (-0.25, 0.25))


def recurrent_options():
    options = [(accum, depth_test, conf_motion, False, False)
               for accum, depth_test, conf_motion in itertools.product((False, True), repeat=3)
               if accum or not conf_motion]
    # Additional pairs cover both new options together and separately, including
    # nearest sampling without accumulation and both depth/motion settings.
    return options + [(False, True, False, True, False), (True, False, False, True, False),
                      (True, True, True, False, True), (True, False, True, True, True)]


def carry_options():
    # Each pair of filter, sampling, confidence and depth settings sees all four combinations.
    for a, b, c in itertools.product((False, True), repeat=3):
        yield dict(hist="bicubic" if a else "bilinear", nearest_sample=b,
                   conf_consistent=c, depth_test=a ^ b ^ c, conf_motion=a ^ b,
                   film=b ^ c, jitter_sign=-1.0 if a ^ c else 1.0)


def base_options():
    rows = ((False, False, False, False, "bilinear", 1, False, 1),
            (False, False, True, True, "bicubic", -1, True, 2),
            (False, True, False, True, "bilinear", -1, True, 3),
            (False, True, True, False, "bicubic", 1, False, 2),
            (True, True, False, False, "bilinear", -1, False, 3),
            (True, True, True, False, "bilinear", 1, True, 2),
            (True, True, False, False, "bicubic", 1, True, 2),
            (True, True, True, False, "bicubic", -1, False, 1))
    for carry, accum, nearest, hres, hist, sign, film, scale in rows:
        yield dict(carry_raw=carry, accum=accum, nearest_sample=nearest, hist_residual=hres,
                   hist=hist, jitter_sign=sign, film=film, scale=scale,
                   conf_consistent=accum and not nearest, conf_motion=accum and film,
                   depth_test=film, base_gate=True)


def assert_base_windows(model):
    windows = [phase_kernels(_signed_jitter(model, jitter), "cpu",
                            model._ph_x.cpu(), model._ph_y.cpu(),
                            model._tap_x.cpu(), model._tap_y.cpu())[-1] for jitter in JITTERS]
    assert {y for frame in windows for y, _ in frame} == {0, 1}
    assert {x for frame in windows for _, x in frame} == {0, 1}


def assert_jitter_offsets(model):
    if model.nearest_sample:
        offsets = [_phase_offsets(model, torch.tensor(jitter)) for jitter in JITTERS]
        ys = {ny for frame in offsets for ny, _ in frame}
        xs = {nx for frame in offsets for _, nx in frame}
        if model.scale > 1:
            assert any(ny != 0 for frame in offsets for ny, _ in frame)
            assert any(nx != 0 for frame in offsets for _, nx in frame)
            assert ys == xs == {-1, 0, 1}
        else:
            assert ys == xs == {0}


def frames(render, device, count):
    h, w = render
    motion = smooth_motion(render, render, device) * 2.0
    motion[..., 0] += 2.5 / w
    x = torch.arange(w, device=device)[None].expand(h, w)
    for i in range(count):
        depth = torch.where(x < w // 2 + i % 3 - 1, 0.2, 0.8)
        depth[0, 0] = 0.001
        yield (torch.rand(h, w, 3, device=device), motion * (1 + i * 0.07), depth,
               torch.tensor(JITTERS[i % len(JITTERS)], device=device))


def close(a, b, atol=2e-5):
    torch.testing.assert_close(a, b, rtol=0, atol=atol)


@torch.no_grad()
def test_cpu():
    worst = 0.0
    # All six valid (accum, depth_test, conf_motion) triples with each history
    # filter and residual setting; two shape/width/FiLM variants cover padding.
    options = recurrent_options()
    variants = (((12, 20), (16,), False), ((11, 18), (16, 32), True))
    for (render, widths, film), settings, hist, hist_residual in itertools.product(
            variants, options, ("bilinear", "bicubic"), (False, True)):
        accum, depth_test, conf_motion, nearest_sample, conf_consistent = settings
        model = model_for(render, widths, accum, hist, film,
                          depth_test=depth_test, conf_motion=conf_motion, hist_residual=hist_residual,
                          nearest_sample=nearest_sample, conf_consistent=conf_consistent)
        assert_jitter_offsets(model)
        fused = FusedFast(model)
        original, reference = model.init_state(), fused.init_state()
        depth_changed_reset = False
        for inputs in frames(render, "cpu", 5):
            expected, original = model(original, *inputs)
            actual, reference = fused.step_reference(reference, *inputs)
            worst = max(worst, (actual - expected).abs().max().item())
            close(actual, expected)
            close(reference.color, original.color)
            close(reference.depth, original.depth)
            close(reference.feat, original.feat)
            assert reference.frame_index == original.frame_index
            if accum:
                close(reference.conf, original.conf)
            else:
                assert reference.conf is None
            assert model._last_reset.sum() > 0
            if original.frame_index > 1:
                assert model._last_reset.sum() < render[0] * render[1]
                uv = model._uv_lr + inputs[1][None]
                uv_reset = (~((uv >= 0) & (uv < 1)).all(-1))[:, None]
                depth_changed_reset |= (model._last_reset.bool() != uv_reset).any().item()
        assert depth_changed_reset == depth_test
    print(f"CPU recurrent matrix ({len(variants) * len(options) * 4} configs, 5 frames): max RGB error {worst:.8g}")


@torch.no_grad()
def test_carry_raw_cpu():
    worst = 0.0
    for scale, settings in itertools.product((1, 2, 3), carry_options()):
        model = model_for((7, 9), (8, 16), True, carry_raw=True, scale=scale, **settings)
        fused = FusedFast(model)
        original, reference = model.init_state(), fused.init_state()
        distinct = False
        for inputs in frames(model.render_size, "cpu", 5):
            expected, original = model(original, *inputs)
            actual, reference = fused.step_reference(reference, *inputs)
            worst = max(worst, (actual - expected).abs().max().item())
            close(actual, expected)
            close(reference.color, original.color)
            close(reference.conf, original.conf)
            close(reference.depth, original.depth)
            assert reference.frame_index == original.frame_index
            assert state_rgb(reference) is actual
            saved = materialize_state(reference)
            close(state_rgb(saved), actual, 0)
            assert state_rgb(saved).data_ptr() != actual.data_ptr()
            distinct |= (actual - reference.color[0].permute(1, 2, 0).clamp(min=0)).abs().max() > 1e-3
        assert distinct
    # Isolate pack/resolve dtype order with exact, stationary reprojection.
    for dtype, settings in itertools.product((torch.float32, torch.float16), carry_options()):
        model = model_for((8, 8), (8,), True, carry_raw=True, **settings).to(dtype=dtype)
        model.box_slack.data = model.box_slack.float()
        if model.conf_motion:
            model.conf_m.data = model.conf_m.float()
        state = model.init_state()
        state.frame_index = 1
        state.color.uniform_(-0.5, 1.5)
        state.conf.uniform_(1, 4)
        lr, _, depth, jitter = next(frames(model.render_size, "cpu", 1))
        state.depth = depth[None, None].to(dtype)
        motion = torch.zeros(8, 8, 2)
        lr = lr.to(dtype)
        seen, heads = [], []
        stem_hook = model.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        head_hook = model.out.register_forward_hook(lambda module, args, out: heads.append(out.clone()))
        expected, new = model(state, lr, motion, depth, jitter)
        stem_hook.remove()
        head_hook.remove()
        signed = _signed_jitter(model, jitter)
        packed = ref_pack(model, state, lr, motion, depth, signed)
        close(packed["X"], seen[0], 0)
        close(packed["h_raw"], F.pixel_unshuffle(state.color.to(dtype), model.scale), 0)
        assert (packed["h_raw"] != packed["h_cl"]).any()
        rgb, color, conf = ref_resolve(model, heads[0], lr, packed, signed)
        close(rgb, expected, 0)
        close(color, new.color, 0)
        close(conf, new.conf, 0)
        changed = dict(packed, h_raw=packed["h_cl"])
        unrectified = ref_resolve(model, heads[0], lr, changed, signed)[1]
        assert (color - unrectified).abs().max() > 1e-3
    for dtype in (torch.float32, torch.float16):
        model = model_for((4, 4), (8,), True, "bilinear", False, carry_raw=True).to(dtype=dtype)
        state = model.init_state()
        state.frame_index = 1
        state.color[..., ::2].fill_(-2)
        state.color[..., 1::2].fill_(2)
        inputs = (torch.full((4, 4, 3), 0.4, dtype=dtype), torch.zeros(4, 4, 2),
                  torch.ones(4, 4), (0., 0.))
        packed = ref_pack(model, state, *inputs)
        innovation = F.pixel_shuffle(packed["X"], 2)[:, -model.p:]
        close(innovation[:, ::2], torch.full_like(innovation[:, ::2], -4), 0)
        close(innovation[:, 1::2], torch.full_like(innovation[:, 1::2], 4), 0)
        state.frame_index = 0
        packed = ref_pack(model, state, *inputs)
        assert not F.pixel_shuffle(packed["X"], 2)[:, -model.p:].any()
    print(f"CPU carry_raw matrix and exact fp32/fp16 pack/resolve: max RGB error {worst:.8g}")


@torch.no_grad()
def test_base_gate_windows():
    model = model_for((5, 7), (8,), False, "bilinear", False, base_gate=True)
    h, w = model.render_size
    p = model.p
    lr = torch.zeros(h, w, 3)
    lr[0, 0, 0], lr[2, 3, 1], lr[-1, -1, 2] = 1, 1, 1
    px = torch.zeros(1, model.n_px, h, w)
    px[:, model._base_gate_offset:model._base_gate_offset + p] = torch.linspace(-1, 1, p)[None, :, None, None]
    o = F.pixel_unshuffle(F.pad(px, (0, model._pad[1], 0, model._pad[0]), mode="replicate"), 2)
    packed = dict(h_cl=torch.zeros(1, 3 * p, h, w), reset=torch.ones(1, 1, h, w))
    taps = F.pad(lr.permute(2, 0, 1).reshape(3, 1, h, w), (2, 1, 2, 1), mode="replicate")
    deringed = False
    phase_bounds_differ = False
    for jitter in JITTERS:
        _, kernels, _, _, _, win = phase_kernels(jitter, "cpu", model._ph_x, model._ph_y,
                                                model._tap_x, model._tap_y)
        lanczos = F.conv2d(taps, kernels).unsqueeze(0)
        lo, hi = window_bounds(taps, win, h, w)
        clipped = torch.minimum(torch.maximum(lanczos, lo), hi)
        deringed |= (lanczos - clipped).abs().max().item() > 1e-3
        phase_bounds_differ |= (hi - hi[:, :, :1]).abs().max().item() > 0
        gate = px[:, model._base_gate_offset:model._base_gate_offset + p].sigmoid()[:, None]
        base = torch.lerp(clipped, lr.permute(2, 0, 1)[None, :, None], gate)
        expected = F.pixel_shuffle(base.reshape(1, 3 * p, h, w), model.scale)
        _, color, _ = ref_resolve(model, o, lr, packed, jitter)
        close(color, expected)
        close(color[..., [0, -1], :], expected[..., [0, -1], :])
        close(color[..., [0, -1]], expected[..., [0, -1]])
    assert deringed and phase_bounds_differ
    # Float32 rounding affects gated nearest samples only outside the carry path.
    jitter = (0.250000001, -0.250000001)
    offsets = []
    for carry in (False, True):
        model = model_for((5, 7), (8,), carry, "bilinear", False,
                          base_gate=True, carry_raw=carry, nearest_sample=True)
        cache = _HostFrameCache(model)
        expected_jitter = jitter if carry else torch.tensor(jitter)
        expected = torch.tensor(_phase_offsets(model, expected_jitter), dtype=torch.int32)
        close(cache.get(jitter)[3], expected, 0)
        offsets.append(expected)
    assert not torch.equal(*offsets)


@torch.no_grad()
def test_base_gate_cpu():
    worst = 0.0
    for settings in base_options():
        model = model_for((7, 9), (8, 16), **settings)
        assert_base_windows(model)
        assert_jitter_offsets(model)
        fused = FusedFast(model)
        original, reference = model.init_state(), fused.init_state()
        heads = []
        hook = model.out.register_forward_hook(lambda module, args, out: heads.append(out.clone()))
        for inputs in frames(model.render_size, "cpu", len(JITTERS)):
            expected, original = model(original, *inputs)
            actual, reference = fused.step_reference(reference, *inputs)
            close(actual, expected)
            close(reference.color, original.color)
            close(reference.depth, original.depth)
            if model.accum:
                close(reference.conf, original.conf)
            else:
                assert reference.conf is None
            # Include all four edges, where the phase window repeats border taps.
            close(actual[[0, -1]], expected[[0, -1]])
            close(actual[:, [0, -1]], expected[:, [0, -1]])
            worst = max(worst, (actual - expected).abs().max().item())
            px = F.pixel_shuffle(heads.pop(), 2)
            q = model._base_gate_offset
            gate = px[:, q:q + model.p].sigmoid()
            assert ((gate > 0.1) & (gate < 0.9)).all()
            if model.carry_raw:
                assert state_rgb(reference) is actual
                assert (actual - reference.color[0].permute(1, 2, 0).clamp(min=0)).abs().max() > 1e-3
        hook.remove()

        cache = _HostFrameCache(model, capacity=2)
        fused._host_frames = cache
        fused._fold_weight = torch.empty_like(model.out.weight, dtype=torch.float16)
        fused._fold_bias = torch.empty_like(model.out.bias, dtype=torch.float16)
        fused._wc = torch.empty(model.p, dtype=torch.float16)
        fused._offsets = torch.empty(model.p, 2, dtype=torch.int32)
        fused._base_kernels = torch.empty(model.p, 16)
        fused._base_windows = torch.empty(model.p, 2, dtype=torch.int32)
        pointers = (fused._base_kernels.data_ptr(), fused._base_windows.data_ptr())
        for jitter in JITTERS:
            signed = _signed_jitter(model, jitter)
            fused._update_frame(signed)
            _, k, _, _, _, win = phase_kernels(signed, "cpu", model._ph_x, model._ph_y,
                                               model._tap_x, model._tap_y)
            close(fused._base_kernels, k.reshape(model.p, 16), 0)
            close(fused._base_windows, torch.tensor(win, dtype=torch.int32), 0)
            assert pointers == (fused._base_kernels.data_ptr(), fused._base_windows.data_ptr())
            assert cache.get(signed) is cache.get(signed)

    # Isolate the base from trunk/folding and history sampling for both working dtypes.
    for dtype, settings in itertools.product((torch.float32, torch.float16), base_options()):
        model = model_for((8, 8), (8,), **settings).to(dtype=dtype)
        state = model.init_state()
        state.frame_index = 1
        state.color.uniform_(-0.5, 1.5)
        if model.accum:
            state.conf.uniform_(1, 4)
        for jitter in JITTERS:
            lr = torch.rand(8, 8, 3, dtype=dtype)
            motion, depth = torch.zeros(8, 8, 2), torch.ones(8, 8)
            hist = torch.cat((state.color, state.conf), 1) if model.accum else state.color
            hist = hist.to(dtype)
            heads, seen = [], []
            hook = model.out.register_forward_hook(lambda module, args, out: heads.append(out.clone()))
            stem_hook = model.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
            expected, new = model(state, lr, motion, depth, jitter, hist=hist)
            hook.remove()
            stem_hook.remove()
            signed = _signed_jitter(model, jitter)
            p = model.p
            cat = F.pixel_shuffle(seen[0], 2)
            reset = cat[:, 8 + 3 * p:9 + 3 * p]
            packed = dict(h_cl=cat[:, 3:3 + 3 * p], rng=cat[:, 3 + 3 * p:6 + 3 * p], reset=reset,
                          h_raw=F.pixel_unshuffle(hist[:, :3], model.scale),
                          conf_w=F.pixel_unshuffle(hist[:, 3:].clamp(min=0), model.scale) * (1 - reset)
                          if model.accum else None)
            rgb, color, conf = ref_resolve(model, heads[0], lr, packed, signed)
            close(rgb, expected)
            close(color, new.color)
            if model.accum:
                close(conf, new.conf)
    print(f"CPU base_gate matrix, borders and fp32/fp16 resolve: max RGB error {worst:.8g}")


@torch.no_grad()
def test_hist_residual_dtypes():
    # Capture forward's head and packed inputs to isolate resolve rounding.
    options = ((False, False, False), (False, True, False), (False, True, True),
               (True, False, False), (True, True, False), (True, False, True), (True, True, True))
    for dtype, (accum, nearest_sample, conf_consistent), scale in itertools.product(
            (torch.float32, torch.float16), options, (1, 2, 3)):
        model = model_for((7, 9), (8,), accum, "bilinear", False, scale=scale,
                          hist_residual=True, nearest_sample=nearest_sample,
                          conf_consistent=conf_consistent).to(dtype=dtype)
        model.hres_gain.data = torch.tensor(-0.3502)
        if conf_consistent and accum:
            logits = torch.tensor([-3., -2., -0.7501, 0.0003, 1.2345, 2., 3.], dtype=dtype)
            model.out.bias[3 * model.p * 4:4 * model.p * 4].copy_(
                logits.repeat((model.p * 4 + 6) // 7)[:model.p * 4])
            model.conf_max = 2.5
        state = model.init_state()
        state.frame_index = 1
        lr, motion, depth, jitter = next(frames(model.render_size, "cpu", 1))
        lr = lr.to(dtype)
        hist = torch.rand(1, 4 if accum else 3, *model.output_size, dtype=dtype)
        if accum:
            hist[:, 3:].mul_(4)
        seen, heads = [], []
        stem_hook = model.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        head_hook = model.out.register_forward_hook(lambda module, args, out: heads.append(out.clone()))
        expected, new = model(state, lr, motion, depth, jitter, hist=hist)
        stem_hook.remove()
        head_hook.remove()
        h, w = model.render_size
        p = model.p
        cat = F.pixel_shuffle(seen[0], 2)[..., :h, :w]
        reset = cat[:, 8 + 3 * p:9 + 3 * p]
        packed = dict(h_cl=cat[:, 3:3 + 3 * p], rng=cat[:, 3 + 3 * p:6 + 3 * p], reset=reset,
                      conf_w=F.pixel_unshuffle(hist[:, 3:].clamp(min=0), scale) * (1 - reset)
                      if accum else None)
        actual = ref_resolve(model, heads[0], lr, packed, jitter)
        close(actual[0], expected)
        close(actual[1], new.color)
        if accum:
            close(actual[2], new.conf)
            if conf_consistent:
                assert (new.conf == model.conf_max).any()
        else:
            assert actual[2] is None
        model.hist_residual = False
        _, uncorrected = model(state, lr, motion, depth, jitter, hist=hist)
        assert (new.color - uncorrected.color).abs().max() > 1e-3
    print("CPU fp32/fp16 resolve options and history residual: forward parity at scales 1, 2, 3")


@torch.no_grad()
def test_nearest_offsets():
    model = model_for((4, 6), (8,), False, "bilinear", False, nearest_sample=True)
    cache = _HostFrameCache(model, fold=False)
    ties = ((0.25, -0.25), (-0.25, 0.25))
    for jitter in ties:
        assert _phase_offsets(model, jitter) == [(0, 0)] * model.p
        close(cache.get(jitter)[3], torch.zeros(model.p, 2, dtype=torch.int32), 0)
    jitter = (0.25 + 1e-10, -0.25 - 1e-10)
    assert torch.equal(torch.tensor(jitter), torch.tensor(ties[0]))
    offsets = torch.tensor([[-1, 0], [-1, 1], [0, 0], [0, 1]], dtype=torch.int32)
    close(cache.get(jitter)[3], offsets, 0)
    assert cache.get(jitter)[3] is cache.get(jitter)[3]
    assert not cache.get(ties[0])[3].any()
    # The first frame selects current colour exactly, exposing edge clamping.
    model.out.weight.zero_()
    model.out.bias[:3 * model.p * 4].zero_()
    lr = torch.arange(4 * 6 * 3).reshape(4, 6, 3).float() / 100
    motion, depth = torch.zeros(4, 6, 2), torch.ones(4, 6)
    for dt, j in itertools.product((torch.float32, torch.float16), (*JITTERS, jitter, *ties)):
        model.to(dtype=dt)
        rgb, state = model(model.init_state(), lr.to(dt), motion, depth, j)
        packed = ref_pack(model, model.init_state(), lr.to(dt), motion, depth, j)
        o = torch.zeros(1, model.n_px * 4, *model._tr_size, dtype=dt)
        actual = ref_resolve(model, o, lr.to(dt), packed, j)
        close(actual[0], rgb, 0)
        close(actual[1], state.color, 0)
    print("CPU nearest sample: Python float ties, cache keys and replicate edges are exact")


@torch.no_grad()
def test_depth_and_conf_reference():
    # Power-of-two sizes retain exact .5 ties through grid normalisation.
    coords = torch.tensor([-3.0, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 10.0])
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack(((x + 0.5) / 8 * 2 - 1, (y + 0.5) / 8 * 2 - 1), -1)[None]
    close(((grid[0, ..., 0] + 1) * 8 - 1) / 2, x, 0)
    assert torch.equal(coords[3:10].round(), torch.tensor([0., 2., 2., 4., 4., 6., 6.]))
    for dtype in (torch.float32, torch.float16):
        src = torch.arange(64, dtype=dtype).reshape(1, 1, 8, 8)
        expected = F.grid_sample(src, grid.to(dtype), mode="nearest",
                                 padding_mode="border", align_corners=False)
        close(_sample_nearest(src, x, y), expected, 0)

    for dtype, prev_dtype, hist in itertools.product(
            (torch.float32, torch.float16), (torch.float32, torch.float16), ("bilinear", "bicubic")):
        model = model_for((8, 8), (8,), True, hist, False, depth_test=True, conf_motion=True)
        state = model.init_state()
        state.frame_index = 1
        state.conf.uniform_(1, 4)
        # Values around both rounded half thresholds exercise strict comparisons.
        values = torch.tensor([0.4498, 0.45, 0.4501, 0.5, 0.5498, 0.55, 0.5502, 0.8])
        state.depth = values.to(prev_dtype)[None, None, None].expand(1, 1, 8, 8)
        lr = torch.rand(8, 8, 3).to(dtype)
        depth = torch.full((8, 8), 0.5)
        motion = torch.zeros(8, 8, 2)
        motion[..., 0] = torch.tensor([0., 0.5, -0.5, 0.4999, 0., 0., 0., -0.5]) / 8
        uv = model._uv_lr + motion[None]
        prev_d = F.grid_sample(state.depth, (uv * 2 - 1).to(prev_dtype), mode="nearest",
                               padding_mode="border", align_corners=False)
        d = depth[None, None].to(dtype)
        expected_reset = (~((uv >= 0) & (uv < 1)).all(-1))[:, None]
        expected_reset |= (prev_d < d * 0.9) | (prev_d > d * 1.1)
        packed = ref_pack(model, state, lr, motion, depth, (0., 0.))
        close(packed["reset"], expected_reset.to(dtype), 0)
        model.conf_motion = False
        unscaled = ref_pack(model, state, lr, motion, depth, (0., 0.))["conf_w"]
        speed = torch.linalg.vector_norm(motion[None] * model._wh, dim=-1).clamp(max=16)[:, None].to(dtype)
        expected_conf = unscaled / (1.0 + model.conf_m.abs() * speed)
        close(packed["conf_w"], expected_conf, 0)
        assert (packed["conf_w"] < unscaled).any()
        cat = F.pixel_shuffle(packed["X"], 2)
        close(cat[:, -model.p:], torch.log1p(expected_conf), 0)

    for dtype, hist in itertools.product((torch.float32, torch.float16), ("bilinear", "bicubic")):
        model = model_for((8, 32), (8,), True, hist, False, conf_motion=True)
        state = model.init_state()
        state.frame_index = 1
        state.conf.fill_(3)
        lr = torch.rand(8, 32, 3).to(dtype)
        depth = torch.ones(8, 32)
        motion = torch.zeros(8, 32, 2)
        motion[..., 0] = torch.tensor([0., 0.5, 2.5, 15.9, 16., 20., -16., -20.])[:, None] / 32
        packed = ref_pack(model, state, lr, motion, depth, (0., 0.))
        speed = torch.linalg.vector_norm(motion[None] * model._wh, dim=-1).clamp(max=16)[:, None].to(dtype)
        expected_conf = (torch.full_like(packed["conf_w"], 3) * (1 - packed["reset"]))
        expected_conf = expected_conf / (1.0 + model.conf_m.abs() * speed)
        close(packed["conf_w"], expected_conf, 2e-5)
        assert (packed["conf_w"][..., 5, :12] > 0).all()
        close(packed["conf_w"][..., 4, :12], packed["conf_w"][..., 5, :12], 2e-5)
        close(F.pixel_shuffle(packed["X"], 2)[:, -model.p:], torch.log1p(packed["conf_w"]), 0)
    print("CPU nearest depth ties, dtype thresholds and motion confidence: exact parity")


@torch.no_grad()
def test_layout_and_film():
    # Capture forward's actual stem input rather than reconstructing its concat.
    for scale, widths, hist_residual in itertools.product((1, 2, 3), ((8,), (8, 16, 24)), (False, True)):
        model = model_for((7, 9), widths, True, "bicubic", True, scale=scale,
                          hist_residual=hist_residual)
        state = model.init_state()
        state.frame_index = 2
        state.color.uniform_(-0.2, 1.2)
        state.conf.uniform_(0, 3)
        inputs = next(frames(model.render_size, "cpu", 1))
        seen = []
        hook = model.stem.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        model(state, *inputs)
        hook.remove()
        packed = ref_pack(model, state, *inputs)
        close(packed["X"], seen[0])
        assert packed["X"].is_contiguous(memory_format=torch.channels_last)
        assert packed["h_cl"].is_contiguous()
        assert packed["conf_w"].is_contiguous()
        # Also verify the documented packed channel formula without the model.
        h, w = model.render_size
        hp, wp = h + model._pad[0], w + model._pad[1]
        X = packed["X"]
        cat = F.pixel_shuffle(X, 2)[..., :h, :w]
        expected = F.pixel_unshuffle(F.pad(cat, (0, wp-w, 0, hp-h), mode="replicate"), 2)
        close(X, expected, 0)
        # Emulate the pack kernel's phase stores into each physical layout.
        padded = F.pad(cat, (0, wp-w, 0, hp-h), mode="replicate")
        C, Ht, Wt = X.shape[1:]
        yp, xp, channel = torch.meshgrid(torch.arange(hp), torch.arange(wp),
                                        torch.arange(C // 4), indexing="ij")
        phase_channel = channel * 4 + (yp % 2) * 2 + xp % 2
        for layout in ("nhwc", "nchw"):
            memory = (X.permute(0, 2, 3, 1).contiguous() if layout == "nhwc"
                      else X.contiguous()).flatten()
            at = _pack_index(layout, yp // 2, xp // 2, phase_channel, Ht, Wt, C)
            emulated = torch.empty_like(memory)
            emulated[at] = padded[0, channel, yp, xp]
            close(emulated, memory, 0)
            assert at.flatten().unique().numel() == X.numel()
        x = torch.randn(1, widths[0], 4, 6).contiguous(memory_format=torch.channels_last)
        g, b = model.film(inputs[-1][None]).chunk(2, 1)
        expected = model.out(x * (1 + g[:, :, None, None]) + b[:, :, None, None])
        weight, bias = _fold_head(model, inputs[-1], x.dtype, x.device)
        close(F.conv2d(x, weight, bias), expected, 2e-6)
        o = run_trunk(model, X, inputs[-1])
        assert o.is_contiguous(memory_format=torch.channels_last)
        actual = ref_resolve(model, o, inputs[0], packed, inputs[-1])
        rgb, new = model(state, *inputs)
        close(actual[0], rgb)
        close(actual[1], new.color)
        close(actual[2], new.conf)


@torch.no_grad()
def test_interleaved_state():
    for accum, hist in itertools.product((False, True), ("bilinear", "bicubic")):
        model = model_for((7, 9), (8, 16), accum, hist, True)
        plain = model.init_state()
        storage = torch.zeros((*model.output_size, 4))
        views = _state_views(storage, plain.feat, plain.depth, 0, accum)
        for inputs in frames(model.render_size, "cpu", 3):
            plain_pack = ref_pack(model, plain, *inputs)
            view_pack = ref_pack(model, views, *inputs)
            for key in plain_pack:
                if plain_pack[key] is not None:
                    close(view_pack[key], plain_pack[key], 0)
            rgb, plain = model(plain, *inputs)
            views.color.copy_(plain.color)
            if accum:
                views.conf.copy_(plain.conf)
                assert views.conf.stride()[-2:] == (model.output_size[1] * 4, 4)
                assert views.conf.data_ptr() == storage.data_ptr() + 3 * storage.element_size()
            else:
                assert views.conf is None
                close(storage[..., 3], torch.zeros_like(storage[..., 3]), 0)
            views.frame_index = plain.frame_index
            assert views.color.data_ptr() == storage.data_ptr()
            assert views.color.stride()[1:] == (1, model.output_size[1] * 4, 4)
            close(state_rgb(views), rgb, 0)
            saved = materialize_state(views)
            assert saved.color.is_contiguous()
            assert saved.color.data_ptr() != views.color.data_ptr()
            close(saved.color, plain.color, 0)
            close(saved.feat, views.feat, 0)
            close(saved.depth, views.depth, 0)
            assert saved.frame_index == views.frame_index
            if accum:
                assert saved.conf.is_contiguous()
                close(saved.conf, plain.conf, 0)
        retained = saved.color.clone()
        storage.fill_(-1)
        close(saved.color, retained, 0)

    # Sampling channels together shares weights without changing reduction order.
    storage = torch.randn(5, 7, 4, dtype=torch.float16)
    src = storage.permute(2, 0, 1)[None].float()
    y, x = torch.meshgrid(torch.tensor([-2.1, -0.5, 0.0, 0.9, 4.1, 6.0]),
                          torch.tensor([-3.2, -0.1, 0.0, 0.3, 5.9, 7.2]), indexing="ij")
    for cubic in (False, True):
        together = _sample(src, x, y, cubic)
        separate = torch.cat([_sample(src[:, c:c+1], x, y, cubic) for c in range(4)], 1)
        close(together, separate, 0)
        close(together.half(), separate.half(), 0)


def out_of_place_trunk(model, X, jitter):
    x = F.relu(model.stem(X))
    skips = []
    for i, block in enumerate(model.enc):
        x = block(x)
        if i < len(model.down):
            skips.append(x)
            x = F.relu(model.down[i](x))
    for i in range(len(model.down) - 1, -1, -1):
        x = skips[i] + F.pixel_shuffle(model.up[i](x), 2)
    x = F.relu(model.fuse(x))
    if model.film is None:
        return model.out(x)
    weight, bias = _fold_head(model, jitter, x.dtype, x.device)
    return F.conv2d(x, weight, bias)


@torch.no_grad()
def test_inplace_trunk():
    for dtype, widths, depth, film, layout, carry_raw in itertools.product(
            (torch.float32, torch.float16), ((8,), (8, 16, 24)), (0, 2), (False, True),
            ("nhwc", "nchw"), (False, True)):
        memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
        # Exercise empty levels and multiple residuals at every level.
        model = FastAccumulator((7, 9), (14, 18), widths=widths,
                                depths=(depth,) * len(widths), n_state=0, film=film,
                                accum=True, carry_raw=carry_raw).eval().to(dtype=dtype, memory_format=memory_format)
        model.out.weight.normal_(0, 0.02)
        if film:
            model.film[1].inplace = True
            model.film[2].weight.normal_(0, 0.03)
            model.film[2].bias.uniform_(-0.08, 0.08)
        X = torch.randn(1, model.stem.in_channels, *model._tr_size, dtype=dtype).contiguous(
            memory_format=memory_format)
        before = X.clone()
        jitter = torch.tensor([0.125, -0.375], dtype=dtype)
        expected = out_of_place_trunk(model, X, jitter)
        module = TrunkModule(model)
        close(module(X, jitter), expected, 0)
        close(module(X, jitter), expected, 0)
        close(X, before, 0)
        # Exercise the captured trunk's buffer writes with the kernel specification.
        fused = FusedFast(model, fold="gpu", layout=layout)
        fused._net, fused._X = model, X
        if film:
            fused._fold_weight, fused._fold_bias = _fold_head(model, jitter, dtype, "cpu")
        else:
            fused._fold_weight, fused._fold_bias = model.out.weight, model.out.bias
        resident = TrunkModule(model, fused._fold_weight, fused._fold_bias)
        close(resident(X, jitter), expected, 0)
        if layout == "nhwc":
            fused._shuffle_add = _shuffle_add_reference
        for _ in range(2):
            fused._middle()
            close(fused._o, expected, 0)
            assert fused._o.is_contiguous(memory_format=memory_format)
            close(X, before, 0)
    print("Interleaved state and in-place fp32/fp16 trunk: exact CPU parity")


@torch.no_grad()
def test_tensorrt_trunk_module():
    for dtype, widths, film, mode in itertools.product(
            (torch.float32, torch.float16), ((8,), (8, 16, 24)), (False, True),
            ("plain", "hist_residual", "carry_raw")):
        model = model_for((7, 9), widths, True, "bilinear", film,
                          hist_residual=mode == "hist_residual", carry_raw=mode == "carry_raw").to(dtype=dtype)
        X = torch.randn(1, model.stem.in_channels, *model._tr_size, dtype=dtype)
        module = _TensorRTTrunkModule(model)
        before = X.clone()
        for jitter in JITTERS[:2]:
            head = (_fold_head(model, jitter, dtype, "cpu") if film else
                    (model.out.weight, model.out.bias))
            expected = run_trunk(model, X, jitter, head)
            actual = module(X, *head)
            close(actual, expected, 1e-6 if dtype == torch.float32 else 3e-3)
            assert actual.dtype == dtype and actual.is_contiguous()
            close(X, before, 0)
        # The head must stay a runtime input, independent of the model's head.
        weight, bias = torch.zeros_like(head[0]), torch.arange(head[1].numel(), dtype=dtype)
        close(module(X, weight, bias), bias[None, :, None, None].expand_as(expected), 0)
    print("CPU TensorRT export module: fp32/fp16 NCHW trunk and runtime matmul head parity")


@torch.no_grad()
def test_host_frames_and_shuffle():
    for film, accum, nearest_sample in itertools.product((False, True), repeat=3):
        model = model_for((7, 9), (8, 16, 24), accum, "bilinear", film, nearest_sample=nearest_sample)
        cache = _HostFrameCache(model, capacity=2)
        jitter = (0.123456789, -0.375)
        weight, bias, phase, offsets, kernels, windows = cache.get(jitter)
        assert kernels is windows is None
        if nearest_sample:
            close(offsets, torch.tensor(_phase_offsets(model, jitter), dtype=torch.int32), 0)
        else:
            assert offsets is None
        if film:
            expected_weight, expected_bias = _fold_head(model, jitter, torch.float32, "cpu")
        else:
            expected_weight, expected_bias = model.out.weight, model.out.bias
        close(weight, expected_weight.half(), 0)
        close(bias, expected_bias.half(), 0)
        if accum:
            close(phase, _phase_weights(model, jitter, torch.float32, "cpu").half().reshape(-1), 0)
        else:
            assert phase is None
        fused = FusedFast(model)
        fused._host_frames = cache
        fused._fold_weight, fused._fold_bias = torch.empty_like(weight), torch.empty_like(bias)
        fused._wc = torch.empty(model.p, dtype=torch.float16)
        fused._offsets = torch.empty(model.p, 2, dtype=torch.int32)
        buffers = (fused._fold_weight, fused._fold_bias, fused._wc, fused._offsets)
        pointers = tuple(t.data_ptr() for t in buffers)
        for key in ((0.25, 0.25), jitter, jitter):
            fused._update_frame(key)
        close(fused._fold_weight, weight, 0)
        close(fused._fold_bias, bias, 0)
        if accum:
            close(fused._wc, phase, 0)
        if nearest_sample:
            close(fused._offsets, offsets, 0)
        assert pointers == tuple(t.data_ptr() for t in buffers)
        x = torch.randn(1, 8, 4, 6).contiguous(memory_format=torch.channels_last)
        if film:
            g, b = model.film(torch.tensor(jitter)[None]).chunk(2, 1)
            expected = model.out(x * (1 + g[:, :, None, None]) + b[:, :, None, None])
        else:
            expected = model.out(x)
        close(F.conv2d(x.half(), weight, bias).float(), expected, 3e-3)
        # Nearest sampling preserves Python floats; the other constants use fp32.
        hit = cache.get(torch.tensor(jitter, dtype=torch.float64 if nearest_sample else torch.float32))
        assert all(a is b for a, b in zip(hit, (weight, bias, phase, offsets)))
        cache.get((0.25, 0.25))
        cache.get(jitter)
        cache.get((-0.25, -0.25))
        assert len(cache.values) == 2 and (0.25, 0.25) not in cache.values
        # Cached parameters must remain a snapshot even after eviction.
        model.out.weight.add_(1)
        model.out.bias.add_(1)
        if film:
            model.film[0].weight.add_(1)
        if accum:
            model.acc_sharp.add_(1)
            model._acc_px.add_(0.1)
        cache.get((0.0, 0.0))
        fresh = cache.get(jitter)
        for a, b in zip(fresh, (weight, bias, phase, offsets)):
            if a is not None:
                close(a, b, 0)

    for dtype, c, shape in itertools.product((torch.float32, torch.float16), (3, 8, 24),
                                            ((1, 1), (3, 5))):
        h, w = shape
        # Distinct phase biases explicitly cover the transposed-convolution pitfall.
        conv = torch.nn.Conv2d(7, 4 * c, 1).to(dtype=dtype, memory_format=torch.channels_last)
        conv.bias.copy_(torch.arange(4 * c, dtype=dtype) / (4 * c))
        up = conv(torch.randn(1, 7, h, w, dtype=dtype).contiguous(memory_format=torch.channels_last))
        skip = torch.randn(1, c, 2 * h, 2 * w, dtype=dtype).contiguous(memory_format=torch.channels_last)
        before = up.clone()
        expected = F.pixel_shuffle(up, 2).add_(skip)
        pointer = skip.data_ptr()
        actual = _shuffle_add_reference(up, skip)
        close(actual, expected, 0)
        close(up, before, 0)
        assert actual.data_ptr() == pointer and actual.is_contiguous(memory_format=torch.channels_last)
    print("Host frame constants and NHWC shuffle+skip indexing: exact CPU parity")


def test_validation():
    for kwargs in (dict(n_state=1), dict(learned_clamp=True), dict(resolve="lanczos"),
                   dict(detail_ch=4), dict(stem_kernel=3)):
        config = dict(n_state=0, widths=(8,), depths=(0,))
        config.update(kwargs)
        model = FastAccumulator((4, 6), (8, 12), **config)
        try:
            FusedFast(model)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported config accepted: {kwargs}")
    try:
        FusedFast(torch.nn.Identity())
    except ValueError:
        pass
    else:
        raise AssertionError("Non-fast model accepted")
    model = model_for((4, 6), (8,), False, "bilinear", False)
    for kwargs in (dict(fold="invalid"), dict(layout="invalid"), dict(trunk="invalid"),
                   dict(trunk="tensorrt")):
        try:
            FusedFast(model, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid runtime options accepted: {kwargs}")
    for layout, trunk in (("nhwc", "cudnn"), ("nchw", "cudnn"), ("nchw", "tensorrt")):
        fused = FusedFast(model, layout=layout, trunk=trunk)
        assert (fused.layout, fused.trunk) == (layout, trunk)
        carry = model_for((4, 6), (8,), True, "bilinear", False, carry_raw=True)
        assert FusedFast(carry, layout=layout, trunk=trunk).model is carry
    for modules in (dict(tensorrt=None),
                    dict(tensorrt=SimpleNamespace(__version__="11.0"), onnx=None),
                    dict(tensorrt=SimpleNamespace(__version__="9.0"), onnx=SimpleNamespace())):
        with patch.dict(sys.modules, modules):
            try:
                FusedFast(model, layout="nchw", trunk="tensorrt")._prepare_tensorrt()
            except RuntimeError as exc:
                assert "TensorRT >= 10" in str(exc)
            else:
                raise AssertionError("Unavailable TensorRT/export dependency accepted")
    try:
        _pack_index("invalid", 0, 0, 0, 1, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid index layout accepted")
    try:
        FusedFast(model).step(model.init_state(), *next(frames((4, 6), "cpu", 1)))
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU kernel launch accepted")


@torch.no_grad()
def test_cuda():
    if not torch.cuda.is_available():
        print("CUDA fused tests skipped: CUDA or CuPy unavailable.")
        return
    try:
        import cupy
    except ImportError:
        print("CUDA fused tests skipped: CUDA or CuPy unavailable.")
        return
    # Stream handoff and graph replay must also work away from the default stream.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Exact .5 depth coordinates must choose even texels in both axes.
        model = model_for((8, 8), (8,), True, "bilinear", False, "cuda",
                          depth_test=True, conf_motion=True)
        fused = FusedFast(model)
        lr = torch.rand(8, 8, 3, device="cuda")
        motion = torch.full((8, 8, 2), 0.5 / 8, device="cuda")
        depth = torch.full((8, 8), 0.2, device="cuda")
        inputs = (lr, motion, depth, (0., 0.))
        fused.step(model.init_state(), *inputs)
        for prev_dtype in (torch.float32, torch.float16):
            state = model.init_state()
            state.frame_index = 1
            state.conf.fill_(3)
            state.depth = state.depth.to(prev_dtype)
            state.depth.fill_(0.2)
            state.depth[..., 1::2, :] = 0.8
            state.depth[..., :, 1::2] = 0.8
            packed = ref_pack(model, state, lr.half(), *inputs[1:])
            source = fused._stage(state, *inputs)
            with cupy.cuda.Device(lr.device.index), fused._stream_context():
                fused._pack(source, False)
            close(fused._reset, packed["reset"], 0)
            assert not fused._reset[..., :7, :7].any()
            close(fused._X, packed["X"], 3e-3)
            close(fused._cw, packed["conf_w"], 3e-3)
        cases = (((540, 960), (32, 64), 2), ((355, 640), (32, 64), 2),
                 ((7, 9), (8, 16, 24), 1), ((7, 9), (8, 16, 24), 3))
        options = recurrent_options()
        # Pairwise FiLM/folding/residual variants cover every pair without doubling
        # the recurrent matrix; each option sees residuals on and off per filter.
        variants = ((False, "host", False), (True, "gpu", False),
                    (False, "gpu", True), (True, "host", True))
        for (render, widths, scale), settings, hist, (film, fold, hist_residual) in itertools.product(
                cases, options, ("bilinear", "bicubic"), variants):
            accum, depth_test, conf_motion, nearest_sample, conf_consistent = settings
            model = model_for(render, widths, accum, hist, film, "cuda", scale=scale,
                              depth_test=depth_test, conf_motion=conf_motion, hist_residual=hist_residual,
                              nearest_sample=nearest_sample, conf_consistent=conf_consistent)
            assert_jitter_offsets(model)
            model.to(memory_format=torch.channels_last)
            fused = FusedFast(model, fold=fold)
            original, actual_state = model.init_state(), fused.init_state()
            errors, color_errors, conf_errors = [], [], []
            pointers = set()
            depth_changed_reset = False
            for i, inputs in enumerate(frames(render, "cuda", 6)):
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, original = model(original, *inputs)
                write_rgb = i % 2 == 0
                if not write_rgb:
                    fused._rgb.fill_(-7)
                actual, actual_state = fused.step(actual_state, *inputs, write_rgb=write_rgb)
                close(actual_state.depth, original.depth, 0)
                close(fused._reset, model._last_reset, 0)
                if i > 0:
                    uv = model._uv_lr + inputs[1][None]
                    uv_reset = (~((uv >= 0) & (uv < 1)).all(-1))[:, None]
                    depth_changed_reset |= (model._last_reset.bool() != uv_reset).any().item()
                lazy = state_rgb(actual_state)
                if write_rgb:
                    close(lazy, actual, 0)
                else:
                    assert actual is None
                    assert (fused._rgb == -7).all()
                    actual = lazy
                err = (expected - actual).abs().max().item()
                errors.append(err)
                color_errors.append((original.color - actual_state.color).abs().max().item())
                if accum:
                    conf_errors.append((original.conf - actual_state.conf).abs().max().item())
                assert actual.dtype == torch.float16 and actual.is_contiguous()
                assert actual_state.color.shape == (1, 3, *model.output_size)
                assert actual_state.color.stride()[1:] == (1, model.output_size[1] * 4, 4)
                assert not actual_state.color.is_contiguous()
                materialized = materialize_state(actual_state)
                assert materialized.color.is_contiguous()
                close(materialized.color, actual_state.color, 0)
                if accum:
                    assert materialized.conf.is_contiguous()
                    close(materialized.conf, actual_state.conf, 0)
                else:
                    assert actual_state.conf is None
                    assert (fused._histories[1 - i % 2][..., 3] == 0).all()
                pointers.add(actual_state.color.data_ptr())
                assert actual_state.frame_index == original.frame_index
                # Without accumulation the history keeps weight ~0.9, so the trunk's own rounding
                # differences (cuDNN algorithm and layout; PyTorch against itself, channels_last
                # against NCHW with equal weights, drifts 4.9e-4 to 1.95e-3 over 12 frames at
                # 540x960) persist for several frames. Accumulating configs decay faster.
                limit = 3e-3 if accum else 4.5e-3
                assert err <= limit, (render, accum, depth_test, conf_motion, hist, film,
                                     hist_residual, nearest_sample, conf_consistent, errors)
            assert len(pointers) == 2
            assert depth_changed_reset == depth_test
            assert all(t is None or t.is_pinned() for t in fused._host_frames.get(inputs[-1]))
            print(f"CUDA {render} scale={scale} accum={accum} depth_test={depth_test} "
                  f"conf_motion={conf_motion} hist={hist} film={film} fold={fold} hist_residual={hist_residual} "
                  f"nearest_sample={nearest_sample} conf_consistent={conf_consistent}: "
                  f"max RGB={max(errors):.8g}, color={max(color_errors):.8g}, "
                  f"conf={max(conf_errors, default=0):.8g}")
            # Independent half reference checks locate kernel errors apart from folding.
            packed = ref_pack(model, original, inputs[0].half(), *inputs[1:])
            source = fused._stage(original, *inputs)
            with cupy.cuda.Device(actual.device.index), fused._stream_context():
                fused._pack(source, False)
            close(fused._X, packed["X"], 3e-3)
            close(fused._hcl, packed["h_cl"], 3e-3)
            close(fused._reset, packed["reset"], 0)
            if hist_residual:
                close(fused._rng, packed["rng"], 0)
                assert fused._net.hres_gain.dtype == torch.float32
            else:
                assert fused._rng is None
            if accum:
                close(fused._cw, packed["conf_w"], 1e-2)
            fused._graph.replay()
            expected_parts = ref_resolve(model, fused._o, inputs[0].half(),
                                        dict(h_cl=fused._hcl, conf_w=fused._cw, reset=fused._reset,
                                             rng=fused._rng), inputs[-1])
            with cupy.cuda.Device(actual.device.index), fused._stream_context():
                fused._resolve(1-source, write_rgb=True)
            close(fused._rgb, expected_parts[0], 2e-3)
            close(fused._colors[1-source], expected_parts[1], 2e-3)
            if accum:
                close(fused._confs[1-source].float(), expected_parts[2], 2e-3)
            # Direct kernel parity, including all phase biases and an existing skip.
            for c, h, w in ((8, 1, 1), (24, 3, 5)):
                up = torch.randn(1, 4*c, h, w, device="cuda", dtype=torch.float16).contiguous(
                    memory_format=torch.channels_last)
                skip = torch.randn(1, c, 2*h, 2*w, device="cuda", dtype=torch.float16).contiguous(
                    memory_format=torch.channels_last)
                expected_up = F.pixel_shuffle(up, 2).add_(skip)
                fused._shuffle_add(up, skip)
                close(skip, expected_up, 0)
            # After capture, report allocation stability without asserting it.
            work_pointers = [t.data_ptr() for level in fused._work for t in level]
            stable = torch.cuda.memory_allocated()
            for _ in range(3):
                _, actual_state = fused.step(actual_state, *inputs)
            print(f"  note: allocated bytes changed by {torch.cuda.memory_allocated() - stable} over 3 steady steps")
            assert work_pointers == [t.data_ptr() for level in fused._work for t in level]
    stream.synchronize()
    test_cuda_backends()
    test_cuda_carry_raw()
    test_cuda_base_gate()
    test_cuda_scalar_gains()


@torch.no_grad()
def test_cuda_backends():
    import cupy
    backends = [("cudnn", 3e-3)]
    try:
        import tensorrt
    except ImportError:
        print("CUDA TensorRT tests skipped: tensorrt unavailable.")
    else:
        backends.append(("tensorrt", 6e-3))
    cases = (((7, 9), (8, 16, 24), 3), ((35, 64), (16, 32), 1),
             ((540, 960), (32, 64), 2))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for (render, widths, scale), enhanced, (trunk, tolerance) in itertools.product(
                cases, (False, True), backends):
            model = model_for(render, widths, enhanced, "bicubic" if enhanced else "bilinear",
                              enhanced, "cuda", scale=scale, depth_test=enhanced,
                              conf_motion=enhanced, hist_residual=enhanced,
                              nearest_sample=enhanced, conf_consistent=enhanced)
            fold = "gpu" if enhanced and scale == 1 else "host"
            fused = FusedFast(model, fold=fold, layout="nchw", trunk=trunk)
            original, state = model.init_state(), fused.init_state()
            worst, pointers = 0.0, None
            for i, inputs in enumerate(frames(render, "cuda", 6)):
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, original = model(original, *inputs)
                actual, state = fused.step(state, *inputs, write_rgb=i % 2 == 0)
                if actual is None:
                    actual = state_rgb(state)
                worst = max(worst, (expected - actual).abs().max().item())
                close(actual, expected, tolerance)
                close(fused._reset, model._last_reset, 0)
                close(state.depth, original.depth, 0)
                assert state.frame_index == original.frame_index
                assert fused._X.is_contiguous() and fused._o.is_contiguous()
                buffers = (fused._X, fused._o, fused._fold_weight, fused._fold_bias)
                current = tuple(t.data_ptr() for t in buffers)
                if pointers is None:
                    pointers = current
                assert pointers == current
            print(f"CUDA layout=nchw trunk={trunk} {render} scale={scale} "
                  f"enhanced={enhanced} fold={fold}: max RGB error {worst:.8g}", flush=True)
            packed = ref_pack(model, original, inputs[0].half(), *inputs[1:])
            source = fused._stage(original, *inputs)
            with cupy.cuda.Device(fused._device.index), fused._stream_context():
                fused._pack(source, False)
                close(fused._X, packed["X"], 3e-3)
                fused._execute_trunk()
                expected_parts = ref_resolve(model, fused._o, inputs[0].half(),
                                            dict(h_cl=fused._hcl, conf_w=fused._cw,
                                                 reset=fused._reset, rng=fused._rng), inputs[-1])
                fused._resolve(1 - source, True)
            close(fused._rgb, expected_parts[0], 2e-3)
            close(fused._colors[1 - source], expected_parts[1], 2e-3)
            if enhanced:
                close(fused._confs[1 - source].float(), expected_parts[2], 2e-3)
    stream.synchronize()


@torch.no_grad()
def test_cuda_carry_raw():
    import cupy
    backends = [("nhwc", "cudnn", 3e-3), ("nchw", "cudnn", 3e-3)]
    try:
        import tensorrt
    except ImportError:
        print("CUDA carry_raw TensorRT tests skipped: tensorrt unavailable.")
    else:
        backends.append(("nchw", "tensorrt", 6e-3))
    cases = (((7, 9), (8, 16, 24), 1), ((35, 64), (16, 32), 2),
             ((7, 9), (8, 16, 24), 3))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for (render, widths, scale), settings, (layout, trunk, tolerance) in itertools.product(
                cases, carry_options(), backends):
            model = model_for(render, widths, True, device="cuda", scale=scale,
                              carry_raw=True, **settings)
            model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
            fold = "gpu" if settings["film"] else "host"
            fused = FusedFast(model, fold=fold, layout=layout, trunk=trunk)
            original, state = model.init_state(), fused.init_state()
            worst = [0.0, 0.0, 0.0]
            distinct = False
            for i, inputs in enumerate(frames(render, "cuda", 6)):
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, original = model(original, *inputs)
                if i:
                    fused._rgb.fill_(-7)
                actual, state = fused.step(state, *inputs, write_rgb=i % 2 == 0)
                assert actual is fused._rgb and state_rgb(state) is actual
                assert actual.dtype == torch.float16 and actual.is_contiguous()
                assert state.color.stride()[1:] == (1, model.output_size[1] * 4, 4)
                for k, (got, wanted) in enumerate(((actual, expected), (state.color, original.color),
                                                  (state.conf, original.conf))):
                    close(got.float(), wanted.float(), tolerance)
                    worst[k] = max(worst[k], (got.float() - wanted.float()).abs().max().item())
                close(state.depth, original.depth, 0)
                close(fused._reset, model._last_reset, 0)
                assert state.frame_index == original.frame_index
                distinct |= (actual - state.color[0].permute(1, 2, 0).clamp(min=0)).abs().max() > 1e-3
                saved = materialize_state(state)
                close(state_rgb(saved), actual, 0)
                assert state_rgb(saved).data_ptr() != actual.data_ptr()
            assert distinct
            signed = _signed_jitter(model, inputs[-1])
            packed = ref_pack(model, original, inputs[0].half(), *inputs[1:3], signed)
            source = fused._stage(original, *inputs[:3], signed)
            with cupy.cuda.Device(fused._device.index), fused._stream_context():
                fused._pack(source, False)
                close(fused._X, packed["X"], 3e-3)
                close(fused._hcl, packed["h_cl"], 3e-3)
                close(fused._hraw, packed["h_raw"], 3e-3)
                close(fused._cw, packed["conf_w"], 1e-2)
                close(fused._reset, packed["reset"], 0)
                fused._execute_trunk()
                expected_parts = ref_resolve(model, fused._o, inputs[0].half(),
                                            dict(h_cl=fused._hcl, h_raw=fused._hraw,
                                                 conf_w=fused._cw, reset=fused._reset), signed)
                fused._rgb.fill_(-7)
                fused._resolve(1 - source, False)
            close(fused._rgb, expected_parts[0], 2e-3)
            close(fused._colors[1 - source], expected_parts[1], 2e-3)
            close(fused._confs[1 - source].float(), expected_parts[2], 2e-3)
            print(f"CUDA carry_raw {render} scale={scale} layout={layout} trunk={trunk} "
                  f"{settings}: max RGB={worst[0]:.8g}, color={worst[1]:.8g}, conf={worst[2]:.8g}",
                  flush=True)
    stream.synchronize()


@torch.no_grad()
def test_cuda_base_gate():
    import cupy
    backends = [("nhwc", "cudnn", 3e-3), ("nchw", "cudnn", 3e-3)]
    try:
        import tensorrt
    except ImportError:
        print("CUDA base_gate TensorRT tests skipped: tensorrt unavailable.")
    else:
        backends.append(("nchw", "tensorrt", 6e-3))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for (i, settings), (layout, trunk, tolerance) in itertools.product(
                enumerate(base_options()), backends):
            model = model_for((7, 9), (8, 16), device="cuda", **settings)
            assert_base_windows(model)
            model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
            fused = FusedFast(model, fold="gpu" if i % 2 else "host", layout=layout, trunk=trunk)
            original, state = model.init_state(), fused.init_state()
            worst = [0.0, 0.0, 0.0]
            pointers = None
            for frame, inputs in enumerate(frames(model.render_size, "cuda", len(JITTERS))):
                with torch.autocast("cuda", dtype=torch.float16):
                    expected, original = model(original, *inputs)
                actual, state = fused.step(state, *inputs, write_rgb=frame % 2 == 0)
                if actual is None:
                    actual = state_rgb(state)
                for k, (got, wanted) in enumerate(((actual, expected), (state.color, original.color))):
                    close(got, wanted, tolerance)
                    worst[k] = max(worst[k], (got - wanted).abs().max().item())
                if model.accum:
                    close(state.conf, original.conf, tolerance)
                    worst[2] = max(worst[2], (state.conf - original.conf).abs().max().item())
                else:
                    assert state.conf is None
                close(state.depth, original.depth, 0)
                close(fused._reset, model._last_reset, 0)
                assert state.frame_index == original.frame_index
                if model.carry_raw:
                    assert state_rgb(state) is actual
                current = (fused._base_kernels.data_ptr(), fused._base_windows.data_ptr())
                if pointers is None:
                    pointers = current
                assert pointers == current
                signed = _signed_jitter(model, inputs[-1])
                assert all(t is None or t.is_pinned() for t in fused._host_frames.get(signed))
            # Check the resolve separately from head folding and both trunk backends.
            signed = _signed_jitter(model, inputs[-1])
            packed = ref_pack(model, original, inputs[0].half(), *inputs[1:3], signed)
            source = fused._stage(original, *inputs[:3], signed)
            with cupy.cuda.Device(fused._device.index), fused._stream_context():
                fused._pack(source, False)
                close(fused._X, packed["X"], 3e-3)
                close(fused._hcl, packed["h_cl"], 3e-3)
                close(fused._reset, packed["reset"], 0)
                if model.carry_raw:
                    close(fused._hraw, packed["h_raw"], 3e-3)
                fused._execute_trunk()
                expected_parts = ref_resolve(model, fused._o, inputs[0].half(),
                                            dict(h_cl=fused._hcl, h_raw=fused._hraw, conf_w=fused._cw,
                                                 reset=fused._reset, rng=fused._rng), signed)
                fused._resolve(1 - source, True)
            close(fused._rgb, expected_parts[0], 3e-3)
            close(fused._colors[1 - source], expected_parts[1], 3e-3)
            if model.accum:
                close(fused._confs[1 - source].float(), expected_parts[2], 3e-3)
            print(f"CUDA base_gate layout={layout} trunk={trunk} {settings}: "
                  f"max RGB={worst[0]:.8g}, color={worst[1]:.8g}, conf={worst[2]:.8g}", flush=True)
    stream.synchronize()


@torch.no_grad()
def test_cuda_scalar_gains():
    """box_slack, conf_m and hres_gain multiply half tensors as 0-dim fp32 CUDA parameters, which
    PyTorch rounds to half first. Non-default values must give the same rounding in the kernels."""
    import cupy
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for carry, layout in itertools.product((False, True), ("nhwc", "nchw")):
            model = model_for((35, 64), (16, 32), True, "bicubic", True, "cuda", depth_test=True,
                              conf_motion=True, hist_residual=not carry, carry_raw=carry,
                              nearest_sample=True, conf_consistent=True)
            with torch.no_grad():
                model.box_slack.fill_(0.2731)
                model.conf_m.fill_(0.4217)
                if model.hist_residual:
                    model.hres_gain.fill_(0.3217)
            model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
            fused = FusedFast(model, layout=layout)
            original, state = model.init_state(), fused.init_state()
            for inputs in frames((35, 64), "cuda", 4):
                with torch.autocast("cuda", dtype=torch.float16):
                    _, original = model(original, *inputs)
                _, state = fused.step(state, *inputs, write_rgb=True)
            packed = ref_pack(model, original, inputs[0].half(), *inputs[1:])
            source = fused._stage(original, *inputs)
            with cupy.cuda.Device(fused._device.index), fused._stream_context():
                fused._pack(source, False)
                fused._execute_trunk()
                parts = ref_resolve(model, fused._o, inputs[0].half(),
                                    dict(h_cl=fused._hcl, h_raw=fused._hraw, conf_w=fused._cw,
                                         reset=fused._reset, rng=fused._rng), inputs[-1])
                fused._resolve(1 - source, True)
            torch.cuda.synchronize()
            mismatch_x = (fused._X != packed["X"]).float().mean().item()
            mismatch_rgb = (fused._rgb != parts[0]).float().mean().item()
            print(f"CUDA scalar gains carry_raw={carry} layout={layout}: "
                  f"pack X mismatch {mismatch_x:.3g}, resolve display mismatch {mismatch_rgb:.3g}")
            assert mismatch_x < 1e-3 and mismatch_rgb < 1e-4, (mismatch_x, mismatch_rgb)
    stream.synchronize()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    test_validation()
    test_depth_and_conf_reference()
    test_cpu()
    test_carry_raw_cpu()
    test_base_gate_windows()
    test_base_gate_cpu()
    test_hist_residual_dtypes()
    test_nearest_offsets()
    test_layout_and_film()
    test_interleaved_state()
    test_inplace_trunk()
    test_tensorrt_trunk_module()
    test_host_frames_and_shuffle()
    test_cuda()
    print("test_fused passed")


if __name__ == "__main__":
    main()
