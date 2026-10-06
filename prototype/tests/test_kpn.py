"""KPN geometry, recurrence, training and checkpoint contracts on CPU."""

import argparse
from dataclasses import fields
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import (add_arch_args, build_model, infer_config, load_checkpoint,
                             load_sidecar, model_kwargs, save_sidecar)
from fsrmamba.kpn_unet import KPNAccumulator, DEFAULT_WIDTHS, LITE_WIDTHS, catmull_sample
from fsrmamba.synth import halton_jitter


def frame(size, i=0):
    h, w = size
    return (torch.rand(h, w, 3) * .8 + .05, torch.zeros(h, w, 2),
            torch.full((h, w), .5), halton_jitter(i))


def test_forward():
    for lite in (False, True):
        for size in ((64, 96), (63, 95)):
            m = KPNAccumulator(size, tuple(2 * v for v in size), lite=lite).eval()
            frames = [frame(size, i) for i in range(6)]
            outputs = []
            with torch.no_grad():
                for repeat in range(2):
                    state = m.init_state()
                    for i, f in enumerate(frames):
                        out, state = m(state, *f)
                        assert out.shape == (*m.output_size, 3)
                        assert torch.isfinite(out).all() and out.min() >= 0 and out.max() < 1
                        assert state.age.shape == (1, 1, *size)
                        assert torch.equal(state.age, torch.full_like(state.age, i))
                        assert state.frame_index == i + 1
                        if repeat:
                            assert torch.equal(out, outputs[i])
                        else:
                            outputs.append(out)
                f = frames[0]
                mv = f[1].repeat_interleave(2, 0).repeat_interleave(2, 1)
                depth = f[2].repeat_interleave(2, 0).repeat_interleave(2, 1)
                out, _ = m(m.init_state(), f[0], mv, depth, f[3])
                assert torch.equal(out, outputs[0])
    print("KPN both presets: deterministic six-frame forward at 64x96 and 63x95 passed")


def test_filter():
    m = KPNAccumulator((16, 24), (32, 48), residual=False)
    lr = torch.rand(1, 3, 16, 24) * .8 + .05
    history = torch.rand(1, 3, 32, 48)
    params = torch.zeros(1, 7, 32, 48)
    one, zero = torch.ones_like(params[:, :1]), torch.zeros_like(params[:, :1])
    for jitter in ((.1, -.2), (-.35, .4)):
        signed = torch.tensor(jitter)
        _, centre = m._kernel(params, signed)
        xy = m._sample_xy + signed
        weights = m.gaussian_weights(xy, centre, zero, params[:, :2] + 1e-4)
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(weights.sum(1), one[:, 0], atol=1e-6, rtol=0)
        out = m.resolve(lr, history, one, weights, centre, zero, 0., zero.bool())
        ix, iy = centre[0, ..., 0].long().clamp(0, 23), centre[0, ..., 1].long().clamp(0, 15)
        torch.testing.assert_close(out, lr[0, :, iy, ix][None], atol=1e-6, rtol=0)
    # The deployed sigma clamp is separate from the delta-limit primitive.
    extreme = params.clone()
    extreme[:, :2] = -100
    minimum = params.clone()
    minimum[:, :2] = math.log(.3)
    assert torch.equal(m._kernel(extreme, signed)[0], m._kernel(minimum, signed)[0])
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
    dither = ((xx + yy) % 2).float()[None, None].expand(1, 3, -1, -1) * (1 - 1e-6)
    params[:, :2] = math.log(2.5)
    weights, centre = m._kernel(params, torch.zeros(2))
    out = m.resolve(dither, history, one, weights, centre, zero, 0., zero.bool())
    box = F.avg_pool2d(dither, 5, 1, 2).repeat_interleave(2, 2).repeat_interleave(2, 3)
    error = (out[..., 6:-6, 6:-6] - box[..., 6:-6, 6:-6]).abs().max().item()
    assert error < 1e-2, error
    assert (out[..., 6:-6, 6:-6] - .5).abs().max() < .015
    # With alpha=0, out-of-range history must be rectified, even without reset.
    constant = torch.full_like(lr, .3)
    rectified = m.resolve(constant, history, zero, weights, centre, one, 0., zero.bool())
    torch.testing.assert_close(rectified, torch.full_like(rectified, .3), atol=1e-6, rtol=0)
    reset = m.resolve(lr, history, zero, weights, centre, one, 0., one.bool())
    current = m.resolve(lr, history, one, weights, centre, one, 0., zero.bool())
    assert torch.equal(reset, current)
    # At the border only real texels contribute, using their actual distances.
    params[:, :2] = math.log(.8)
    weights, centre = m._kernel(params, torch.tensor([.4, -.3]))
    out = m.resolve(lr, history, one, weights, centre, zero, 0., zero.bool())
    x, y = 0, 0
    cx, cy = (int(v) for v in centre[0, y, x])
    numerator, denominator = torch.zeros(3), 0.
    for iy in range(max(0, cy - 2), min(16, cy + 3)):
        for ix in range(max(0, cx - 2), min(24, cx + 3)):
            dx, dy = ix + .5 - .4 - (x + .5) / 2, iy + .5 + .3 - (y + .5) / 2
            weight = math.exp(-.5 * (dx * dx + dy * dy) / .8 ** 2)
            numerator += lr[0, :, iy, ix] * weight
            denominator += weight
    torch.testing.assert_close(out[0, :, y, x], numerator / denominator, atol=1e-6, rtol=0)
    # Sign -1 means actual positions are texel centre + capture jitter.
    opposite = KPNAccumulator((16, 24), (32, 48), residual=False, jitter_sign=-1)
    opposite.load_state_dict(dict(m.state_dict(), _jitter_sign=torch.tensor(-1.)))
    f = frame((16, 24))
    with torch.no_grad():
        a, _ = m(m.init_state(), *f[:3], (.2, -.3))
        b, _ = opposite(opposite.init_state(), *f[:3], (-.2, .3))
    assert torch.equal(a, b)
    print(f"KPN filter: delta, sigma clamp, sign, rectification/reset passed; dither vs box max error {error:.6g}")


@torch.no_grad()
def test_proximity():
    h, w = 8, 12
    scene = torch.rand(h, w, 3)*.8+.05
    motion, depth = torch.zeros(h, w, 2), torch.full((h, w), .5)
    jitters = ((.25, .25), (-.25, -.25), (.25, -.25), (-.25, .25), (0., 0.))
    for sign in (-1., 1.):
        for logit in (-.4, 2.):
            m = KPNAccumulator((h, w), (2*h, 2*w), widths=(3,)*6, lite=True,
                               residual=False, jitter_sign=sign, sigma_min=.05,
                               proximity=.35, proximity_gain=2.).eval()
            m.trunk.head.weight.zero_()
            m.trunk.head.bias.zero_()
            m.trunk.head.bias[:8] = -100
            m.trunk.head.bias[12:16] = logit
            state = m.init_state()
            near_count = far_count = 0
            for i, jitter in enumerate(jitters*2):
                _, state = m(state, scene, motion, depth, jitter)
                if i == 0:
                    assert m._last_alpha.eq(1).all()
                    continue
                signed = torch.tensor(jitter)*sign
                xy = m._sample_xy+signed
                centre = torch.floor(xy+.5).clamp(min=0)
                centre = torch.minimum(centre, torch.tensor([w-1, h-1]))
                r2 = ((centre-xy)*2).square().sum(-1)[:, None]
                bound = m.proximity_gain*torch.exp(-r2/(2*m.proximity**2))
                base = torch.tensor(logit).sigmoid()
                expected = (logit+math.log(m.proximity_gain)-r2/(2*m.proximity**2)).sigmoid()
                old = base*bound/(base*bound+(1-base)).clamp(min=1e-6)
                assert m._last_reset.eq(0).all()
                torch.testing.assert_close(m._last_alpha, expected, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(m._last_alpha, old, atol=1e-6, rtol=0)
                far, near = r2 >= 1., r2 < 1e-8
                assert (m._last_alpha[far] < base).all()   # far samples barely update the pixel
                g = m.proximity_gain
                torch.testing.assert_close(m._last_alpha[near],
                                           torch.full_like(m._last_alpha[near], float(base*g/(base*g+1-base))),
                                           atol=1e-6, rtol=0)
                far_count += int(far.sum())
                near_count += int(near.sum())
            assert far_count and near_count
            # The non-default lower bound changes the deployed kernel, too.
            params = torch.zeros(1, 7, 2*h, 2*w)
            params[:, :2] = -100
            minimum = params.clone()
            minimum[:, :2] = math.log(.05)
            assert torch.equal(m._kernel(params, signed)[0], m._kernel(minimum, signed)[0])
    print("KPN static scene/cycling jitter: proximity far bound, near alpha, saturation, reset and .05 sigma floor passed")


@torch.no_grad()
def test_proximity_extremes():
    h, w = 8, 12
    scene = torch.rand(h, w, 3)*.8+.05
    motion, depth = torch.zeros(h, w, 2), torch.full((h, w), .5)
    for proximity in (.35, .001, 0.):
        for logit in (100., -100.):
            m = KPNAccumulator((h, w), (2*h, 2*w), widths=(3,)*6, lite=True,
                               residual=False, proximity=proximity, proximity_gain=2.).eval()
            m.trunk.head.weight.zero_()
            m.trunk.head.bias.zero_()
            m.trunk.head.bias[12:16] = logit
            out, state = m(m.init_state(), scene, motion, depth, (0., 0.))
            assert m._last_reset.eq(1).all() and m._last_alpha.eq(1).all()
            assert torch.isfinite(out).all()
            for jitter in ((-.5, -.5), (-.25, -.25), (.5, .5), (.25, .25)):
                out, state = m(state, scene, motion, depth, jitter)
                assert m._last_reset.eq(0).all()
                assert torch.isfinite(out).all() and torch.isfinite(m._last_alpha).all()
                assert ((m._last_alpha >= 0) & (m._last_alpha <= 1)).all()
                if proximity == .35 and logit == 100. and jitter == (-.5, -.5):
                    # The clamped sample at (0, 0) is 1.5 output pixels away on each axis.
                    assert abs(float(m._last_alpha[0, 0, 0, 0])-1) <= 1e-6
                if logit == -100. and jitter == (-.25, -.25):
                    # Output (1, 1) lies exactly on the jittered sample at (0, 0).
                    assert float(m._last_alpha[0, 0, 1, 1]) <= 1e-6
                if proximity == 0.:
                    torch.testing.assert_close(m._last_alpha,
                                               torch.full_like(m._last_alpha, float(torch.tensor(logit).sigmoid())),
                                               atol=0, rtol=0)
    print("KPN proximity: saturated corner/near logits, .001 sigma finiteness, resets and disabled identity passed")


def test_reset_and_pack():
    m = KPNAccumulator((16, 24), (32, 48), mv_dilate=False).eval()
    f = frame(m.render_size)
    packed = []
    hook = m.trunk.register_forward_pre_hook(lambda module, args: packed.append(args[0].detach()))
    with torch.no_grad():
        _, state = m(m.init_state(), *f)
        state.age.fill_(32)
        _, aged = m(state, *f)
        assert aged.age.min() == 32 and aged.age.max() == 32
        assert packed[-1].shape == (1, 16, 16, 32)
        torch.testing.assert_close(packed[-1][:, 13, :, :24], torch.full((1, 16, 24), math.log2(33) / 5))
        detached = aged.detach()
        assert all(not getattr(detached, v.name).requires_grad for v in fields(detached)
                   if torch.is_tensor(getattr(detached, v.name)))
        fresh, _ = m(m.init_state(), *f)
        state.frame_index = 0
        reset, _ = m(state, *f)
        assert torch.equal(fresh, reset)
        assert m._last_alpha.eq(1).all()
        state.frame_index = 4
        state.depth.fill_(.1)
        reset, reset_state = m(state, *f)
        assert reset_state.age.eq(0).all() and m._last_reset.eq(1).all()
        assert torch.equal(fresh, reset)
        partial_mv = torch.zeros_like(f[1])
        partial_mv[:, :12, 0] = -2
        _, partial = m(aged, f[0], partial_mv, f[2], f[3])
        assert partial.age[..., :10].eq(0).all() and partial.age[..., 14:].eq(32).all()
        assert m._last_alpha[..., :20].eq(1).all()
        # A phase-varying history retains four independent lumas, with their mean RGB.
        hist = torch.zeros_like(state.color)
        for i, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            hist[..., y::2, x::2] = .1 * (i + 1)
        state.depth.fill_(.5)
        _, _ = m(state, *f, hist=hist)
        x = packed[-1][..., :24]
        torch.testing.assert_close(x[:, 3], torch.full_like(x[:, 3], .25))
        for i in range(4):
            torch.testing.assert_close(x[:, 6 + i], torch.full_like(x[:, 6 + i], .1 * (i + 1)))
        # Cached history obeys the same API as internal bicubic reprojection.
        history = m.reproject(aged, torch.zeros(1, 32, 48, 2))
        a, _ = m(aged, *f)
        b, _ = m(aged, *f, hist=history)
        assert torch.equal(a, b)
    hook.remove()
    soft = KPNAccumulator((16, 24), (32, 48), depth_soft=True)
    state = soft.init_state()
    state.frame_index = 3
    state.depth.fill_(.1)
    state.age.fill_(20)
    with torch.no_grad():
        _, state = soft(state, *f)
        assert soft._last_reset.eq(0).all() and state.age.eq(0).all()
        _, state = soft(state, f[0], torch.full_like(f[1], 2), *f[2:])
        assert soft._last_reset.eq(1).all() and soft._last_alpha.eq(1).all()
    # Nearest-depth MV dilation changes the geometry used for rejection.
    dilated = KPNAccumulator((16, 24), (32, 48))
    motion = torch.zeros_like(f[1])
    depth = f[2].clone()
    depth[8, 12], motion[8, 12, 0] = .9, 2
    state = dilated.init_state()
    state.frame_index = 1
    with torch.no_grad():
        dilated(state, f[0], motion, depth, f[3])
    assert dilated._last_reset[..., 14:20, 22:28].eq(1).all()
    print("KPN phase pack, cached history, resets, age, depth soft and nearest-depth MV dilation passed")


def test_gradients_and_cost():
    for lite in (False, True):
        for residual in (True, False):
            m = KPNAccumulator((32, 48), (64, 96), lite=lite, residual=residual)
            state, loss = m.init_state(), 0
            for i in range(3):
                out, state = m(state, *frame(m.render_size, i))
                loss = loss + (out - torch.rand_like(out)).square().mean()
            loss.backward()
            for name, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                assert p.grad.abs().sum() > 0, name
            # Independent shape hooks verify the analytic conv MAC accounting.
            measured = []
            hooks = [layer.register_forward_hook(lambda layer, args, out: measured.append(
                out.shape[-2] * out.shape[-1] * layer.weight.numel()))
                for layer in m.trunk.modules() if isinstance(layer, torch.nn.Conv2d)]
            with torch.no_grad():
                m.trunk(torch.zeros(1, 16, *m.padded_size))
            for hook in hooks:
                hook.remove()
            assert sum(measured) == m.cost()["trunk_macs"]
    print("KPN both presets/residual modes: finite nonzero parameter gradients and conv MAC accounting passed")


def test_config(directory):
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    for flags in ([], ["--kpn-lite"], ["--kpn-trunk-stride", "2", "--kpn-taps", "3",
                                              "--kpn-history-filter", "catmull"], ["--kpn-lite", "--kpn-widths", "8,12,16,24,32,48",
                                      "--kpn-no-residual", "--kpn-no-mv-dilate", "--kpn-depth-soft"]):
        args = ap.parse_args(["--arch", "kpn", "--jitter-sign", "-1", *flags])
        m = build_model(args, (16, 24), (32, 48))
        if not flags:
            assert m.widths == DEFAULT_WIDTHS
        elif len(flags) == 1:
            assert m.widths == LITE_WIDTHS
        path = directory / "roundtrip.pt"
        torch.save(m.state_dict(), path)
        save_sidecar(path, args)
        loaded, cfg = load_checkpoint(path, (16, 24), (32, 48))
        assert cfg == load_sidecar(path) == infer_config(m.state_dict())
        assert model_kwargs(cfg) == model_kwargs(args)
        for name, value in m.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name]), name
        path.with_suffix(".pt.json").unlink()
        bare, inferred = load_checkpoint(path, (63, 95), (126, 190))
        assert inferred == cfg and bare.jitter_sign == -1
    for kwargs in (dict(widths=(1, 2)), dict(widths=(0,) * 6), dict(jitter_sign=0)):
        try:
            KPNAccumulator((16, 24), (32, 48), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    print("KPN CLI, presets, custom widths, strict sidecar/bare checkpoint and resized load passed")


def test_training(directory):
    data, ckpt = directory / "engine", directory / "kpn.pt"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def run(*args):
        result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                                text=True, timeout=180, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    run("tools/make_fake_engine.py", "--out", data, "--scenes", 2, "--frames", 6, "--render", "32x48")
    for path in data.glob("*/_cache_tm1.pt"):
        seq = torch.load(path, weights_only=True)
        for f in seq:
            f["mask"] = torch.ones(f["gt"].shape[:2])
            f["mask"][:4] = 0
        torch.save(seq, path)
    output = run("train.py", "--engine-data", data, "--arch", "kpn", "--jitter-sign", -1,
                 "--kpn-trunk-stride", 2, "--kpn-taps", 3, "--kpn-history-filter", "catmull",
                 "--epochs", 1, "--crop", 24, "--bptt", 2, "--val-scenes", 1, "--eval-crops", 1,
                 "--device", "cpu", "--no-full-eval", "--save", ckpt, "--lr", ".0001",
                 "--warmup-frames", 1, "--cold-start-prob", 0, "--augment", "--dither-aug", 1,
                 "--exposure-aug", ".8,1.2", "--edge-loss-weight", ".2", "--flicker-weight", ".1",
                 "--temporal-through", "--alpha-penalty", ".01", "--ssim-weight", ".01",
                 "--grad-weight", ".01", "--freq-weight", ".01", "--ema", ".999")
    losses = re.findall(r"epoch\s+\d+\s+loss\s+(\S+)", output)
    assert losses and all(math.isfinite(float(value)) for value in losses), output
    last = ckpt.with_name("kpn_last.pt")
    trained, cfg = load_checkpoint(last, (32, 48), (64, 96))
    assert cfg["arch"] == "kpn" and cfg["jitter_sign"] == -1
    assert (cfg["trunk_stride"], cfg["taps"], cfg["history_filter"]) == (2, 3, "catmull")
    assert infer_config(trained.state_dict()) == cfg
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    print(f"KPN one CPU training epoch: loss {losses[0]}; masks, BPTT, augmentation and losses passed")
    run("eval_full.py", "--engine-data", data, "--ckpt", last,
        "--device", "cpu", "--frames", 4, "--skip", 1, "--block", 1)
    cache = data / "scene_00" / "_cache_tm1.pt"
    scripts = Path(__file__).resolve().parents[2] / "rdr2_mod" / "tools"
    run(scripts / "eval_cache.py", cache, "--ckpt", last, "--device", "cpu", "--warmup", 1, "--frames", 4)
    run(scripts / "accumulate_gt.py", cache, "--ckpt", last, "--device", "cpu", "--warmup", 1, "--frames", "0:4")
    print("KPN eval_full.py, eval_cache.py and accumulate_gt.py CPU entry points passed")



def test_variants():
    for stride in (1, 2):
        for taps in (3, 5):
            for history_filter in ("bicubic", "catmull"):
                for size in ((1, 1), (17, 31), (32, 48)):
                    m = KPNAccumulator(size, tuple(2*v for v in size), widths=(3, 4, 5, 6, 7, 8),
                                       trunk_stride=stride, taps=taps, history_filter=history_filter)
                    state = m.init_state()
                    for i in range(2):
                        out, state = m(state, *frame(size, i))
                    assert out.shape == (*m.output_size, 3) and torch.isfinite(out).all()
                    out.square().mean().backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
                    assert m.trunk.enc[0].in_channels == 16*stride**2
                    assert m.trunk.head.out_channels == 28*stride**2
                    measured = []
                    hooks = [layer.register_forward_hook(lambda layer, args, out: measured.append(
                        out.shape[-2]*out.shape[-1]*layer.weight.numel()))
                        for layer in m.trunk.modules() if isinstance(layer, torch.nn.Conv2d)]
                    with torch.no_grad():
                        m.trunk(torch.zeros(1, 16, *m.padded_size))
                    for hook in hooks:
                        hook.remove()
                    assert sum(measured) == m.cost()["trunk_macs"]
                    assert m.cost()["filter_macs"] == math.prod(m.output_size)*taps*taps*3
                    cfg = infer_config(m.state_dict())
                    loaded = build_model(cfg, size, m.output_size)
                    loaded.load_state_dict(m.state_dict(), strict=True)
                    assert (loaded.trunk_stride, loaded.taps, loaded.history_filter) == (stride, taps, history_filter)
    old = KPNAccumulator((8, 12), (16, 24), widths=(3,)*6)
    assert not any(k in old.state_dict() for k in ("_kpn_stride", "_kpn_taps", "_kpn_catmull"))
    loaded = build_model(infer_config(old.state_dict()), old.render_size, old.output_size)
    loaded.load_state_dict(old.state_dict(), strict=True)
    inputs = frame(old.render_size)
    with torch.no_grad():
        assert torch.equal(old(old.init_state(), *inputs)[0], loaded(loaded.init_state(), *inputs)[0])
    print("KPN stride/taps/history combinations: shapes, gradients, MACs, strict legacy load passed")


def test_catmull():
    h, w = 16, 24
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xy = torch.stack((x, y), -1).float()[None]
    image = torch.rand(1, 3, h, w, requires_grad=True)
    for offset in ((0, 0), (2, -1)):
        pos = xy + torch.tensor(offset)
        actual = catmull_sample(image, (pos+.5)/torch.tensor([w, h]))
        expected = image[:, :, pos[0, ..., 1].long().clamp(0, h-1), pos[0, ..., 0].long().clamp(0, w-1)]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    pos = xy[:, 2:-3, 2:-3] + torch.tensor([.37, .63])
    ramp = (x+2*y).float()[None, None]
    actual = catmull_sample(ramp, (pos+.5)/torch.tensor([w, h]))
    torch.testing.assert_close(actual[:, 0], pos[..., 0]+2*pos[..., 1], atol=1e-5, rtol=1e-6)
    uv = (torch.rand(1, 11, 13, 2)*1.4-.2).requires_grad_()
    actual = catmull_sample(image, uv)
    # Independent Keys kernel evaluated at absolute distance for all sixteen taps.
    p = uv*torch.tensor([w, h])-.5
    base = p.floor().long()
    def keys(distance):
        d = distance.abs()
        return torch.where(d <= 1, 1.5*d**3-2.5*d**2+1,
                           torch.where(d < 2, -.5*d**3+2.5*d**2-4*d+2, 0.))
    expected = torch.zeros_like(actual)
    for j in range(-1, 3):
        for i in range(-1, 3):
            ix, iy = base[..., 0]+i, base[..., 1]+j
            sample = image[:, :, iy[0].clamp(0, h-1), ix[0].clamp(0, w-1)]
            expected = expected + sample*(keys(p[..., 0]-ix)*keys(p[..., 1]-iy))[:, None]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert image.grad.abs().sum() > 0 and uv.grad.abs().sum() > 0
    assert torch.isfinite(image.grad).all() and torch.isfinite(uv.grad).all()
    print("Catmull-Rom: integer identity/offsets, linear ramp, 16-tap reference and gradients passed")

def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    test_variants()
    test_catmull()
    test_forward()
    test_filter()
    test_proximity()
    test_proximity_extremes()
    test_reset_and_pack()
    test_gradients_and_cost()
    with tempfile.TemporaryDirectory() as temp:
        test_config(Path(temp))
        test_training(Path(temp))
    print("test_kpn passed")


if __name__ == "__main__":
    main()
