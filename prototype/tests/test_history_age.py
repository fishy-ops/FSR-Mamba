"""Frame-count reprojection, deployment parity, legacy identity and static convergence."""
import argparse
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, infer_config, load_checkpoint,
                            load_model_state, model_kwargs, save_sidecar)
from fsrmamba.fast import FastAccumulator
from fsrmamba.fused import FusedFast, _CUDA, _cuda_source, materialize_state, ref_pack
from test_fused import frames


def make(render=(16, 24), **opts):
    m = FastAccumulator(render, tuple(2 * v for v in render), widths=(8, 16),
                        depths=(1, 1), n_state=0, **opts).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, .015)
        m.out.bias.normal_(0, .1)
    return m


@torch.no_grad()
def parity(device="cpu"):
    options = [dict(), dict(accum=True), dict(coverage=True),
               dict(depth_test=True, depth_soft=True, depth_soft_osc=True, coverage=True,
                    mv_dilate=True, carry_raw=True, base_gate=True, accum=True,
                    nearest_sample=True, conf_consistent=True, hist_filter="bicubic"),
               dict(depth_test=True, depth_soft=True, depth_soft_osc=True, coverage=True,
                    mv_dilate=True, depth_dilate=True, thin_lock=True, carry_raw=True,
                    base_gate=True, accum=True, nearest_sample=True,
                    conf_consistent=True, hist_filter="bicubic")]
    worst = 0
    for render in ((64, 96), (63, 95)):
        for opts in options:
            m = make(render, history_age=True, **opts).to(device)
            f = FusedFast(m)
            a, b = m.init_state(device), f.init_state()
            for i, inputs in enumerate(frames(render, device, 8)):
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        rgb, a = m(a, *inputs)
                    other, b = f.step(b, *inputs, write_rgb=True)
                else:
                    rgb, a = m(a, *inputs)
                    other, b = f.step_reference(b, *inputs)
                error = float((rgb - other).abs().max())
                worst = max(worst, error)
                assert error < (8e-3 if device == "cuda" else 3e-5), (render, opts, i, error)
                assert torch.equal(a.age, b.age)
                assert a.age.min() >= 0 and a.age.max() <= 32
                if a.conf is not None:
                    assert torch.allclose(a.conf, b.conf, atol=.08 if device == "cuda" else 3e-5, rtol=1e-5)
            clone = materialize_state(b)
            assert clone.age.data_ptr() != b.age.data_ptr()
            assert a.detach().age.grad_fn is None
    print(f"history_age {device}: 10 configurations, 8 frames; max RGB error {worst:.8g}")


@torch.no_grad()
def configuration():
    assert FAST_DEFAULTS["history_age"] is False
    parser = argparse.ArgumentParser()
    add_arch_args(parser)
    args = parser.parse_args(["--arch", "fast", "--fast-history-age", "--fast-state", "0"])
    assert model_kwargs(args)["history_age"]
    from train import make_model
    assert make_model(args, (8, 12), (16, 24), "cpu").history_age
    m = make(history_age=True, coverage=True)
    cfg = infer_config(m.state_dict())
    assert cfg["history_age"] and cfg["n_state"] == 0
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.pt"
        torch.save(m.state_dict(), path)
        save_sidecar(path, cfg)
        loaded, _ = load_checkpoint(path, m.render_size, m.output_size)
        assert loaded.history_age and loaded.alpha_min.item() == m.alpha_min.item()
    for coverage in (False, True):
        old = make(coverage=coverage)
        new = make(history_age=True, coverage=True)
        load_model_state(new, old.state_dict(), strict=True)
        assert torch.equal(new.stem.weight[:, -4:], torch.zeros_like(new.stem.weight[:, -4:]))
    for detail in (0, 4):
        old = FastAccumulator((16, 24), (32, 48), widths=(8, 16), depths=(1, 1),
                              n_state=2, detail_ch=detail)
        new = FastAccumulator((16, 24), (32, 48), widths=(8, 16), depths=(1, 1),
                              n_state=2, detail_ch=detail, history_age=True)
        load_model_state(new, old.state_dict())
        assert torch.equal(new.stem.weight[:, -2:], old.stem.weight[:, -2:])
    path = Path("../ckpt/dither_osc_mac_last.pt")
    if path.exists():
        old, cfg = load_checkpoint(path, (16, 24), (32, 48))
        cfg["history_age"] = True
        from fsrmamba.config import build_model
        new = build_model(cfg, (16, 24), (32, 48))
        load_model_state(new, old.state_dict())
        a, b = old.init_state(), new.init_state()
        for inputs in frames((16, 24), "cpu", 8):
            _, a = old(a, *inputs)
            rgb, b = new(b, *inputs)
            assert torch.isfinite(rgb).all()
        print("dither_osc_mac_last warm start: 8 CPU frames passed")
    plain = make()
    assert _cuda_source(plain) is _CUDA
    assert not any("HISTORY_AGE" in v for v in FusedFast(plain)._kernel_options())
    source = _cuda_source(make(history_age=True, coverage=True, mv_dilate=True))
    assert 'const float *prev_age' in source and '+K_HISTORY_AGE)' in source
    assert 'alpha=fminf(alpha' in source
    # Explicit off must preserve every tensor and result, including optional paths.
    for opts in ({}, dict(accum=True, coverage=True, carry_raw=True, base_gate=True,
                         mv_dilate=True, nearest_sample=True, conf_consistent=True,
                         hist_filter="bicubic", depth_test=True, depth_soft=True, depth_soft_osc=True)):
        torch.manual_seed(59)
        old = make(**opts)
        torch.manual_seed(59)
        off = make(history_age=False, **opts)
        assert old.state_dict().keys() == off.state_dict().keys()
        assert _cuda_source(old) == _cuda_source(off)
        a, b = old.init_state(), off.init_state()
        for inputs in frames(old.render_size, "cpu", 8):
            rgb, a = old(a, *inputs)
            other, b = off(b, *inputs)
            assert torch.equal(rgb, other) and b.age is None
    print("history_age defaults, metadata, warm-start expansion and off identity passed")


@torch.no_grad()
def count_and_convergence():
    m = make(history_age=True, depth_test=True, mv_dilate=True)
    state = m.init_state()
    lr = torch.rand(16, 24, 3)
    mv = torch.zeros(16, 24, 2)
    depth = torch.full((16, 24), .5)
    for i in range(40):
        _, state = m(state, lr, mv, depth, (0, 0))
        assert state.age.eq(min(i, 32)).all()
        if i:
            limit = max(1 / (1 + min(i-1, 32)), float(m.alpha_min))
            assert m._last_alpha.max() <= limit + 1e-7
    depth = torch.full_like(depth, .2)
    _, state = m(state, lr, mv, depth, (0, 0))
    assert state.age.eq(0).all() and m._last_alpha.eq(1).all()
    state.frame_index = 0
    state.age.fill_(32)
    assert ref_pack(m, state, lr, mv, depth, (0, 0))["new_age"].eq(0).all()
    soft = make(history_age=True, depth_test=True, depth_soft=True,
                depth_soft_osc=True, coverage=True).train()
    state = soft.init_state()
    state.frame_index = 20
    state.age.fill_(20)
    state.depth.fill_(.44)
    flat = torch.full_like(lr, .5)
    state.m1.fill_(.5)
    state.m2.fill_(.25)
    state.luma.fill_(.5)
    # A depth mismatch with strong oscillation retains history under the soft gate.
    state.osc.fill_(.02)
    _, next_state = soft(state, flat, mv, torch.full_like(depth, .5), (0, 0))
    assert ((soft._last_reset > 0) & (soft._last_reset < .5)).all()
    assert next_state.age.eq(21).all() and not next_state.age.requires_grad
    state.osc.zero_()
    _, next_state = soft(state, flat, mv, torch.full_like(depth, .5), (0, 0))
    assert soft._last_reset.gt(.5).all() and next_state.age.eq(0).all()
    dilated = make(history_age=True, mv_dilate=True)
    state = dilated.init_state()
    state.frame_index = 1
    state.age.copy_(torch.arange(24)[None, None, None].expand_as(state.age))
    moving = mv.clone()
    moving[8, 12, 0] = 1 / 24
    near_depth = torch.full_like(depth, .3)
    near_depth[8, 12] = .8
    _, state = dilated(state, lr, moving, near_depth, (0, 0))
    assert state.age[0, 0, 8, 11] == 13
    assert state.age[0, 0, 8, 10] == 11
    old = make()
    old.out.weight[:, :].mul_(.1)
    old.out.weight[3 * old.p * 4:4 * old.p * 4].zero_()
    old.out.bias[3 * old.p * 4:4 * old.p * 4].fill_(8)
    new = make(history_age=True)
    load_model_state(new, old.state_dict())
    states = [old.init_state(), new.init_state()]
    outputs = [[], []]
    y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
    for i in range(48):
        jitter = ((.4 if i % 2 else -.4), 0.)
        lr = (.5 + .3 * torch.sin((x + jitter[0]) * 1.8))[..., None].expand(-1, -1, 3).contiguous()
        for k, model in enumerate((old, new)):
            rgb, states[k] = model(states[k], lr, mv, torch.full_like(depth, .5), jitter)
            if i >= 16:
                outputs[k].append(rgb)
    variances = [float(torch.stack(v).var(0, unbiased=False).mean()) for v in outputs]
    assert variances[1] < variances[0], variances
    print(f"Static jitter variance after 16 frames: off={variances[0]:.8g}, on={variances[1]:.8g}")
    m = make(history_age=True).train()
    state = m.init_state()
    state.frame_index = 40
    state.age.fill_(32)
    m.out.bias[3*m.p*4:4*m.p*4].fill_(8)
    with torch.enable_grad():
        m(state, torch.rand(16, 24, 3), mv, depth, (0, 0))
        m._last_alpha.mean().backward()
    assert m.alpha_min.grad is not None and m.alpha_min.grad > 0


if __name__ == "__main__":
    torch.manual_seed(712)
    configuration()
    parity()
    count_and_convergence()
    from test_coverage import test_training
    test_training(depth_soft_osc=True, history_age=True, flicker_weight=.1)
    if torch.cuda.is_available():
        try:
            import cupy
        except ImportError:
            print("SKIP: CUDA history_age requires CuPy")
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            parity("cuda")
    else:
        print("SKIP: CUDA history_age unavailable")
    print("test_history_age passed")
