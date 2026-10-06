"""HQ recurrence, dynamic reconstruction, gradients, checkpoint and training contracts."""

import argparse
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import (add_arch_args, build_model, infer_config, load_checkpoint,
                             load_sidecar, model_kwargs, save_sidecar)
from fsrmamba.hq import HQAccumulator, _WindowBlock
from fsrmamba.synth import halton_jitter


def frame(size, i=0):
    h, w = size
    return (torch.rand(h, w, 3) * .8 + .05, torch.zeros(h, w, 2),
            torch.full((h, w), .5), halton_jitter(i))


def same_state(a, b):
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert torch.equal(x, y) if torch.is_tensor(x) else x == y, f.name


def test_forward():
    for size in ((64, 96), (63, 95)):
        m = HQAccumulator(size, tuple(v * 2 for v in size)).eval()
        frames = [frame(size, i) for i in range(6)]
        outputs = []
        with torch.no_grad():
            for repeat in range(2):
                state = m.init_state()
                for i, f in enumerate(frames):
                    out, state = m(state, *f)
                    assert out.shape == (*m.output_size, 3)
                    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() < 1
                    assert state.frame_index == i + 1
                    assert torch.equal(state.age, torch.full_like(state.age, i + 1))
                    assert state.feat.abs().max() <= 1
                    assert m._last_reset.shape == m._last_alpha.shape == (1, 1, *m.output_size)
                    if repeat:
                        assert torch.equal(out, outputs[i])
                    else:
                        outputs.append(out)
        # Render- and output-grid auxiliary inputs obey the same UV contract.
        f = frames[0]
        mv = F.interpolate(f[1].permute(2, 0, 1)[None], size=m.output_size,
                           mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        depth = F.interpolate(f[2][None, None], size=m.output_size, mode="nearest")[0, 0]
        with torch.no_grad():
            out, _ = m(m.init_state(), f[0], mv, depth, f[3])
        assert torch.equal(out, outputs[0])
    print("HQ deterministic six-frame forward: even and odd sizes passed")


def test_reset():
    size = (32, 48)
    m = HQAccumulator(size, (64, 96)).eval()
    f = frame(size)
    with torch.no_grad():
        _, state = m(m.init_state(), *f)
        state.age.fill_(32)
        _, aged = m(state, *f)
        assert aged.age.min() == 32 and aged.age.max() == 32
        detached = aged.detach()
        assert all(not getattr(detached, x.name).requires_grad for x in fields(detached)
                   if torch.is_tensor(getattr(detached, x.name)))
        fresh_out, fresh = m(m.init_state(), *f)
        state.frame_index = 0
        reset_out, reset = m(state, *f)
        assert torch.equal(fresh_out, reset_out)
        same_state(fresh, reset)
        # A depth discontinuity clears all recurrent evidence before processing.
        state.frame_index = 4
        state.depth.fill_(.1)
        changed, changed_state = m(state, *f)
        assert changed_state.age.max() == 1
        assert m._last_reset.min() == 1
        other = m.init_state()
        other.frame_index = 4
        other.depth.fill_(.1)
        clean, clean_state = m(other, *f)
        assert torch.equal(changed, clean)
        same_state(changed_state, clean_state)
        moved = (f[0], torch.ones_like(f[1]) * 2, f[2], f[3])
        _, offscreen = m(aged, *moved)
        assert offscreen.age.max() == 1
        # Reset is per pixel, not just a frame-wide counter.
        partial_mv = torch.zeros_like(f[1])
        partial_mv[:, :12, 0] = -2
        _, partial = m(aged, f[0], partial_mv, f[2], f[3])
        assert partial.age[..., :20].max() == 1
        assert partial.age[..., 40:].min() == 32
    print("HQ hard resets, disocclusion, offscreen, age cap and detach passed")


def test_resolve():
    m = HQAccumulator((8, 12), (16, 24))
    lr = torch.rand(1, 3, 8, 12) * .5
    hist = torch.rand(1, 3, 16, 24) * .9
    hist[..., 0, 0] = torch.nextafter(torch.tensor(1.), torch.tensor(0.))
    params = torch.zeros(1, 8, 16, 24)
    _, centre = m._kernel(params, torch.tensor([[.2, -.3]]))
    weights = m.gaussian_weights(centre, centre, params[:, :1], params[:, :2])
    assert torch.equal(weights[:, 12], torch.ones_like(weights[:, 12]))
    assert weights.sum(1).eq(1).all()
    one, zero = torch.ones_like(params[:, :1]), torch.zeros_like(params[:, :1])
    out = m.resolve(lr, hist, one, weights, centre, zero, hist, zero.bool())
    assert torch.equal(out, hist), (out - hist).abs().max()
    out = m.resolve(lr, hist, zero, weights, centre, zero, torch.zeros_like(hist), zero.bool())
    ix, iy = centre[0, ..., 0].long().clamp(0, 11), centre[0, ..., 1].long().clamp(0, 7)
    assert torch.equal(out, lr[0, :, iy, ix][None])
    # Jitter sign equivalence and constant-image preservation.
    m2 = HQAccumulator((8, 12), (16, 24), jitter_sign=-1)
    m2.load_state_dict(dict(m.state_dict(), _jitter_sign=torch.tensor(-1.)))
    f = frame((8, 12))
    with torch.no_grad():
        a, _ = m(m.init_state(), *f[:3], (.2, -.3))
        b, _ = m2(m2.init_state(), *f[:3], (-.2, .3))
    assert torch.equal(a, b)
    weights, _ = m._kernel(torch.randn_like(params), torch.tensor([[.2, -.3]]))
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-6)
    constant = torch.full_like(lr, .3)
    out = m.resolve(constant, hist, one, weights, centre, zero, torch.zeros_like(hist), one.bool())
    assert torch.allclose(out, torch.full_like(out, .3), atol=1e-6)
    print("HQ resolve exact history/delta endpoints and jitter convention passed")


def test_gradients():
    m = HQAccumulator((32, 48), (64, 96))
    state = m.init_state()
    loss = 0
    for i in range(3):
        out, state = m(state, *frame(m.render_size, i))
        loss = loss + (out - torch.rand_like(out)).square().mean()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    print("HQ finite nonzero gradients reach every parameter tensor through recurrence")


def test_attention():
    block = _WindowBlock(48, (13, 19), True)
    assert block.heads == 2 and block.qkv.out_features == 3 * 64
    assert (block.mask < 0).any() and (block.mask == 0).any()
    x = torch.randn(1, 48, 13, 19, requires_grad=True)
    out = block(x)
    out[..., 0, 0].sum().backward()
    # A shifted corner window must not communicate with the wrapped far corner.
    assert x.grad[..., -1, -1].abs().max() == 0
    assert torch.isfinite(out).all()


def test_config(directory):
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    args = ap.parse_args(['--arch', 'hq', '--hq-c0', '12', '--hq-widths', '32,64,96',
                          '--hq-depths', '1,2,1', '--hq-state', '8', '--jitter-sign', '-1'])
    m = build_model(args, (16, 24), (32, 48))
    cfg = model_kwargs(args)
    path = directory / 'roundtrip.pt'
    torch.save(m.state_dict(), path)
    save_sidecar(path, args)
    loaded, saved = load_checkpoint(path, (16, 24), (32, 48))
    assert load_sidecar(path) == saved == infer_config(m.state_dict())
    assert model_kwargs(saved) == cfg
    for name, value in m.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    path.with_suffix('.pt.json').unlink()
    inferred, _ = load_checkpoint(path, (16, 24), (32, 48))
    assert inferred.n_state == 8
    build_model(saved, (63, 95), (126, 190)).load_state_dict(m.state_dict(), strict=True)
    print("HQ custom CLI, sidecar, bare checkpoint and resized reconstruction passed")


def test_training(directory):
    data, ckpt = directory / 'engine', directory / 'hq.pt'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def run(*args):
        result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                                text=True, timeout=120, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    run('tools/make_fake_engine.py', '--out', data, '--scenes', 2, '--frames', 6, '--render', '32x48')
    for path in data.glob('*/_cache_tm1.pt'):
        seq = torch.load(path, weights_only=True)
        for f in seq:
            f['mask'] = torch.ones(f['gt'].shape[:2])
            f['mask'][:4] = 0
        torch.save(seq, path)
    output = run('train.py', '--engine-data', data, '--arch', 'hq', '--epochs', 1,
                 '--crop', 24, '--bptt', 2, '--val-scenes', 1, '--eval-crops', 1,
                 '--device', 'cpu', '--no-full-eval', '--save', ckpt, '--lr', '.0001',
                 '--warmup-frames', 1, '--cold-start-prob', 0, '--augment',
                 '--dither-aug', 1, '--exposure-aug', '.8,1.2', '--edge-loss-weight', '.2',
                 '--flicker-weight', '.1', '--temporal-through', '--alpha-penalty', '.01',
                 '--ssim-weight', '.01', '--grad-weight', '.01', '--freq-weight', '.01',
                 '--ema', '.999')
    assert 'epoch   0' in output
    last = ckpt.with_name(ckpt.stem + '_last.pt')
    trained, cfg = load_checkpoint(last, (32, 48), (64, 96))
    assert cfg['arch'] == 'hq'
    assert infer_config(trained.state_dict()) == cfg
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    print("HQ train.py: one CPU engine epoch with defaults, masks, BPTT, augmentations and losses passed")
    run('eval_full.py', '--engine-data', data, '--ckpt', last,
        '--device', 'cpu', '--frames', 4, '--skip', 1, '--block', 1)
    cache = data / 'scene_00' / '_cache_tm1.pt'
    scripts = Path(__file__).resolve().parents[2] / 'rdr2_mod' / 'tools'
    run(scripts / 'eval_cache.py', cache, '--ckpt', last, '--device', 'cpu', '--warmup', 1, '--frames', 4)
    run(scripts / 'accumulate_gt.py', cache, '--ckpt', last, '--device', 'cpu', '--warmup', 1, '--frames', '0:4')
    print("HQ checkpoint: eval_full.py, eval_cache.py and accumulate_gt.py CPU entry points passed")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    test_forward()
    test_reset()
    test_resolve()
    test_gradients()
    test_attention()
    with tempfile.TemporaryDirectory() as temp:
        test_config(Path(temp))
        test_training(Path(temp))
    print('test_hq passed')


if __name__ == '__main__':
    main()
