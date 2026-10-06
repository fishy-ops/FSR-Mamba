"""Generate independent PyTorch fixtures and run the native reader/math tests."""
import argparse
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_weights import export
import torch
import torch.nn.functional as F
from fsrmamba.fast import FastAccumulator, _nearest_depth, _thin_feature, _coverage_coefficient
from fsrmamba.fused import _HostFrameCache, _signed_jitter, ref_pack


def check_kpn_export(root):
    from fsrmamba.kpn_unet import KPNAccumulator
    for key, values in (("sigma_min", (0, -.05, .000999999999, 2.500000001)),
                        ("proximity", (-.35, -1e-100, .000999999999, 16.000000001)),
                        ("proximity_gain", (0, -2, .000999999999, 64.000000001))):
        values += (1e-100, 1e-46, 1e-40, float("nan"), float("inf"), -float("inf"), 1e100)
        for value in values:
            model = KPNAccumulator((7, 9), (14, 18), widths=(3,)*6,
                                   sigma_min=.05, proximity=.35, proximity_gain=2.)
            setattr(model, key, value)
            try:
                export(model, root/"invalid_kpn.bin")
            except ValueError as error:
                assert key in str(error) and "[1e-3," in str(error), str(error)
            else:
                raise AssertionError(f"export accepted invalid KPN {key}={value}")
    print("PASS: KPN export rejects nonfinite, out-of-range and fp32 overflow/underflow options")


@torch.no_grad()
def run_model(m, root, cases, binary, legacy=False):
    weight_path = root / "weights.bin"
    tensors = export(m, weight_path)
    if legacy:
        raw = weight_path.read_bytes()
        size = struct.unpack_from("<I", raw, 8)[0]
        cfg = json.loads(raw[12:12+size])
        del cfg["coverage_bias"]
        header = json.dumps(cfg).encode("ascii")
        weight_path.write_bytes(raw[:8] + struct.pack("<I", len(header)) + header + raw[12+size:])
    result = subprocess.run([binary, weight_path], check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(tensors)
    for line in lines:
        name, count, total, weighted = line.split()
        v = tensors[name].double().flatten()
        assert int(count) == v.numel()
        assert abs(float(total) - v.sum().item()) < 1e-8
        assert abs(float(weighted) - (v * torch.arange(1, v.numel()+1)).sum().item()) < 1e-7
    cache = _HostFrameCache(m)
    ref_path = root / "ref.bin"
    with ref_path.open("wb") as f:
        f.write(struct.pack("<II", int(m.depth_soft), int(m.coverage)))
        calibration_bias = float(m.coverage_bias) if m.coverage else -6.0
        f.write(struct.pack("<ff", calibration_bias, float(torch.sigmoid(torch.tensor(calibration_bias).half()))))
        f.write(struct.pack("<I", len(cases)))
        for j in cases:
            signed = _signed_jitter(m, j)
            weight, bias, phase, offsets, kernels, windows = cache.get(signed)
            if kernels is None:
                from fsrmamba.resolve import phase_kernels
                ph = torch.tensor([.25, .75])
                yy, xx = torch.meshgrid(ph, ph, indexing="ij")
                ty, tx = torch.meshgrid(torch.arange(-2, 2), torch.arange(-2, 2), indexing="ij")
                _, kernels, _, _, _, windows = phase_kernels(signed, "cpu", xx.reshape(4, 1, 1), yy.reshape(4, 1, 1), tx[None], ty[None])
                windows = torch.tensor(windows)
            f.write(struct.pack("<dd", *j))
            for t in (torch.tensor(signed), phase.float(), offsets, windows, kernels.float()):
                f.write(t.numpy().astype("<i4" if t.dtype in (torch.int32, torch.int64) else "<f4").tobytes())
            for t in (weight, bias):
                f.write(struct.pack("<I", t.numel()))
                f.write(t.float().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<f", float((1 + m.thin_slack.detach()).half()) if m.thin_lock else 1))
        f.write(struct.pack("<I", 2))
        for h, w in ((8, 12), (63, 95)):
            rgb = torch.rand(h, w, 3).half()
            rgb[:, w // 2 - 1:w // 2 + 2] = .1
            rgb[:, w // 2] = .9
            motion = torch.randn(h, w, 2) / w
            # Repeated levels exercise ties after half rounding, including all borders.
            depth = (torch.randint(0, 4, (h, w)).float() * .2 + .1).half()
            depth[-2:, -2:] = .8
            lr = rgb.permute(2, 0, 1)[None]
            near, selected = _nearest_depth(depth[None, None], motion[None])
            thin = _thin_feature(lr) if m.thin_lock else torch.zeros(1, 1, h, w).half()
            mn = -torch.nn.functional.max_pool2d(-lr, 3, 1, 1)
            mx = torch.nn.functional.max_pool2d(lr, 3, 1, 1)
            slack = (mx - mn) * m.box_slack.abs().half()
            if m.thin_lock:
                slack *= torch.where(thin > .5, (1 + m.thin_slack.detach()).half(), torch.ones_like(thin))
            expected = torch.cat((selected[0] if m.mv_dilate else motion,
                                  (near[0, 0] if m.depth_dilate else depth)[..., None].float(),
                                  thin[0].permute(1, 2, 0).float(),
                                  slack[0].permute(1, 2, 0).float()), dim=-1)
            f.write(struct.pack("<II", h, w))
            for tensor in (rgb, motion, depth, expected):
                f.write(tensor.detach().float().numpy().astype("<f4").tobytes())
        logits = torch.tensor([-10., -6., -5., -3., -2., 0., 4., 10.]).half()
        coeff = _coverage_coefficient(logits, calibration_bias)
        f.write(struct.pack("<I", logits.numel()))
        f.write(torch.stack((logits, coeff), 1).float().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<I", 8 if m.coverage else 0))
        if m.coverage:
            state = m.init_state()
            # Pack at its configured extent; all moments are independent of trunk execution.
            h, width = m.render_size
            for i in range(8):
                rgb = torch.rand(h, width, 3).half()
                rgb[4:8, 7:12] = .2 if i % 2 else .8
                motion = torch.randn(h, width, 2) * .01
                motion[0, :, 0] = -1
                depth = torch.full((h, width), .5).half()
                depth[9:13, 15:19] = .2 if i % 2 else .8
                previous = torch.cat((state.m1, state.m2, state.osc, state.luma), 1)
                packed = ref_pack(m, state, rgb, motion, depth, (0, 0))
                cov_start = 25 + 4 * m.carry_raw + m.thin_lock
                evidence = F.pixel_shuffle(packed["X"], 2)[:, cov_start:cov_start+2, :h, :width]
                _, selected = _nearest_depth(depth[None, None], motion[None])
                reset = packed["reset"]
                next_state = torch.cat(packed["coverage_state"], 1)
                f.write(struct.pack("<II", h, width))
                for t in (rgb, previous[0].permute(1, 2, 0),
                          selected[0] if m.mv_dilate else motion, reset[0].permute(1, 2, 0),
                          evidence[0].permute(1, 2, 0), next_state[0].permute(1, 2, 0)):
                    f.write(t.float().numpy().astype("<f4").tobytes())
                state.m1, state.m2, state.osc, state.luma = packed["coverage_state"]
                state.depth = packed["depth"]
                state.frame_index += 1
        threshold = float(m.soft_osc_threshold.detach().clamp(.05, 4)) if m.depth_soft_osc else .5
        f.write(struct.pack("<If", int(m.depth_soft_osc), threshold))
        cases_depth = list(itertools.product((False, True), (False, True),
                                            (.2, .36, .4, .48, .5, .6, .8),
                                            (0., .049, .05, .5, .501, 3.999, 4.)))
        f.write(struct.pack("<I", len(cases_depth)))
        mn, mx = torch.tensor([.4, .48]).half()
        for offscreen, first, prev, osc_n in cases_depth:
            mismatch = prev < float(mn * .9) or prev > float(mx * 1.1)
            departure = prev > float(mx * 1.25)
            dither = float(torch.tensor(osc_n).clamp(0, 4).half()) > threshold
            reset = offscreen or first or (m.depth_test and (
                mismatch if not m.depth_soft else m.depth_soft_osc and (departure or (mismatch and not dither))))
            f.write(struct.pack("<IIffffI", int(offscreen), int(first), prev, float(mn), float(mx), osc_n, int(reset)))
        f.write(struct.pack("<If", int(m.history_age), float(m.alpha_min.detach().clamp(0, 1)) if m.history_age else .05))
        f.write(struct.pack("<I", 4))
        for h, width in ((8, 12), (63, 95), (64, 96), (1, 1)):
            y, x = torch.meshgrid(torch.arange(h), torch.arange(width), indexing="ij")
            previous = torch.randint(0, 33, (h, width)).float()
            motion = torch.randn(h, width, 2) / width
            motion[0, :, 0] = -1
            raw_reset = torch.rand(h, width) < .15
            u, v = (x + .5) / width + motion[..., 0], (y + .5) / h + motion[..., 1]
            reset = raw_reset | (u < 0) | (u >= 1) | (v < 0) | (v >= 1)
            grid = torch.stack((u, v), -1)[None] * 2 - 1
            age = F.grid_sample(previous[None, None], grid, mode="nearest", padding_mode="border",
                                align_corners=False)[0, 0] * (~reset)
            next_age = torch.where(reset, 0., (age + 1).clamp(max=32))
            feature = (torch.log2(1 + age) / 5).half().float()
            floor = float(m.alpha_min.detach().clamp(0, 1)) if m.history_age else .05
            alpha = torch.rand(h, width).half().float()
            limit = torch.maximum(1 / (1 + age), torch.tensor(floor)).half().float()
            expected_alpha = torch.where(reset, 1., torch.minimum(alpha, limit))
            f.write(struct.pack("<II", h, width))
            for t in (previous, motion, raw_reset.float(), alpha,
                      torch.stack((age, next_age, feature, expected_alpha), -1)):
                f.write(t.float().numpy().astype("<f4").tobytes())
    subprocess.run([binary, weight_path, ref_path], check=True)
    return weight_path


def emulate_dml_depth_to_space(t, order):
    """Pure-Python model of DML_DEPTH_TO_SPACE1 with block size 2 on an NCHW tensor.

    COLUMN_ROW_DEPTH (the assumed order): input channel = c*4 + row*2 + col, i.e. PyTorch pixel_shuffle.
    DEPTH_COLUMN_ROW (the ini 'depth_to_space_alt' order): input channel = (row*2 + col)*C + c, ONNX DCR.
    """
    _, cc, h, w = t.shape
    c = cc // 4
    out = torch.zeros(1, c, 2 * h, 2 * w, dtype=t.dtype)
    for ch in range(c):
        for row in range(2):
            for col in range(2):
                src = ch * 4 + row * 2 + col if order == "crd" else (row * 2 + col) * c + ch
                out[0, ch, row::2, col::2] = t[0, src]
    return out


def check_depth_to_space():
    t = torch.randn(1, 12, 5, 7)
    assert torch.equal(emulate_dml_depth_to_space(t, "crd"), torch.nn.functional.pixel_shuffle(t, 2))
    assert not torch.equal(emulate_dml_depth_to_space(t, "dcr"), torch.nn.functional.pixel_shuffle(t, 2))
    # The shaders and gpu.cpp assume the first order; the model's up-convs are trained against pixel_shuffle.


@torch.no_grad()
def check_motion_cache():
    """Compare the FAST 5x5 cache with independently clamped nearest_depth()/motion() taps."""
    for h, w in ((1, 1), (2, 3), (8, 12), (63, 95), (64, 96)):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        for inverted in (False, True):
            raw_depth = torch.randint(0, 4, (h, w)).float() * .2 + .1
            depth = (raw_depth if inverted else 1 - raw_depth).half()
            for display in (False, True):
                raw = torch.randn(h * (2 if display else 1), w * (2 if display else 1), 2)
                motion = (raw[1::2, 1::2] if display else raw) * torch.tensor([-.003, .007]) - torch.tensor([.001, -.002])
                _, selected = _nearest_depth(depth[None, None], motion[None])
                old = selected[0]
                padded = F.pad(depth[None, None], (2, 2, 2, 2), mode="replicate")[0, 0]
                window = [padded[dy:dy+h, dx:dx+w] for dy in range(5) for dx in range(5)]
                cache = []
                for dy in range(3):
                    for dx in range(3):
                        near = window[dy*5+dx]
                        bx, by = x + dx - 2, y + dy - 2
                        for sy in range(3):
                            for sx in range(3):
                                d = window[(dy+sy)*5+dx+sx]
                                take = d > near
                                bx = torch.where(take, x + dx + sx - 2, bx)
                                by = torch.where(take, y + dy + sy - 2, by)
                                near = torch.maximum(near, d)
                        cache.append(motion[by.clamp(0, h-1), bx.clamp(0, w-1)].clone())
                for k in range(3):
                    cache[k*3][:, 0] = cache[k*3+1][:, 0]
                    cache[k*3+2][:, -1] = cache[k*3+1][:, -1]
                for k in range(3):
                    cache[k][0] = cache[k+3][0]
                    cache[k+6][-1] = cache[k+3][-1]
                for dy in range(3):
                    for dx in range(3):
                        assert torch.equal(cache[dy*3+dx], old[(y+dy-1).clamp(0, h-1), (x+dx-1).clamp(0, w-1)])
                cache = torch.stack(cache)
                def cached_tap(px, py):
                    return cache[(py-y+1)*3+px-x+1, y, x]
                def old_tap(px, py):
                    return old[py.clamp(0, h-1), px.clamp(0, w-1)]
                for q in range(4):
                    pos_x = ((2*x+q%2+.5)/2-.5).clamp(min=0)
                    pos_y = ((2*y+q//2+.5)/2-.5).clamp(min=0)
                    ix, iy = pos_x.floor().long(), pos_y.floor().long()
                    tx, ty = (pos_x-ix)[..., None], (pos_y-iy)[..., None]
                    def interpolate(tap):
                        a = (1-tx)*tap(ix, iy) + tx*tap(ix+1, iy)
                        b = (1-tx)*tap(ix, iy+1) + tx*tap(ix+1, iy+1)
                        return (1-ty)*a + ty*b
                    assert torch.equal(interpolate(cached_tap), interpolate(old_tap))
    print("PASS: FAST motion cache bit-identical centre and all 4 phases; borders, ties, depth inversion, display motion")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("--temp-dir", required=True, type=Path)
    a = p.parse_args()
    binary = a.binary.resolve()
    a.temp_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(824)
    cases = [(x, y) for x in (-.5, -.25, 0., .25, .5) for y in (-.5, -.25, 0., .25, .5)]
    cases += [(float(torch.nextafter(torch.tensor(.25, dtype=torch.float64), torch.tensor(v, dtype=torch.float64))), 0.) for v in (0., 1.)]
    cases += [tuple(j.tolist()) for j in torch.rand(25, 2) - .5]
    with tempfile.TemporaryDirectory(dir=a.temp_dir) as temp:
        root = Path(temp)
        runs = 0
        for sign in (-1., 1.):
            for film in (False, True):
                for carry, gate in ((False, False), (False, True), (True, True)):
                    m = FastAccumulator((9, 11), (18, 22), widths=(7, 11, 13), depths=(0, 1, 2),
                                        n_state=0, accum=True, film=film, carry_raw=carry,
                                        base_gate=gate, nearest_sample=True, jitter_sign=sign).eval()
                    with torch.no_grad():
                        m.out.weight.normal_(0, .13)
                        m.out.bias.normal_(0, .13)
                        if film:
                            m.film[2].weight.normal_(0, .07)
                            m.film[2].bias.normal_(0, .07)
                    weight_path = run_model(m, root, cases, binary)
                    runs += 1
        # Missing optional keys are the original v1 header and must still load.
        valid = weight_path.read_bytes()
        n = struct.unpack_from("<I", valid, 8)[0]
        cfg = json.loads(valid[12:12+n])
        for key in ("history_age", "coverage_bias", "coverage", "depth_soft", "depth_soft_osc", "mv_dilate", "depth_dilate", "thin_lock"):
            del cfg[key]
        header = json.dumps(cfg).encode("ascii")
        weight_path.write_bytes(valid[:8] + struct.pack("<I", len(header)) + header + valid[12+n:])
        subprocess.run([binary, weight_path, root / "ref.bin"], check=True, capture_output=True)
        for opts in (dict(history_age=True),
                     dict(history_age=True, coverage=True, depth_soft=True, depth_soft_osc=True, mv_dilate=True),
                     dict(history_age=True, coverage=True, depth_soft=True, depth_soft_osc=True, mv_dilate=True, thin_lock=True, depth_dilate=True),
                     dict(mv_dilate=True), dict(depth_dilate=True), dict(thin_lock=True),
                     dict(mv_dilate=True, depth_dilate=True, thin_lock=True), dict(depth_soft=True),
                     dict(coverage=True), dict(coverage=True, coverage_bias=-6), dict(coverage=True, coverage_bias=-2.75),
                     dict(coverage=True, depth_soft=True, mv_dilate=True),
                     dict(coverage=True, depth_soft=True, depth_soft_osc=True, mv_dilate=True),
                     dict(coverage=True, depth_soft=True, depth_soft_osc=True, mv_dilate=True, thin_lock=True, depth_dilate=True),
                     dict(coverage=True, depth_dilate=True, thin_lock=True, mv_dilate=True),
                     dict(depth_soft=True, mv_dilate=True),
                     dict(depth_soft=True, mv_dilate=True, depth_dilate=True, thin_lock=True)):
            m = FastAccumulator((63, 95), (126, 190), widths=(16, 32), depths=(1, 2),
                                n_state=0, accum=True, film=True, carry_raw=True, base_gate=True,
                                depth_test=True, nearest_sample=True, conf_consistent=True,
                                hist_filter="bicubic", jitter_sign=-1, **opts).eval()
            with torch.no_grad():
                m.box_slack.fill_(.2731)
                if m.thin_lock:
                    m.thin_slack.fill_(1.0317)
            weight_path = run_model(m, root, cases, binary, legacy=opts.get("coverage_bias") == -6)
            runs += 1
        for floor in (-1., .1234, 10.):
            m = FastAccumulator((9, 11), (18, 22), widths=(8, 16), depths=(1, 1), n_state=0,
                                accum=True, carry_raw=True, base_gate=True, nearest_sample=True,
                                history_age=True).eval()
            with torch.no_grad():
                m.alpha_min.fill_(floor)
            run_model(m, root, cases, binary)
            runs += 1
        for threshold in (-1., 10.):
            m = FastAccumulator((9, 11), (18, 22), widths=(8, 16), depths=(1, 1), n_state=0,
                                accum=True, carry_raw=True, base_gate=True, nearest_sample=True,
                                depth_test=True, depth_soft=True, depth_soft_osc=True, coverage=True).eval()
            with torch.no_grad():
                m.soft_osc_threshold.fill_(threshold)
            run_model(m, root, cases, binary)
            runs += 1
        valid = weight_path.read_bytes()
        n = struct.unpack_from("<I", valid, 8)[0]
        cfg = json.loads(valid[12:12+n])
        assert cfg["depth_soft"] is True
        for value in (True, 1, "true"):
            bad_cfg = dict(cfg, depth_soft=value, depth_test=False)
            header = json.dumps(bad_cfg).encode("ascii")
            weight_path.write_bytes(valid[:8] + struct.pack("<I", len(header)) + header + valid[12+n:])
            assert subprocess.run([binary, weight_path], capture_output=True).returncode != 0
        for missing in ("depth_test", "depth_soft", "coverage"):
            bad_cfg = dict(cfg, depth_soft_osc=True)
            bad_cfg[missing] = False
            header = json.dumps(bad_cfg).encode("ascii")
            weight_path.write_bytes(valid[:8] + struct.pack("<I", len(header)) + header + valid[12+n:])
            assert subprocess.run([binary, weight_path], capture_output=True).returncode != 0
        for bad in (b"", valid[:8], valid[:-1], valid + b"!", b"BADMAGIC" + valid[8:]):
            weight_path.write_bytes(bad)
            assert subprocess.run([binary, weight_path], capture_output=True).returncode != 0
        for kwargs in (dict(n_state=1), dict(accum=False), dict(hist_residual=True), dict(resolve="lanczos")):
            config = dict(n_state=0, accum=True)
            config.update(kwargs)
            model = FastAccumulator((8, 8), (16, 16), **config)
            try:
                export(model, weight_path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted unsupported config {kwargs}")
    from test_kpn import run as check_kpn
    with tempfile.TemporaryDirectory(dir=a.temp_dir) as temp:
        check_kpn_export(Path(temp))
        check_kpn(Path(temp), binary)
    check_depth_to_space()
    check_motion_cache()
    ckpt = Path(__file__).resolve().parents[2] / "ckpt" / "z_s16_carry_bg_sf.pt"
    if ckpt.exists():
        from export_weights import load
        with tempfile.TemporaryDirectory(dir=a.temp_dir) as temp:
            run_model(load(ckpt).float(), Path(temp), cases, binary)
        print("PASS: target checkpoint z_s16_carry_bg_sf round trip (tensor checksums + 52 frame references)")
    else:
        print("SKIP: target checkpoint not present")
    print(f"PASS: {runs} model configurations, {runs * len(cases)} frame references, tensor checksums, malformed files, unsupported models")


if __name__ == "__main__":
    main()
