"""Independent shader arithmetic and deployment-format checks; no GPU required."""
import copy
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
import torch
import torch.nn.functional as F
from export_weights import export, model_config
from export_sequence import export_sequence
from fsrmamba.config import save_sidecar
from fsrmamba.kpn_unet import KPNAccumulator


def bilinear(x, scale=2):
    # DML RESAMPLE1: output = (input + .5) * scale - .5, clamp border.
    h, w = x.shape[:2]
    yy = (np.arange(h * scale, dtype=np.float32) + .5) / scale - .5
    xx = (np.arange(w * scale, dtype=np.float32) + .5) / scale - .5
    iy, ix = np.floor(yy).astype(int), np.floor(xx).astype(int)
    ty, tx = (yy - iy).astype(np.float32)[:, None, None], (xx - ix).astype(np.float32)[None, :, None]
    a = x[iy.clip(0, h-1)[:, None], ix.clip(0, w-1)[None]]
    b = x[iy.clip(0, h-1)[:, None], (ix+1).clip(0, w-1)[None]]
    c = x[(iy+1).clip(0, h-1)[:, None], ix.clip(0, w-1)[None]]
    d = x[(iy+1).clip(0, h-1)[:, None], (ix+1).clip(0, w-1)[None]]
    return (1-ty)*((1-tx)*a+tx*b)+ty*((1-tx)*c+tx*d)


def grid(h, w):
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return (np.stack((x, y), -1)+.5)/np.array([w, h], np.float32)


def position(uv, h, w):
    return (((uv*2-1)+1)*np.array([w, h], np.float32)-1)*.5


def warp(x, uv, cubic=False, catmull=False):
    h, w = x.shape[:2]
    p = uv*np.array([w, h], np.float32)-.5 if catmull else position(uv, h, w)
    if not cubic:
        ix, iy = np.rint(p.clip(0, [w-1, h-1])).astype(int).transpose(2, 0, 1)
        return x[iy, ix]
    ip = np.floor(p).astype(int)
    t = p-ip
    def weights(t):
        if catmull:
            return np.stack((-.5*t+t*t-.5*t*t*t, 1-2.5*t*t+1.5*t*t*t,
                             .5*t+2*t*t-1.5*t*t*t, -.5*t*t+.5*t*t*t), -1).astype(np.float32)
        def c1(x): return ((1.25*x-2.25)*x)*x+1
        def c2(x): return ((-.75*x+3.75)*x-6)*x+3
        return np.stack((c2(t+1), c1(t), c1(1-t), c2(2-t)), -1).astype(np.float32)
    wx, wy = weights(t[..., 0]), weights(t[..., 1])
    out = np.zeros((*uv.shape[:2], x.shape[-1]), np.float32)
    for j in range(4):
        row = np.zeros_like(out)
        for i in range(4):
            row += x[(ip[..., 1]+j-1).clip(0, h-1), (ip[..., 0]+i-1).clip(0, w-1)]*wx[..., i, None]
        out += row*wy[..., j, None]
    return out


def taps(x):
    h, w = x.shape[:2]
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return np.stack([p[y:y+h, z:z+w] for y in range(3) for z in range(3)])


def yc(x):
    r, g, b = np.moveaxis(x, -1, 0)
    return np.stack((.25*r+.5*g+.25*b, .5*r-.5*b, -.25*r+.5*g-.25*b), -1)


def avg2(x):
    h, w, c = x.shape
    return x.reshape(h//2, 2, w//2, 2, c).mean((1, 3))


def pack(model, state, lr, motion, depth, jitter, tracker=None, foliage=None, tracker_reset=False):
    h, w = lr.shape[:2]
    ml = motion if motion.shape[:2] == (h, w) else avg2(motion)
    d = depth if depth.shape == (h, w) else depth[::2, ::2]
    dt = taps(d[..., None])
    if model.mv_dilate:
        best = dt.argmax(0)[..., 0]
        ml = np.take_along_axis(taps(ml), best[None, ..., None], axis=0)[0]
    mh = motion if not model.mv_dilate and motion.shape[:2] == (2*h, 2*w) else bilinear(ml)
    uv, uvh = grid(h, w)+ml, grid(2*h, 2*w)+mh
    off = np.any((uv < 0) | (uv >= 1), -1)
    old = warp(state.depth[0].permute(1, 2, 0).numpy(), uv)[..., 0]
    mismatch = (old < .9*dt.min(0)[..., 0]) | (old > 1.1*dt.max(0)[..., 0])
    invalid = mismatch | off
    reset = off.copy() if model.depth_soft else invalid.copy()
    if state.frame_index == 0:
        reset[:] = True
        invalid[:] = True
    outside = np.any((uvh < 0) | (uvh >= 1), -1)
    reset_hr = reset.repeat(2, 0).repeat(2, 1) | outside
    invalid |= outside.reshape(h, 2, w, 2).any((1, 3))
    hist = warp(state.color[0].permute(1, 2, 0).numpy(), uvh, True, model.history_filter == "catmull").clip(0, 1-1e-6)
    hist[reset_hr] = 0
    age = warp(state.age[0].permute(1, 2, 0).numpy(), uv)[..., 0].clip(0, 32)
    age[invalid] = 0
    current, yh = yc(lr), yc(hist)
    mean = avg2(yh)
    phases = yh[..., 0].reshape(h, 2, w, 2).transpose(0, 2, 1, 3).reshape(h, w, 4)
    lumas = taps(current[..., :1])
    disagreement = ((mean[..., :1]-current[..., :1])/(lumas.max(0)-lumas.min(0)+.02)).clip(-4, 4)
    speed = (np.linalg.norm(ml*np.array([w, h], np.float32), axis=-1)/10).clip(0, 1)
    packed = np.concatenate((current, mean, phases, disagreement, invalid[..., None], speed[..., None],
                             (np.log2(1+age)/5)[..., None],
                             np.broadcast_to(np.array(jitter, np.float32)*model.jitter_sign, (h, w, 2))), -1)
    hp, wp = model.padded_size
    packed = np.pad(packed, ((0, hp-h), (0, wp-w), (0, 0)), mode="edge")
    next_age = np.where(invalid, 0, np.minimum(age+1, 32))
    if foliage is not None and (foliage["foliage_strength"] > 0 or foliage["fallback_strength"] > 0):
        tracker, confidence = pack_tracker(lr, tracker, ml, d, state.depth[0, 0].numpy(), invalid,
                                           np.array(jitter)*model.jitter_sign, foliage,
                                           reset=tracker_reset or state.frame_index == 0)
        return packed, hist, reset_hr, next_age, tracker, confidence
    return packed, hist, reset_hr, next_age


def foliage_controls(**kwargs):
    controls = dict(foliage_strength=0, foliage_ema=.2, foliage_threshold=.08, foliage_eps=.004,
                    foliage_spatial=.3, foliage_history_scale=.5, foliage_alpha_floor=.1,
                    fallback_strength=0, fallback_sigma=.65, fallback_alpha=.5)
    controls.update(kwargs)
    return controls


def smoothstep(a, b, x):
    t = ((x-a)/(b-a)).clip(0, 1)
    return t*t*(3-2*t)


def linear_sample(x, xy):
    h, w = x.shape[:2]
    ip = np.floor(xy).astype(int)
    t = xy-ip
    out = np.zeros((*xy.shape[:2], x.shape[-1]), np.float32)
    for dy in range(2):
        for dx in range(2):
            weight = (t[..., 0] if dx else 1-t[..., 0])*(t[..., 1] if dy else 1-t[..., 1])
            out += weight[..., None]*x[(ip[..., 1]+dy).clip(0, h-1), (ip[..., 0]+dx).clip(0, w-1)]
    return out


def pack_tracker(lr, previous, motion, depth, old_depth, invalid, jitter, controls, reset=False):
    # Motion is the pack's selected LR motion, after any depth dilation.
    h, w = lr.shape[:2]
    xy = grid(h, w)*[w, h]-.5
    uv = grid(h, w)+motion
    yy = yc(lr.astype(np.float16).astype(np.float32))[..., :1]
    L = linear_sample(yy, xy+jitter)[..., 0]
    B = np.zeros((h, w), np.float32)
    mn, mx = np.full((h, w), np.inf), np.full((h, w), -np.inf)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            Y = linear_sample(yy, xy+jitter+[dx, dy])[..., 0]
            B += Y*(.5 if dx == 0 else .25)*(.5 if dy == 0 else .25)
            mn, mx = np.minimum(mn, Y), np.maximum(mx, Y)
    seeded = np.stack((L, np.zeros_like(L), np.zeros_like(L), B), -1)
    if reset:
        return seeded.astype(np.float16).astype(np.float32), np.zeros_like(L)
    old = warp(previous, uv)
    old_B = linear_sample(previous, uv*[w, h]-.5)[..., 3]
    z = warp(old_depth[..., None], uv)[..., 0]
    valid = ~invalid & ~np.any((uv < 0) | (uv >= 1), -1) & (np.abs(z-depth) <= .1*np.maximum(depth, .001))
    agreement = 1-smoothstep(.03, .06, np.abs(B-old_B))
    spread = np.linalg.norm((taps(motion)-motion)*[w, h], axis=-1).max(0)
    confidence = np.where(valid, agreement*(1-spread/1.5).clip(0, 1), 0)
    d = (L-B)-(old[..., 0]-old[..., 3])
    eps = controls["foliage_eps"]
    event = np.where(d*old[..., 1] < -eps*eps, (np.minimum(np.abs(d), np.abs(old[..., 1]))/(mx-mn+eps)).clip(0, 1), 0)
    I = (old[..., 2]+(event-old[..., 2])*controls["foliage_ema"])*agreement
    seeded[..., 1] = np.where(valid & (agreement > 0), d, 0)
    seeded[..., 2] = np.where(valid, I, 0)
    return seeded.astype(np.float16).astype(np.float32), confidence


def foliage_evidence(lr, tracker, confidence, depth, invalid, controls, reset=False):
    zt, it = taps(depth[..., None])[..., 0], taps(tracker)[..., 2]
    weights = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], np.float32)[:, None, None]/16
    weights = weights*(np.abs(zt-depth) <= .1*np.maximum(depth, .001))
    I = (it*weights).sum(0)/np.maximum(weights.sum(0), 1e-8)
    ys = yc(taps(lr))[..., 0]
    R = ys.max(0)-ys.min(0)
    HF = R/(R+controls["foliage_eps"])
    valid = ~invalid & ~np.full(invalid.shape, reset)
    G = np.where(valid, confidence, 0)
    k = np.where(valid, controls["foliage_strength"]*HF*smoothstep(controls["foliage_threshold"], controls["foliage_threshold"]+.1, I), 0)
    bad = controls["fallback_strength"]*HF*(1-G)
    return np.stack((k, G, bad), -1).repeat(2, 0).repeat(2, 1)


def stabilize_value(value, rectified, lo, hi, reset, stabilize=0, stabilize_tau=.5,
                    stabilize_eps=.004, invalid=None, first_frame=False):
    if stabilize == 0 or first_frame:
        return value
    d = value.astype(np.float32)-rectified.astype(np.float32)
    luma = np.array([.2126, .7152, .0722], np.float32)
    L = (np.abs(d)*luma).sum(-1)
    R = ((hi.astype(np.float32)-lo.astype(np.float32))*luma).sum(-1)
    t = (L/(np.float32(stabilize_tau)*R+np.float32(stabilize_eps))).clip(0, 1)
    weight = np.float32(stabilize)*(1-t)
    weight = np.where(reset if invalid is None else reset | invalid, np.float32(0), weight)
    filtered = rectified.astype(np.float32)+d*(1-weight[..., None])
    return np.where((weight > 0)[..., None], filtered, value)


def resolve(model, lr, hist, params, reset, jitter, stabilize=0, stabilize_tau=.5,
            stabilize_eps=.004, invalid=None, first_frame=False, foliage=None, evidence=None):
    h, w = lr.shape[:2]
    y, x = np.meshgrid(np.arange(2*h, dtype=np.float32), np.arange(2*w, dtype=np.float32), indexing="ij")
    xy = (np.stack((x, y), -1)+.5)*.5-.5+np.array(jitter, np.float32)*model.jitter_sign
    centre = np.floor(xy+.5).clip(0, [w-1, h-1]).astype(int)
    sigma = np.maximum(np.exp(params[..., :2].clip(np.log(model.sigma_min), np.log(2.5))), 1e-4)
    controls = foliage_controls(**(foliage or {}))
    k, confidence, bad = (0, 0, 0) if evidence is None else np.moveaxis(evidence, -1, 0)
    active = controls["foliage_strength"] > 0 or controls["fallback_strength"] > 0
    if active:
        sigma = sigma+(np.maximum(sigma, controls["fallback_sigma"])-sigma)*np.asarray(bad)[..., None]
    cs, sn = np.cos(params[..., 2]), np.sin(params[..., 2])
    logits, samples = [], []
    for oy in range(-(model.taps//2), model.taps//2+1):
        for ox in range(-(model.taps//2), model.taps//2+1):
            tap = centre+[ox, oy]
            delta = tap-xy
            u = (cs*delta[..., 0]+sn*delta[..., 1])/sigma[..., 0]
            v = (-sn*delta[..., 0]+cs*delta[..., 1])/sigma[..., 1]
            valid = np.all((tap >= 0) & (tap < [w, h]), -1)
            logits.append(np.where(valid, -.5*(u*u+v*v), -np.inf))
            samples.append(lr[tap[..., 1].clip(0, h-1), tap[..., 0].clip(0, w-1)])
    logits = np.stack(logits)
    weights = np.exp(logits-logits.max(0))
    weights /= weights.sum(0)
    current = (weights[..., None]*samples).sum(0)
    lo, hi = taps(lr).min(0), taps(lr).max(0)
    ix, iy = centre[..., 0], centre[..., 1]
    lo, hi = lo[iy, ix], hi[iy, ix]
    def sigmoid(x):
        e = np.exp(-np.abs(x))
        return np.where(x >= 0, 1/(1+e), e/(1+e))
    slack = sigmoid(params[..., 4:5])
    rectified = hist.clip(lo-slack*(hi-lo), hi+slack*(hi-lo))
    alpha = sigmoid(params[..., 3:4])
    if model.proximity:
        r2 = (((centre-xy)*2)**2).sum(-1)[..., None]
        term = np.log(model.proximity_gain)-r2/(2*model.proximity**2)
        if active:
            term = term*(1-np.asarray(bad)[..., None])
        alpha = sigmoid(params[..., 3:4]+term)
    if active:
        alpha = alpha+(np.maximum(alpha, controls["fallback_alpha"])-alpha)*np.asarray(bad)[..., None]
        spatial = np.zeros_like(current)
        norm = np.zeros_like(alpha)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tap = centre+[dx, dy]
                weight = np.exp(-((tap-xy)**2).sum(-1)/(2*.65*.65))
                weight *= np.all((tap >= 0) & (tap < [w, h]), -1)
                spatial += weight[..., None]*lr[tap[..., 1].clip(0, h-1), tap[..., 0].clip(0, w-1)]
                norm += weight[..., None]
        current = current+(spatial/np.maximum(norm, 1e-8)-current)*np.asarray(k*controls["foliage_spatial"]*(1-bad))[..., None]
        target = np.minimum(alpha, np.maximum(controls["foliage_alpha_floor"], alpha*controls["foliage_history_scale"]))
        alpha = alpha+(target-alpha)*np.asarray(k*confidence*(1-bad))[..., None]
    alpha = np.where(reset[..., None], 1, alpha)
    y, co = params[..., 5], params[..., 6]
    residual = .05*np.tanh(np.stack((y+co, y, y-co), -1)) if model.residual else 0
    value = (alpha*current+(1-alpha)*rectified+residual).clip(0, 1-1e-6)
    if active:
        value = np.where((reset & (bad > 0))[..., None], current, value)
    legacy = stabilize_value(value, rectified, lo, hi, reset, stabilize, stabilize_tau,
                             stabilize_eps, invalid, first_frame)
    return np.where(((np.asarray(k) > 0) | (np.asarray(bad) > 0))[..., None], value, legacy) if active else legacy


def check_foliage_tiles():
    rng = np.random.default_rng(210)
    for h, w in ((1, 1), (5, 7), (9, 17)):
        lr = rng.uniform(0, 1, (h, w, 3)).astype(np.float16).astype(np.float32)
        yy = yc(lr)[..., 0]
        tracker = rng.uniform(0, 1, (h, w, 4)).astype(np.float32)
        depth = rng.uniform(.1, 1, (h, w)).astype(np.float32)
        for gy in range(0, h, 8):
            for gx in range(0, w, 8):
                y12 = np.clip(gy-2+np.arange(12), 0, h-1)
                x12 = np.clip(gx-2+np.arange(12), 0, w-1)
                luma_tile = yy[y12[:, None], x12[None]]
                y10 = np.clip(gy-1+np.arange(10), 0, h-1)
                x10 = np.clip(gx-1+np.arange(10), 0, w-1)
                state_tile = np.stack((tracker[..., 2], depth), -1)[y10[:, None], x10[None]]
                for y in range(gy, min(gy+8, h)):
                    for x in range(gx, min(gx+8, w)):
                        for dy, dx in itertools.product(range(-1, 2), repeat=2):
                            py, px = np.clip(y+dy, 0, h-1), np.clip(x+dx, 0, w-1)
                            np.testing.assert_array_equal(state_tile[y-gy+1+dy, x-gx+1+dx], [tracker[py, px, 2], depth[py, px]])
                            for jitter in ((-.5, .5), (.5, -.5), (2.5, -1.5)):
                                xy = np.array([x+dx, y+dy])+jitter
                                ip = np.floor(xy).astype(int)
                                t = xy-ip
                                total = 0
                                for j, i in itertools.product(range(2), repeat=2):
                                    px, py = (ip+[i, j]).clip(0, [w-1, h-1])
                                    lx, ly = px-(gx-2), py-(gy-2)
                                    sample = luma_tile[ly, lx] if 0 <= lx < 12 and 0 <= ly < 12 else yy[py, px]
                                    total += sample*(t[0] if i else 1-t[0])*(t[1] if j else 1-t[1])
                                expected = linear_sample(yy[..., None], xy[None, None])[0, 0, 0]
                                np.testing.assert_allclose(total, expected, atol=1e-7)
    print("PASS: foliage shared luma/tracker tiles, borders, odd sizes, partial groups and arbitrary jitter")


def check_foliage():
    rng = np.random.default_rng(381)
    reductions = []
    for stride, ntaps, filt in itertools.product((1, 2), (3, 5), ("bicubic", "catmull")):
        h, w = 7, 9
        model = KPNAccumulator((h, w), (2*h, 2*w), widths=(3,)*6, trunk_stride=stride,
                               taps=ntaps, history_filter=filt, sigma_min=.05, residual=False)
        c = foliage_controls(foliage_strength=1)
        params = np.zeros((2*h, 2*w, 7), np.float32)
        params[..., :2] = np.log(.15)
        params[..., 4] = 8
        state, baseline_state = model.init_state(), model.init_state()
        tracker = np.full((h, w, 4), np.nan, np.float32)
        motion = np.zeros((h, w, 2), np.float32)
        depth = np.full((h, w), .5, np.float32)
        traces = [[], []]
        for frame in range(64):
            lr = np.full((h, w, 3), .5, np.float32)
            lr[:, w//2] += .025 if frame % 2 else -.025
            lr = lr.astype(np.float16).astype(np.float32)
            packed, hist, reset, age, tracker, confidence = pack(model, state, lr, motion, depth, (0, 0), tracker, c)
            invalid = packed[:h, :w, 11] > .5
            evidence = foliage_evidence(lr, tracker, confidence, depth, invalid, c, frame == 0)
            value = resolve(model, lr, hist, params, reset, (0, 0), foliage=c, evidence=evidence)
            base_pack, base_hist, base_reset, base_age = pack(model, baseline_state, lr, motion, depth, (0, 0))
            base = resolve(model, lr, base_hist, params, base_reset, (0, 0))
            # Disabled controls ignore even stale/poisoned evidence, preserving legacy stabilization too.
            legacy = resolve(model, lr, hist, params, reset, (0, 0), stabilize=.6)
            np.testing.assert_array_equal(resolve(model, lr, hist, params, reset, (0, 0), stabilize=.6,
                                                   foliage=foliage_controls(), evidence=np.full_like(evidence, np.nan)), legacy)
            for st, output, next_age in ((state, value, age), (baseline_state, base, base_age)):
                st.color = torch.from_numpy(output.astype(np.float16).astype(np.float32)).permute(2, 0, 1)[None]
                st.depth.fill_(.5)
                st.age = torch.from_numpy(next_age.astype(np.float32))[None, None]
                st.frame_index += 1
            traces[0].append(value[h, w-1, 0])
            traces[1].append(base[h, w-1, 0])
            assert np.isfinite(value).all() and np.isfinite(tracker).all()
        assert tracker[:, w//2, 2].mean() > .2
        ratio = np.abs(np.diff(traces[0][-16:])).mean()/np.abs(np.diff(traces[1][-16:])).mean()
        assert ratio <= .5, (stride, ntaps, filt, ratio)
        reductions.append(1-ratio)
        # A broad one-step shading change clears the accumulated instability immediately.
        for _ in range(2):
            lr = np.full((h, w, 3), .8, np.float32).astype(np.float16).astype(np.float32)
            packed, hist, reset, age, tracker, confidence = pack(model, state, lr, motion, depth, (0, 0), tracker, c)
            evidence = foliage_evidence(lr, tracker, confidence, depth, packed[:h, :w, 11] > .5, c)
            out = resolve(model, lr, hist, params, reset, (0, 0), foliage=c, evidence=evidence)
            np.testing.assert_allclose(out, lr.repeat(2, 0).repeat(2, 1), atol=1e-6)
            assert not evidence[..., 0].any()
            state.color = torch.from_numpy(out.astype(np.float32)).permute(2, 0, 1)[None]
        # Invalid / reset A bypass; standalone and combined D widen only untrusted detail.
        lr = rng.uniform(.3, .7, (h, w, 3)).astype(np.float16).astype(np.float32)
        invalid = np.zeros((h, w), bool)
        invalid[:, :3] = True
        tracker = np.full((h, w, 4), .8, np.float32)
        G = np.ones((h, w), np.float32)
        G[:, 4:6] = 0
        reset = invalid.repeat(2, 0).repeat(2, 1)
        hist = np.full((2*h, 2*w, 3), .5, np.float32)
        base = resolve(model, lr, hist, params, reset, (0, 0))
        for strength in (0, 1):
            d = foliage_controls(foliage_strength=strength, fallback_strength=1)
            ev = foliage_evidence(lr, tracker, G, depth, invalid, d)
            assert np.all(ev[:, 6:8, 2] == 0) and np.all(ev[:, :6, 2] > 0)
            out = resolve(model, lr, hist, params, reset, (0, 0), foliage=d, evidence=ev)
            if strength == 0:
                np.testing.assert_array_equal(out[:, 6:8], base[:, 6:8])
            assert np.abs(out[:, :6]-base[:, :6]).max() > .01
        # Without confidence, A can prefilter but cannot extend history.
        no_history = foliage_controls(foliage_strength=1, foliage_spatial=0)
        ev = foliage_evidence(lr, tracker, np.zeros_like(G), depth, invalid, no_history)
        np.testing.assert_array_equal(resolve(model, lr, hist, params, reset, (0, 0),
                                              foliage=no_history, evidence=ev), base)
        # Fallback never narrows a broader learned kernel.
        broad = params.copy()
        broad[..., :2] = 0
        hard_reset = np.ones_like(reset)
        d = foliage_controls(fallback_strength=1)
        ev = foliage_evidence(lr, tracker, np.zeros_like(G), depth, invalid, d)
        np.testing.assert_array_equal(resolve(model, lr, hist, broad, hard_reset, (0, 0), foliage=d, evidence=ev),
                                      resolve(model, lr, hist, broad, hard_reset, (0, 0)))
        for restart in (False, True):
            ev = foliage_evidence(lr, tracker, G, depth, invalid, c, restart)
            out = resolve(model, lr, hist, params, reset, (0, 0), foliage=c, evidence=ev)
            mask = np.ones_like(reset) if restart else reset
            np.testing.assert_array_equal(out[mask], base[mask])
    # A new thin-feature step is not an alternating event, including on its second frame.
    lr = np.full((7, 9, 3), .5, np.float32)
    depth = np.full((7, 9), .5, np.float32)
    motion = np.zeros((7, 9, 2), np.float32)
    invalid = np.zeros((7, 9), bool)
    c = foliage_controls(foliage_strength=1)
    tracker, _ = pack_tracker(lr, None, motion, depth, depth, invalid, (0, 0), c, True)
    lr[:, 4] = .55
    for _ in range(2):
        tracker, G = pack_tracker(lr, tracker, motion, depth, depth, invalid, (0, 0), c)
        assert not tracker[..., 2].any()
        assert not foliage_evidence(lr, tracker, G, depth, invalid, c)[..., 0].any()
    # Tracker edge cases, nearest signed deltas, inconsistent motion, exposure steps, and odd/singleton extents.
    for h, w in ((1, 1), (1, 7), (5, 1), (5, 7)):
        c = foliage_controls(foliage_strength=1, fallback_strength=1)
        lr = np.full((h, w, 3), .5, np.float32)
        motion = np.zeros((h, w, 2), np.float32)
        depth = np.full((h, w), .5, np.float32)
        invalid = np.zeros((h, w), bool)
        old = np.full((h, w, 4), np.nan, np.float32)
        for jitter in ((-.5, .5), (0, 0), (.25, -.25), (2.5, -1.5)):
            seeded, G = pack_tracker(lr, old, motion, depth, depth, invalid, jitter, c, True)
            assert np.isfinite(seeded).all() and not seeded[..., 1:3].any() and not G.any()
            nxt, G = pack_tracker(lr, seeded, motion, depth, depth, invalid, jitter, c)
            assert not nxt[..., 1:3].any()
            np.testing.assert_array_equal(G, np.ones_like(G))
            ev = foliage_evidence(lr, nxt, G, depth, invalid, c)
            assert not ev[..., (0, 2)].any()
            for bad_motion, bad_depth, bad_invalid in ((motion+2, depth, invalid), (motion, depth*.1, invalid), (motion, depth, ~invalid)):
                nxt, G = pack_tracker(lr, seeded, bad_motion, depth, bad_depth, bad_invalid, jitter, c)
                assert not nxt[..., 1:3].any() and not G.any()
        if w > 1:
            motion[:, :w//2, 0] = 1/w
            _, G = pack_tracker(lr, seeded, motion, depth, depth, invalid, (0, 0), c)
            assert G[:, w//2].max() < 1
        # Uniform camera motion alone retains confidence on in-bounds samples.
        motion[:] = (0.1/w, 0)
        _, G = pack_tracker(lr, seeded, motion, depth, depth, invalid, (0, 0), c)
        np.testing.assert_allclose(G, 1)
        # Opposite neighboring signed deltas must not cancel through bilinear reprojection.
        old = seeded.copy()
        old[..., 1] = np.where(np.indices((h, w))[1] % 2, -.1, .1)
        old[..., 0] = .55
        nxt, _ = pack_tracker(lr, old, np.zeros_like(motion), depth, depth, invalid, (0, 0), c)
        assert np.all(nxt[..., 2][:, ::2] > 0)
        assert np.all(nxt[..., 2][:, 1::2] == 0)
    print(f"PASS: foliage pack/resolve, all stride/taps/history variants; minimum alternating-feature oscillation reduction {min(reductions):.1%}; step/reset/invalid/flat/motion/odd-size cases")


def check_stabilizer():
    history = np.full((1, 2, 3), .5, np.float32)
    value = history+np.array([.01, -.01], np.float32)[None, :, None]
    lo, hi = history-.1, history+.1
    reset = np.zeros((1, 2), bool)
    for s in (.25, .6, .95):
        damped = stabilize_value(value, history, lo, hi, reset, s)
        reduction = 1-np.abs(damped-history)/np.abs(value-history)
        np.testing.assert_allclose(reduction, s*(1-.01/.104), atol=4e-6)
        np.testing.assert_allclose(reduction, s, atol=.1*s)
    np.testing.assert_array_equal(stabilize_value(value, history, lo, hi, reset), value)
    step = history+np.float32(.3)
    np.testing.assert_array_equal(stabilize_value(step, history, lo, hi, reset, .95), step)
    np.testing.assert_array_equal(stabilize_value(value, history, lo, hi, ~reset, .95), value)
    np.testing.assert_array_equal(stabilize_value(value, history, lo, hi, reset, .95, first_frame=True), value)
    np.testing.assert_array_equal(stabilize_value(value, history, lo, hi, reset, .95, invalid=~reset), value)
    mixed = np.array([[True, False]])
    damped = stabilize_value(value, history, lo, hi, mixed, .6)
    np.testing.assert_array_equal(damped[:, 0], value[:, 0])
    assert np.all(np.abs(damped[:, 1]-history[:, 1]) < np.abs(value[:, 1]-history[:, 1]))
    flat = stabilize_value(value, history, history, history, reset, .95)
    np.testing.assert_array_equal(flat, value)
    assert np.isfinite(stabilize_value(history, history, history, history, reset, .95)).all()
    print("PASS: fp32 stabilizer zero identity, oscillation damping, steps, reset/invalid/first-frame bypass and zero contrast")


def check_stabilizer_variants():
    rng = np.random.default_rng(191)
    h, w = 5, 7
    lr = rng.uniform(.4, .6, (h, w, 3)).astype(np.float32)
    motion = np.zeros((h, w, 2), np.float32)
    depth = np.full((h, w), .5, np.float32)
    depth[:3, :3] = .8
    params = np.zeros((2*h, 2*w, 7), np.float32)
    for stride, ntaps, history_filter, soft in itertools.product((1, 2), (3, 5), ("bicubic", "catmull"), (False, True)):
        model = KPNAccumulator((h, w), (2*h, 2*w), widths=(3,)*6, trunk_stride=stride,
                               taps=ntaps, history_filter=history_filter, depth_soft=soft)
        state = model.init_state()
        state.frame_index = 1
        state.color.fill_(.5)
        state.depth.fill_(.5)
        packed, hist, reset, _ = pack(model, state, lr, motion, depth, (0, 0))
        tensor = torch.from_numpy(packed).permute(2, 0, 1)[None]
        raw = F.pixel_unshuffle(tensor, stride).contiguous().numpy().ravel()
        hp, wp = packed.shape[:2]
        yy, xx = np.indices((h, w))
        phase = (yy % stride)*stride+xx % stride
        at = (11*stride*stride+phase)*(wp//stride)*(hp//stride)+(yy//stride)*(wp//stride)+xx//stride
        invalid_lr = raw[at] > .5
        np.testing.assert_array_equal(invalid_lr, packed[:h, :w, 11] > .5)
        invalid = invalid_lr.repeat(2, 0).repeat(2, 1)
        assert invalid.any() and (~invalid).any()
        if soft:
            assert np.any(invalid & ~reset)
        baseline = resolve(model, lr, hist, params, reset, (0, 0))
        c = foliage_controls(foliage_strength=1, fallback_strength=1)
        tracked = pack(model, state, lr, motion, depth, (0, 0), np.full((h, w, 4), .8, np.float32), c)
        next_tracker, confidence = tracked[-2:]
        assert not next_tracker[invalid_lr, 1:3].any() and not confidence[invalid_lr].any()
        ev = foliage_evidence(lr, next_tracker, confidence, depth, invalid_lr, c)
        assert not ev[invalid, 0].any() and np.all(ev[invalid, 2] > 0)
        a_only = foliage_controls(foliage_strength=1)
        ev[..., 2] = 0
        a_output = resolve(model, lr, hist, params, reset, (0, 0), foliage=a_only, evidence=ev)
        np.testing.assert_array_equal(a_output[invalid | reset], baseline[invalid | reset])
        filtered = resolve(model, lr, hist, params, reset, (0, 0), stabilize=.6, invalid=invalid)
        np.testing.assert_array_equal(filtered[invalid | reset], baseline[invalid | reset])
        assert np.any(np.abs(filtered[~invalid]-baseline[~invalid]) > 1e-5)
        np.testing.assert_array_equal(resolve(model, lr, hist, params, reset, (0, 0),
                                              stabilize=0, invalid=invalid), baseline)
    print("PASS: stabilizer resolve and packed invalid indexing, all stride/taps/history and hard/soft-depth variants")


def check_resample():
    rng = np.random.default_rng(21)
    for h, w in ((1, 1), (1, 7), (3, 1), (3, 5), (16, 32)):
        x = rng.normal(size=(h, w, 7)).astype(np.float32)
        expected = F.interpolate(torch.from_numpy(x).permute(2, 0, 1)[None], scale_factor=2,
                                 mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(bilinear(x), expected, atol=3e-7, rtol=1e-6)
    print("PASS: DirectML half-pixel resample emulation, singleton dimensions and borders")


@torch.no_grad()
def check_shaders():
    torch.manual_seed(27)
    worst = 0
    for case in range(64):
        h, w = ((1, 1), (7, 9), (17, 31), (63, 95))[case % 4]
        model = KPNAccumulator((h, w), (2*h, 2*w), widths=(3, 4, 5, 6, 7, 8),
                               lite=bool(case & 1), residual=bool(case & 2), mv_dilate=bool(case & 4),
                               depth_soft=bool(case & 8), jitter_sign=-1 if case & 2 else 1,
                               trunk_stride=2 if case & 32 else 1, taps=3 if case & 32 else 5,
                               history_filter="catmull" if case & 32 else "bicubic",
                               sigma_min=.05 if case & 16 else .3,
                               proximity=.35 if case & 16 else 0., proximity_gain=2.).eval()
        # Nontrivial parameters exercise sigma clamps, rotation, slack, alpha and both residual controls.
        model.trunk.head.bias.copy_(torch.linspace(-2, 2, model.trunk.head.out_channels))
        half = copy.deepcopy(model)
        half.trunk.half()
        half.trunk.register_forward_pre_hook(lambda m, args: (args[0].half(),))
        original_resolve = half.resolve
        half.resolve = lambda lr, history, *args: original_resolve(lr, history.half().float(), *args)
        state, half_state = model.init_state(), half.init_state()
        captured = {}
        def capture(module, inputs, output):
            captured["input"] = inputs[0][0].permute(1, 2, 0).numpy()
            captured["params"] = output[0, :, :2*h, :2*w].permute(1, 2, 0).numpy()
        hook = model.trunk.register_forward_hook(capture)
        for frame in range(6):
            if frame == 4:
                state, half_state = model.init_state(), half.init_state()
            if frame == 3:
                state.age.fill_(40)
                half_state.age.fill_(40)
            lr = (.05+.85*torch.rand(h, w, 3)).half().float()
            size = (2*h, 2*w) if case & 1 else (h, w)
            mv = torch.randn(*size, 2)*.03
            mv[0, :, 0] = -1 if frame == 2 else 0
            depth = torch.full(size, .5)
            depth[size[0]//2:, size[1]//2:] = .2 if frame % 2 else .8
            jitter = ((-.5, .5), (.5, -.5), (0, 0), (.25, -.25), (-.25, .25), (.1, -.2))[frame]
            packed, hist, reset, age = pack(model, state, lr.numpy(), mv.numpy(), depth.numpy(), jitter)
            out, state = model(state, lr, mv, depth, jitter)
            np.testing.assert_allclose(packed, captured["input"], atol=3e-5, rtol=2e-5)
            np.testing.assert_array_equal(reset, model._last_reset[0, 0].numpy())
            np.testing.assert_array_equal(age, state.age[0, 0].numpy())
            resolved = resolve(model, lr.numpy(), hist, captured["params"], reset, jitter)
            np.testing.assert_array_equal(resolve(model, lr.numpy(), hist, captured["params"], reset, jitter,
                                                  stabilize=0), resolved)
            invalid = (packed[:h, :w, 11] > .5).repeat(2, 0).repeat(2, 1)
            stabilized = resolve(model, lr.numpy(), hist, captured["params"], reset, jitter,
                                 stabilize=.6, invalid=invalid, first_frame=frame in (0, 4))
            blocked = np.ones_like(reset) if frame in (0, 4) else reset | invalid
            np.testing.assert_array_equal(stabilized[blocked], resolved[blocked])
            np.testing.assert_allclose(resolved, out.numpy(), atol=4e-6, rtol=2e-5)
            half_state.color = half_state.color.half().float()
            rounded, half_state = half(half_state, lr, mv, depth, jitter)
            worst = max(worst, float((rounded.half().float()-out).abs().max()))
        hook.remove()
    assert worst <= 3e-3, worst
    print(f"PASS: KPN pack/resolve emulation, 384 frames; CPU fp16-trunk/output drift max {worst:.3e} (not GPU validation)")


@torch.no_grad()
def check_trunk(model):
    stride = model.trunk_stride
    x = torch.randn(1, 16, 16*stride, 32*stride)
    expected = model.trunk(x)
    if stride == 2:
        # Pack writes c*4 + dy*2 + dx directly into the half-resolution NCHW tensor.
        direct = torch.empty(1, 64, 16, 32)
        for c in range(16):
            for dy in range(2):
                for dx in range(2):
                    direct[:, c*4+dy*2+dx] = x[:, c, dy::2, dx::2]
        assert torch.equal(direct, F.pixel_unshuffle(x, 2))
        x = direct
    parameters = dict(model.named_parameters())
    def conv(x, name, relu=False):
        weight, bias = parameters[name+".weight"], parameters[name+".bias"]
        result = F.conv2d(x, weight, bias, padding=weight.shape[-1]//2)
        return result.relu() if relu else result
    skips = []
    for i in range(5):
        if i:
            n, c, h, w = x.shape
            pooled = x.numpy().reshape(n, c, h//2, 2, w//2, 2).max((3, 5))
            x = torch.from_numpy(pooled)
        x = conv(x, f"trunk.enc.{i}", True)
        skips.append(x)
    x = conv(x, "trunk.bottleneck", True)
    for i in range(4):
        skip = skips[3-i]
        if skip.shape[1] != x.shape[1]:
            skip = conv(skip, f"trunk.skip.{i}")
        x = torch.from_numpy(bilinear(x[0].permute(1, 2, 0).numpy())).permute(2, 0, 1)[None]
        x = conv(x+skip, f"trunk.dec.{i}.0", True)
        if not model.lite:
            x = conv(x, f"trunk.dec.{i}.2", True)
    x = conv(x, "trunk.head")
    n, c, h, w = x.shape
    # DML COLUMN_ROW_DEPTH: [N, C, block_y, block_x, H, W].
    block = 2*stride
    direct = torch.empty(n, 7, block*h, block*w)
    for ch in range(7):
        for dy in range(block):
            for dx in range(block):
                direct[:, ch, dy::block, dx::block] = x[:, ch*block*block+dy*block+dx]
    assert torch.equal(direct, F.pixel_shuffle(x, block))
    x = direct
    torch.testing.assert_close(x, expected, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def write_pixels(model, f):
    sizes = ((1, 1), (3, 5), (7, 9))
    jitters = ((-.5, .5), (0., 0.), (.25, -.25), (.1, -.2),
               (-.5, -.5), (.5, .5), (-.25, -.25), (.25, .25))
    f.write(struct.pack("<I", len(sizes)*len(jitters)))
    for h, w in sizes:
        m = KPNAccumulator((h, w), (2*h, 2*w), widths=(3,)*6, residual=model.residual,
                           taps=model.taps, trunk_stride=model.trunk_stride, history_filter=model.history_filter,
                           sigma_min=model.sigma_min, proximity=model.proximity,
                           proximity_gain=model.proximity_gain, jitter_sign=model.jitter_sign)
        for jitter in jitters:
            lr = torch.rand(1, 3, h, w).half().float()
            history = torch.rand(1, 3, 2*h, 2*w)*1.5-.25
            params = (torch.randn(1, 7, 2*h, 2*w)*2).half().float()
            params[:, :2, ::2] = -100
            params[:, :2, 1::2] = 100
            params[:, 3, :, ::2] = 10
            params[:, 3, 0, 0] = 100
            params[:, 3, 1, 1] = -100
            reset = torch.rand(1, 1, 2*h, 2*w) > .75
            reset[..., 0, 0] = False
            reset[..., 1, 1] = False
            signed = torch.tensor(jitter)*m.jitter_sign
            weights, centre = m._kernel(params, signed)
            alpha = params[:, 3:4].sigmoid()
            if m.proximity:
                xy = m._sample_xy+signed
                r2 = ((centre-xy)*2).square().sum(-1)[:, None]
                alpha = (params[:, 3:4]+np.log(m.proximity_gain)-r2/(2*m.proximity**2)).sigmoid()
            alpha = torch.where(reset, 1., alpha)
            assert torch.isfinite(alpha).all()
            if m.proximity == .35 and tuple(signed.tolist()) == (-.5, -.5):
                assert abs(float(alpha[0, 0, 0, 0])-1) <= 1e-6
            if tuple(signed.tolist()) == (-.25, -.25):
                assert float(alpha[0, 0, 1, 1]) <= 1e-6
            y, co = params[:, 5:7].chunk(2, 1)
            residual = .05*torch.cat((y+co, y, y-co), 1).tanh() if m.residual else 0.
            out = m.resolve(lr, history, alpha, weights, centre, params[:, 4:5].sigmoid(), residual, reset)
            assert torch.isfinite(out).all()
            arrays = [t[0].permute(1, 2, 0).numpy() for t in (lr, history, params)]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                emulated = resolve(m, *arrays, reset[0, 0].numpy(), jitter)
            np.testing.assert_allclose(emulated, out[0].permute(1, 2, 0).numpy(), atol=4e-6, rtol=2e-5)
            f.write(struct.pack("<IIdd", h, w, *jitter))
            for tensor in (lr, history, params, reset.float(), alpha, out):
                f.write(tensor[0].permute(1, 2, 0).contiguous().numpy().astype("<f4").tobytes())
            rng = np.random.default_rng(h*100+w)
            evidence = rng.uniform(0, 1, (2*h, 2*w, 3)).astype(np.float32)
            evidence[reset[0, 0].numpy(), 0] = 0
            for mode in range(4):
                c = foliage_controls(foliage_strength=float(mode & 1), fallback_strength=float(mode >> 1))
                ev = evidence.copy()
                ev[..., 0] *= c["foliage_strength"]
                ev[..., 2] *= c["fallback_strength"]
                expected = resolve(m, *arrays, reset[0, 0].numpy(), jitter, foliage=c, evidence=ev)
                f.write(ev.astype("<f4").tobytes())
                f.write(expected.astype("<f4").tobytes())
    rng = np.random.default_rng(811)
    f.write(struct.pack("<I", 128))
    for case in range(128):
        L, B, R, old_B, spread = rng.uniform(0, 1, 5).astype(np.float32)
        if case % 2:
            old_B = B+np.float32(.04)
        old = rng.uniform(-.5, .5, 4).astype(np.float16).astype(np.float32)
        invalid = case % 5 == 0
        c = foliage_controls()
        d = (L-B)-(old[0]-old[3])
        agreement = 1-smoothstep(.03, .06, np.abs(B-old_B))
        event = np.clip(min(abs(d), abs(old[1]))/(R+c["foliage_eps"]), 0, 1) if d*old[1] < -c["foliage_eps"]**2 else 0
        I = (old[2]+(event-old[2])*c["foliage_ema"])*agreement
        state = np.array([L, d if agreement > 0 and not invalid else 0, I if not invalid else 0, B], np.float16).astype(np.float32)
        G = agreement*np.clip(1-spread/1.5, 0, 1) if not invalid else 0
        f.write(np.array([L, B, R, *old, old_B, spread, float(invalid), *state, G], "<f4").tobytes())


def check_files(root, binary):
    cases = ((False, None, {}), (True, None, {}), (False, (3,)*6, {}),
             (True, (3, 4, 5, 6, 7, 8), {}),
             (False, (3,)*6, dict(sigma_min=.05)),
             (True, (3,)*6, dict(proximity=.35)),
             (False, (3,)*6, dict(sigma_min=.05, proximity=.35)),
             (True, (3,)*6, dict(sigma_min=.05, proximity=.35, proximity_gain=3., residual=False)),
             (False, (3,)*6, dict(proximity=.001)))
    cases += tuple((lite, (3, 4, 5, 6, 7, 8), dict(trunk_stride=stride, taps=taps,
                    history_filter=history, sigma_min=.05, proximity=.4))
                   for lite in (False, True) for stride in (1, 2)
                   for taps in (3, 5) for history in ("bicubic", "catmull"))
    for index, (lite, widths, options) in enumerate(cases):
        model = KPNAccumulator((7, 9), (14, 18), lite=lite, widths=widths, depth_soft=True,
                               jitter_sign=1 if index & 1 else -1, **options).eval()
        check_trunk(model)
        path = root / "kpn.bin"
        tensors = export(model, path)
        lines = subprocess.run([binary, path], check=True, capture_output=True, text=True).stdout.splitlines()
        assert len(lines) == len(tensors)
        for line in lines:
            name, count, total, weighted = line.split()
            v = tensors[name].double().flatten()
            assert tensors[name].dtype == torch.float16 and int(count) == v.numel()
            assert abs(float(total)-v.sum().item()) < 1e-8
            assert abs(float(weighted)-(v*torch.arange(1, v.numel()+1)).sum().item()) < 1e-7
        reference = root / "kpn.ref"
        with reference.open("wb") as f:
            f.write(struct.pack("<IIIIf", model.lite, model.residual, model.mv_dilate, model.depth_soft, model.jitter_sign))
            for dims in (model.widths, model.render_size, model.output_size, model.padded_size):
                f.write(struct.pack("<"+"I"*len(dims), *dims))
            f.write(struct.pack("<fff", model.sigma_min, model.proximity, model.proximity_gain))
            f.write(struct.pack("<III", model.trunk_stride, model.taps, model.history_filter == "catmull"))
            for jitter in ((-.5, .5), (0, 0), (.1, -.2)):
                f.write(struct.pack("<ddff", *jitter, *(j*model.jitter_sign for j in jitter)))
            write_pixels(model, f)
        subprocess.run([binary, path, reference], check=True, capture_output=True)
        valid = path.read_bytes()
        n = struct.unpack_from("<I", valid, 8)[0]
        cfg = json.loads(valid[12:12+n])
        assert (cfg["sigma_min"], cfg["proximity"], cfg["proximity_gain"]) == (
            model.sigma_min, model.proximity, model.proximity_gain)
        if not options:
            # Old KPN files omit all three keys; each optional key also defaults independently.
            for missing in (("sigma_min", "proximity", "proximity_gain", "trunk_stride", "taps", "history_filter"), ("sigma_min",),
                            ("proximity",), ("proximity_gain",)):
                legacy = {k: v for k, v in cfg.items() if k not in missing}
                header = json.dumps(legacy).encode()
                path.write_bytes(valid[:8]+struct.pack("<I", len(header))+header+valid[12+n:])
                subprocess.run([binary, path, reference], check=True, capture_output=True)
        for bad in (dict(cfg, scale=3), dict(cfg, widths=[1]), dict(cfg, lite=1), dict(cfg, arch="oops"),
                    dict(cfg, trunk_stride=3), dict(cfg, taps=4), dict(cfg, history_filter="linear"),
                    dict(cfg, trunk_stride=True), dict(cfg, taps=3.5), dict(cfg, history_filter=1),
                    dict(cfg, jitter_sign=0), dict(cfg, residual="true"), dict(cfg, extra=1),
                    dict(cfg, output_size=[1, 1]), dict(cfg, padded_size=[7, 9]), dict(cfg, preset="unknown")):
            header = json.dumps(bad).encode()
            path.write_bytes(valid[:8]+struct.pack("<I", len(header))+header+valid[12+n:])
            assert subprocess.run([binary, path], capture_output=True).returncode != 0
        for key, values in (("sigma_min", (0, -.05, .000999999999, 2.500000001)),
                            ("proximity", (-.35, -1e-100, .000999999999, 16.000000001)),
                            ("proximity_gain", (0, -2, .000999999999, 64.000000001))):
            values += (1e-100, 1e-46, 1e-40, float("nan"), float("inf"), -float("inf"), 1e100, True, "0.3")
            for value in values:
                header = json.dumps(dict(cfg, **{key: value})).encode()
                path.write_bytes(valid[:8]+struct.pack("<I", len(header))+header+valid[12+n:])
                result = subprocess.run([binary, path], capture_output=True, text=True)
                assert result.returncode != 0, (key, value)
                if type(value) in (int, float) and np.isfinite(value):
                    assert key in result.stderr and "[1e-3," in result.stderr, result.stderr
        if index == 0:
            # Export and load every boundary, preserving exact zero as the disabled mode.
            for key, values in (("sigma_min", (1e-3, .05, .3, 2.5)),
                                ("proximity", (0., 1e-3, .35, .4, 16.)),
                                ("proximity_gain", (1e-3, 2., 64.))):
                for value in values:
                    boundary = copy.deepcopy(model)
                    setattr(boundary, key, value)
                    export(boundary, path)
                    raw = path.read_bytes()
                    size = struct.unpack_from("<I", raw, 8)[0]
                    assert json.loads(raw[12:12+size])[key] == value
                    subprocess.run([binary, path], check=True, capture_output=True)
        # Tensor rank/type/name corruption must be rejected independently of configuration.
        start = 12+n+4
        length = struct.unpack_from("<I", valid, start)[0]
        dtype_offset = start+4+length
        for offset, value in ((dtype_offset, 2), (dtype_offset+8, 99)):
            bad = bytearray(valid)
            struct.pack_into("<I", bad, offset, value)
            path.write_bytes(bad)
            assert subprocess.run([binary, path], capture_output=True).returncode != 0
        for bad in (valid[:-1], valid+b"!", valid.replace(b"trunk.head.bias", b"trunk.fake.bias")):
            path.write_bytes(bad)
            assert subprocess.run([binary, path], capture_output=True).returncode != 0
        check_sequence(model, root, binary, legacy=index == 0)
    print("PASS: KPN graphs, option/default parsing, native resolve, malformed files and fp32 sequences/resets")


def check_sequence(model, root, binary, legacy=False):
    checkpoint, sequence, weights = root/"kpn.pt", root/"kpn.seq", root/"kpn.weights.bin"
    torch.save(model.state_dict(), checkpoint)
    cfg = {k: v for k, v in model_config(model).items()
           if k not in ("scale", "preset", "render_size", "output_size", "padded_size")}
    if legacy:
        for key in ("sigma_min", "proximity", "proximity_gain", "trunk_stride", "taps", "history_filter"):
            del cfg[key]
        assert "_sigma_min" not in model.state_dict() and "_proximity" not in model.state_dict()
    save_sidecar(checkpoint, cfg)
    color = torch.rand(7, 9, 3)
    source = [(color, torch.zeros(7, 9, 2), torch.full((7, 9), .1234567), jitter)
              for jitter in ((.25, -.25), (-.25, .25), (.5, -.5), (0., 0.))]
    export_sequence(checkpoint, sequence, 4, weights, source, reset_frames=(2,))
    subprocess.run([binary, weights], check=True, capture_output=True)
    raw = sequence.read_bytes()
    w, h, frames, meta = struct.unpack_from("<IIII", raw, 8)
    assert (w, h, frames) == (9, 7, 4)
    metadata = json.loads(raw[24:24+meta])
    assert metadata["reference"].startswith("KPNAccumulator.forward fp32")
    assert metadata["config"] == model_config(model)
    offset, state = 24+meta, model.init_state()
    with torch.no_grad():
        for i, (lr, mv, depth, jitter) in enumerate(source):
            jx, jy, reset = struct.unpack_from("<ddI", raw, offset)
            offset += 20
            assert (jx, jy) == jitter and reset == (i in (0, 2))
            if reset:
                state = model.init_state()
            lr = (lr/(1+lr)).half().float()
            expected, state = model(state, lr, mv, depth, jitter)
            for tensor in (lr, mv, depth, expected):
                n = tensor.numel()
                got = np.frombuffer(raw, "<f4", n, offset).reshape(tensor.shape)
                np.testing.assert_array_equal(got, tensor.numpy())
                offset += 4*n
    assert offset == len(raw)



def catmull9(image, uv):
    """Ideal hardware bilinear: normalized texel centres, independently clamped taps."""
    h, w = image.shape[:2]
    size = np.array([w, h], np.float32)
    p = uv*size-.5
    ip, t = np.floor(p), p-np.floor(p)
    def axis(t, ip):
        t2, t3 = t*t, t*t*t
        weights = np.stack((-.5*t+t2-.5*t3, 1-2.5*t2+1.5*t3,
                            .5*t+2*t2-1.5*t3, -.5*t2+.5*t3), -1)
        middle = weights[..., 1]+weights[..., 2]
        positions = np.stack((ip-1, ip+weights[..., 2]/middle, ip+2), -1)
        return positions, np.stack((weights[..., 0], middle, weights[..., 3]), -1)
    px, wx = axis(t[..., 0], ip[..., 0])
    py, wy = axis(t[..., 1], ip[..., 1])
    def fetch(uv):
        pos = uv*size-.5
        base = np.floor(pos).astype(int)
        f = (pos-base).astype(np.float32)
        x, y = base[..., 0], base[..., 1]
        tx, ty = f[..., 0, None], f[..., 1, None]
        a = image[y.clip(0, h-1), x.clip(0, w-1)]
        b = image[y.clip(0, h-1), (x+1).clip(0, w-1)]
        c = image[(y+1).clip(0, h-1), x.clip(0, w-1)]
        d = image[(y+1).clip(0, h-1), (x+1).clip(0, w-1)]
        return (1-ty)*((1-tx)*a+tx*b)+ty*((1-tx)*c+tx*d)
    out = np.zeros((*uv.shape[:2], image.shape[-1]), np.float32)
    for j in range(3):
        row = np.zeros_like(out)
        for i in range(3):
            sample_uv = (np.stack((px[..., i], py[..., j]), -1)+.5)/size
            row += fetch(sample_uv)*wx[..., i, None]
        out += row*wy[..., j, None]
    return out


def check_catmull9():
    from fsrmamba.kpn_unet import catmull_sample
    rng = np.random.default_rng(837)
    worst = 0.
    for h, w in ((1, 1), (1, 7), (9, 1), (13, 17), (32, 48)):
        image = rng.random((h, w, 3), dtype=np.float32)
        uv = rng.uniform(-.2, 1.2, (29, 31, 2)).astype(np.float32)
        ref = warp(image, uv, True, True)
        actual = catmull9(image, uv)
        expected = catmull_sample(torch.from_numpy(image).permute(2, 0, 1)[None],
                                  torch.from_numpy(uv)[None])[0].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(actual, ref, atol=5e-6, rtol=2e-6)
        np.testing.assert_allclose(expected, ref, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(catmull9(image, grid(h, w)), image, atol=5e-6, rtol=0)
        worst = max(worst, float(np.max(np.abs(actual-ref))))
    print(f"PASS: nine bilinear Catmull-Rom fetches vs sixteen taps/PyTorch, borders/singletons; max {worst:.3e}")


def check_coordinate_decisions():
    # A reciprocal implementation of normalized-grid division can move exact half ties.
    extent = 720
    dst = np.arange(2*extent, dtype=np.float32)
    numerator = dst+.5
    divided = numerator/np.float32(2*extent)*np.float32(extent)-.5+.25
    reciprocal = numerator*(np.float32(1)/np.float32(2*extent))*np.float32(extent)-.5+.25
    changes = np.floor(divided+.5) != np.floor(reciprocal+.5)
    assert changes.any()
    # A nearest-centred min/max window can change discontinuously at such a tie,
    # even though depth, UV, age and all filter arithmetic remain fp32.
    pixel = int(np.flatnonzero(changes)[len(np.flatnonzero(changes))//2])
    a, b = int(np.floor(divided[pixel]+.5)), int(np.floor(reciprocal[pixel]+.5))
    assert abs(a-b) == 1
    signal = np.zeros(extent, np.float32)
    signal[max(a,b)+1] = .8
    limits = [signal[c-1:c+2].max() for c in (a, b)]
    alpha = 1/(1+np.exp(-(np.log(2)-1/(2*.4**2))))
    error = abs(np.clip(.15, 0, limits[0])-np.clip(.15, 0, limits[1]))*(1-alpha)
    assert error > .1
    direct = (dst+.5)*.5-.5+.25
    np.testing.assert_array_equal(np.floor(direct+.5), np.arange(2*extent)//2+np.arange(2*extent)%2)
    # Half rounding really can flip the depth gate, but KPN does not use it here.
    d = np.float32(.5)
    threshold = np.float32(.9)*d
    old = np.nextafter(threshold, np.float32(0))
    assert old < threshold
    assert not (np.float32(np.float16(old)) < np.float32(np.float16(np.float32(.9)*np.float32(np.float16(d)))))
    print(f"PASS: fp32 reciprocal tie regression: {int(changes.sum())} changed rows; constructed clamp error {error:.3e}; direct pixel coordinates stable")



def check_pack_motion_tile():
    rng = np.random.default_rng(819)
    for h, w in ((1, 1), (2, 3), (7, 9), (17, 31), (32, 48)):
        depth = rng.integers(0, 4, (h, w)).astype(np.float32)*.2+.1
        for display in (False, True):
            raw = rng.normal(size=(h*(2 if display else 1), w*(2 if display else 1), 2)).astype(np.float32)
            motion = ((raw[0::2,0::2]+raw[0::2,1::2]+raw[1::2,0::2]+raw[1::2,1::2])*.25
                      if display else raw)
            for dilate in (False, True):
                expected = np.take_along_axis(taps(motion), taps(depth[...,None]).argmax(0)[None], axis=0)[0] if dilate else motion
                for gy in range((h+31)//32*4):
                    for gx in range((w+31)//32*4):
                        origin = np.minimum([gx*8, gy*8], [w-1, h-1])
                        ys, xs = (np.arange(12)+origin[1]-2).clip(0,h-1), (np.arange(12)+origin[0]-2).clip(0,w-1)
                        ds, ms = depth[ys[:,None],xs[None]], motion[ys[:,None],xs[None]]
                        selected = np.empty((10,10,2), np.float32)
                        def source(p):
                            local = (p-(origin-2)).clip(0,11)
                            return local[1], local[0]
                        for y in range(10):
                            for x in range(10):
                                p = (origin-1+[x,y]).clip(0,[w-1,h-1])
                                best = source(p-1 if dilate else p)
                                near = ds[best]
                                if dilate:
                                    for dy in range(-1,2):
                                        for dx in range(-1,2):
                                            at = source(p+[dx,dy])
                                            if ds[at] > near:
                                                near, best = ds[at], at
                                selected[y,x] = ms[best]
                        for y in range(-1,9):
                            for x in range(-1,9):
                                p = (origin+[x,y]).clip(0,[w-1,h-1])
                                local = (p-(origin-1)).clip(0,9)
                                np.testing.assert_array_equal(selected[local[1],local[0]], expected[p[1],p[0]])
    print("PASS: KPN shared motion tile, depth ties, display averaging, borders and stride-2 padding")

def check_resolve_tile():
    rng = np.random.default_rng(182)
    for h, w in ((1, 1), (7, 9), (17, 31)):
        color = rng.random((h, w, 3), dtype=np.float32)
        for gy in range((h+7)//8):
            for gx in range((w+7)//8):
                origin = np.array([gx*8-3, gy*8-3])
                tile = color[(np.arange(14)+origin[1]).clip(0,h-1)[:,None],
                             (np.arange(14)+origin[0]).clip(0,w-1)[None]]
                for jx, jy in ((-.5,.5), (.5,-.5), (.25,-.25), (2.,-3.)):
                    for y in range(gy*8,min(gy*8+8,h)):
                        for x in range(gx*8,min(gx*8+8,w)):
                            for q in range(4):
                                xy = (np.array([2*x+q%2,2*y+q//2])+.5)*.5-.5+[jx,jy]
                                centre = np.floor(xy+.5).clip(0,[w-1,h-1]).astype(int)
                                for dy in range(-2,3):
                                    for dx in range(-2,3):
                                        tap = (centre+[dx,dy]).clip(0,[w-1,h-1])
                                        local = tap-origin
                                        value = tile[local[1],local[0]] if np.all((local>=0)&(local<14)) else color[tap[1],tap[0]]
                                        np.testing.assert_array_equal(value,color[tap[1],tap[0]])
    print("PASS: resolve shared color tile, partial groups, borders, phases and arbitrary-jitter fallback")

def run(root, binary):
    check_foliage_tiles()
    check_foliage()
    check_stabilizer()
    check_stabilizer_variants()
    check_catmull9()
    check_coordinate_decisions()
    check_pack_motion_tile()
    check_resolve_tile()
    check_resample()
    check_shaders()
    check_files(root, binary)
