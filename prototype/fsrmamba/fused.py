"""Two fused memory passes around the fast accumulator's convolutional trunk.

The reference functions deliberately use gather arithmetic for reprojection and
packing. CUDA colour storage is fp16; motion/jitter are fp32 and sample offsets
are int32. Runtime buffers and the trunk backend are owned by one FusedFast:
returned images/states are borrowed and overwritten by subsequent steps.
With carry_raw, history holds the accumulation before the display residual;
resolve always writes a separate fp16 display, even with write_rgb=False.
state_rgb returns that borrowed display. Clone them to retain frames. Weights are
snapshotted on first CUDA use; construct a new wrapper after changing weights.
"""
from __future__ import annotations

import copy
import io
import time

import torch
import torch.nn.functional as F

from .fast import (FastAccumulator, FastState, _nearest_depth, _thin_feature,
                   _coverage_evidence, _coverage_coefficient, _osc_reset)
from .resolve import phase_kernels


def _validate(model):
    if not isinstance(model, FastAccumulator):
        raise ValueError("FusedFast requires arch fast")
    checks = {"n_state": 0, "learned_clamp": False, "resolve": "nearest",
              "detail_ch": 0, "reset_lanczos": False}
    for name, value in checks.items():
        if getattr(model, name) != value:
            raise ValueError(f"FusedFast requires {name}={value!r}")
    if model.depth_soft and not model.depth_test:
        raise ValueError("depth_soft requires depth_test=True")
    if model.depth_soft_osc and not (model.depth_test and model.depth_soft and model.coverage):
        raise ValueError("depth_soft_osc requires depth_test, depth_soft and coverage")
    if model.carry_raw and (not model.accum or model.hist_residual):
        raise ValueError("carry_raw requires accum=True and hist_residual=False")
    if model.stem.kernel_size != (1, 1):
        raise ValueError("FusedFast requires stem_kernel=1")
    if model.hist_filter not in ("bilinear", "bicubic"):
        raise ValueError("FusedFast requires bilinear or bicubic history")
    s = model.scale
    if not isinstance(s, int) or s < 1 or tuple(v * s for v in model.render_size) != model.output_size:
        raise ValueError("FusedFast requires a positive integer uniform scale")


def _cubic(t):
    # PyTorch's cubic convolution coefficients, A = -0.75.
    def c1(x):
        return ((1.25 * x - 2.25) * x) * x + 1
    def c2(x):
        return ((-0.75 * x + 3.75) * x - 6) * x + 3
    return c2(t + 1), c1(t), c1(1 - t), c2(2 - t)


def _sample(src, x, y, cubic=False):
    """NCHW source, float32 pixel coordinates, independently clamped taps."""
    h, w = src.shape[-2:]
    if not cubic:
        x, y = x.clamp(0, w - 1), y.clamp(0, h - 1)
    ix, iy = x.floor().long(), y.floor().long()
    tx, ty = x - ix, y - iy
    wx = _cubic(tx) if cubic else (1 - tx, tx)
    wy = _cubic(ty) if cubic else (1 - ty, ty)
    first = -1 if cubic else 0
    result = 0
    for j, cy in enumerate(wy):
        row = 0
        for i, cx in enumerate(wx):
            tap = src[..., (iy + j + first).clamp(0, h - 1),
                      (ix + i + first).clamp(0, w - 1)]
            row = row + tap * cx
        result = result + row * cy
    return result


def _sample_nearest(src, x, y):
    """Border sampling rounds the unnormalised coordinates half to even."""
    h, w = src.shape[-2:]
    ix = x.clamp(0, w - 1).round().long()
    iy = y.clamp(0, h - 1).round().long()
    return src[..., iy, ix]


def _phase_gather(t, s):
    h, w = t.shape[-2] // s, t.shape[-1] // s
    return t.reshape(1, -1, h, s, w, s).permute(0, 1, 3, 5, 2, 4).reshape(1, -1, h, w)


def _phase_scatter(t, s):
    _, cp, h, w = t.shape
    return t.reshape(1, cp // (s * s), s, s, h, w).permute(0, 1, 4, 2, 5, 3).reshape(
        1, cp // (s * s), h * s, w * s)


def _pack_index(layout, ty, tx, c, Ht, Wt, C):
    """Physical element offset for the single-batch pack and resolve buffers."""
    if layout == "nhwc":
        return (ty * Wt + tx) * C + c
    if layout == "nchw":
        return (c * Ht + ty) * Wt + tx
    raise ValueError("layout must be 'nhwc' or 'nchw'")


def _signed_jitter(model, jitter):
    """The model's jitter convention, applied once at the public entry points."""
    sign = getattr(model, "jitter_sign", 1.0)
    if sign == 1.0:
        return jitter
    if torch.is_tensor(jitter):
        return jitter * sign
    return (float(jitter[0]) * sign, float(jitter[1]) * sign)


def _phase_weights(model, jitter, dtype, device):
    j = torch.as_tensor(jitter, device=device).float()
    dx = (0.5 - j[0] - model._acc_px + 0.5) % 1.0 - 0.5
    dy = (0.5 - j[1] - model._acc_py + 0.5) % 1.0 - 0.5
    return torch.exp(-model.acc_sharp.abs() * (dx * dx + dy * dy)).to(dtype).reshape(1, -1, 1, 1)


def _phase_offsets(model, jitter):
    jx, jy = float(jitter[0]), float(jitter[1])
    return [(round(py - 0.5 + jy), round(px - 0.5 + jx)) for py, px in model._ph_list]


def _base_constants(model, jitter):
    geometry = [getattr(model, name).detach().to(device="cpu", dtype=torch.float32)
                for name in ("_ph_x", "_ph_y", "_tap_x", "_tap_y")]
    j = torch.as_tensor(jitter).detach().to(device="cpu", dtype=torch.float32)
    _, kernels, _, _, _, windows = phase_kernels(j, "cpu", *geometry)
    return kernels.reshape(model.p, 16), torch.tensor(windows, dtype=torch.int32)


def ref_pack(model, state, lr_rgb, mv_lr, depth, jitter):
    """Slow executable specification of pack; no grid_sample/pixel_unshuffle."""
    _validate(model)
    h, w = model.render_size
    s, p = model.scale, model.p
    dt, dev = lr_rgb.dtype, lr_rgb.device
    lr = lr_rgb.permute(2, 0, 1)[None]
    d = depth[None, None].to(dt)
    stored_depth = d
    if model.mv_dilate:
        near, motion = _nearest_depth(d, mv_lr[None])
        mv_lr = motion[0]
        if model.depth_dilate:
            stored_depth = near
    elif model.depth_dilate:
        stored_depth = _nearest_depth(d)
    H, W = model.output_size
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    motion = _sample(mv_lr.permute(2, 0, 1)[None].float(),
                     (xx + 0.5) / s - 0.5, (yy + 0.5) / s - 0.5)[0]
    # Keep both normalisation operations: cancelling them changes fp32 rounding.
    gx = ((xx + 0.5) / W + motion[0]) * 2 - 1
    gy = ((yy + 0.5) / H + motion[1]) * 2 - 1
    sx, sy = ((gx + 1) * W - 1) / 2, ((gy + 1) * H - 1) / 2
    hist = _sample(state.color.float(), sx, sy, model.hist_filter == "bicubic").to(dt)
    y, x = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                          torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    u, v = (x + 0.5) / w + mv_lr[..., 0], (y + 0.5) / h + mv_lr[..., 1]
    reset = ((u < 0) | (u >= 1) | (v < 0) | (v >= 1))[None, None]
    reset_feature = None
    if model.depth_test:
        dgx = (u * 2 - 1).to(state.depth.dtype).float()
        dgy = (v * 2 - 1).to(state.depth.dtype).float()
        prev_d = _sample_nearest(state.depth, ((dgx + 1) * w - 1) / 2,
                                 ((dgy + 1) * h - 1) / 2)
        d_max = F.max_pool2d(d, 3, stride=1, padding=1)
        d_min = -F.max_pool2d(-d, 3, stride=1, padding=1)
        mismatch = (prev_d < d_min * 0.9) | (prev_d > d_max * 1.1)
        if model.depth_soft:
            reset_feature = (reset | mismatch).to(dt)
        else:
            reset = reset | mismatch
    reset = reset.to(dt)
    if state.frame_index == 0:
        reset.fill_(1)
        reset_feature = reset
    taps = [lr[..., (y.long() + dy).clamp(0, h - 1), (x.long() + dx).clamp(0, w - 1)]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    mn, mx = torch.stack(taps).amin(0), torch.stack(taps).amax(0)
    rng = mx - mn
    previous = None
    if model.depth_soft_osc:
        previous = torch.cat((state.m1, state.m2, state.osc, state.luma), 1)
        gx, gy = u * 2 - 1, v * 2 - 1
        previous = _sample_nearest(previous, ((gx + 1) * w - 1) / 2,
                                   ((gy + 1) * h - 1) / 2).chunk(4, 1)
        evidence, _ = _coverage_evidence(lr, rng, previous, reset)
        reset = _osc_reset(model, reset, mismatch, prev_d, d_max, evidence, training=False)
    slack = rng * model.box_slack.abs()
    thin = None
    if model.thin_lock:
        thin = _thin_feature(lr)
        factor = (1 + model.thin_slack).to(dt)
        slack = slack * torch.where(thin > .5, factor, torch.ones_like(thin))
    hist = _phase_gather(hist, s).reshape(1, 3, p, h, w)
    h_cl = torch.minimum(torch.maximum(hist, (mn - slack)[:, :, None]),
                         (mx + slack)[:, :, None]).reshape(1, 3 * p, h, w)
    motion_lr = mv_lr.float()
    speed_pixels = ((motion_lr[..., 0] * w).square() + (motion_lr[..., 1] * h).square()).sqrt()
    speed = (speed_pixels * 0.1).clamp(0, 1)[None, None].to(dt)
    parts = [lr, h_cl, rng, 1.0 / d.clamp(min=1e-2), speed, reset_feature if reset_feature is not None else reset]
    conf_w = None
    if model.accum:
        conf = _sample(state.conf.float(), sx, sy, model.hist_filter == "bicubic").to(dt).clamp(min=0)
        conf_w = _phase_gather(conf, s) * (1 - reset)
        if model.conf_motion:
            spd = speed_pixels.clamp(max=16)[None, None].to(dt)
            conf_w = conf_w / (1.0 + model.conf_m.abs() * spd)
        parts.append(torch.log1p(conf_w))
    if model.carry_raw:
        if model.nearest_sample:
            base_q = torch.stack([lr[..., (y.long() + ny).clamp(0, h - 1),
                                        (x.long() + nx).clamp(0, w - 1)]
                                  for ny, nx in _phase_offsets(model, jitter)], dim=2)
        else:
            base_q = lr.unsqueeze(2)
        raw_luma = 0.25 * hist[:, 0] + 0.5 * hist[:, 1] + 0.25 * hist[:, 2]
        base_luma = 0.25 * base_q[:, 0] + 0.5 * base_q[:, 1] + 0.25 * base_q[:, 2]
        luma_range = 0.25 * rng[:, 0:1] + 0.5 * rng[:, 1:2] + 0.25 * rng[:, 2:3]
        parts.append(((raw_luma - base_luma) / (luma_range + 0.02)).clamp(-4, 4) * (1 - reset))
    if model.thin_lock:
        parts.append(thin)
    coverage_state = (None,) * 4
    if model.coverage:
        if previous is None:
            previous = torch.cat((state.m1, state.m2, state.osc, state.luma), 1)
            gx, gy = u * 2 - 1, v * 2 - 1
            previous = _sample_nearest(previous, ((gx + 1) * w - 1) / 2,
                                       ((gy + 1) * h - 1) / 2).chunk(4, 1)
        evidence, coverage_state = _coverage_evidence(lr, rng, previous, reset)
        parts.append(evidence)
    age = new_age = None
    if model.history_age:
        gx, gy = u * 2 - 1, v * 2 - 1
        age = _sample_nearest(state.age, ((gx + 1) * w - 1) / 2,
                              ((gy + 1) * h - 1) / 2)
        age = torch.where(reset > .5, torch.zeros_like(age), age)
        parts.append((torch.log2(1 + age) / 5).to(dt))
        new_age = torch.where(reset > .5, torch.zeros_like(age), (age + 1).clamp(max=32))
    cat = torch.cat(parts, 1)
    hp, wp = h + model._pad[0], w + model._pad[1]
    # Each padded pixel owns one channel phase; its source is edge-clamped.
    yp = torch.arange(hp, device=dev).clamp(max=h - 1)
    xp = torch.arange(wp, device=dev).clamp(max=w - 1)
    padded = cat[..., yp[:, None], xp[None, :]]
    X = _phase_gather(padded, 2).contiguous(memory_format=torch.channels_last)
    return dict(X=X, age=age, new_age=new_age, coverage_state=coverage_state, depth=stored_depth, h_cl=h_cl.contiguous(), conf_w=conf_w, reset=reset, rng=rng,
                h_raw=hist.reshape(1, 3 * p, h, w).contiguous() if model.carry_raw else None)


def _fold_head(model, jitter, dtype, device):
    j = torch.as_tensor(jitter, device=device, dtype=dtype).reshape(1, 2)
    g, b = model.film(j).chunk(2, dim=1)
    weight = model.out.weight * (1 + g).reshape(1, -1, 1, 1)
    bias = model.out.bias + F.linear(b, model.out.weight[:, :, 0, 0])[0]
    return weight, bias


class _HostFrameCache:
    """Immutable pinned frame constants, computed once per jitter pair."""

    def __init__(self, model, *, pin_memory=False, capacity=64, fold=True):
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.fold = fold
        self.values = {}
        self.out_weight = model.out.weight.detach().to(device="cpu", dtype=torch.float32).clone()
        self.out_bias = model.out.bias.detach().to(device="cpu", dtype=torch.float32).clone()
        self.film = (None if model.film is None else
                     copy.deepcopy(model.film).to(device="cpu", dtype=torch.float32).eval())
        if self.film is not None:
            self.film.requires_grad_(False)
        self.accum = model.accum
        self.nearest_sample = model.nearest_sample
        self.base_gate = model.base_gate
        self.carry_raw = model.carry_raw
        self._ph_list = list(model._ph_list)
        if self.base_gate:
            for name in ("_ph_x", "_ph_y", "_tap_x", "_tap_y"):
                setattr(self, name, getattr(model, name).detach().to(
                    device="cpu", dtype=torch.float32).clone())
            self.p = model.p
        if self.accum:
            for name in ("acc_sharp", "_acc_px", "_acc_py"):
                setattr(self, name, getattr(model, name).detach().to(
                    device="cpu", dtype=torch.float32).clone())

    def _store(self, value, dtype=torch.float16):
        result = torch.empty_like(value, dtype=dtype, device="cpu",
                                  pin_memory=self.pin_memory)
        result.copy_(value)
        return result

    @torch.no_grad()
    def get(self, jitter):
        j = torch.as_tensor(jitter).detach().to(device="cpu", dtype=torch.float32).reshape(2)
        # Nearest sample ties depend on the original Python floats, before fp32 conversion.
        key = tuple(float(jitter[i]) for i in range(2)) if self.nearest_sample else tuple(j.tolist())
        if key in self.values:
            result = self.values.pop(key)
            self.values[key] = result
            return result
        weight = bias = None
        if self.fold:
            weight, bias = self.out_weight, self.out_bias
            if self.film is not None:
                g, b = self.film(j.reshape(1, 2)).chunk(2, dim=1)
                weight = weight * (1 + g).reshape(1, -1, 1, 1)
                bias = bias + F.linear(b, self.out_weight[:, :, 0, 0])[0]
            weight, bias = self._store(weight), self._store(bias)
        phase = (_phase_weights(self, j, torch.float32, "cpu").reshape(-1)
                 if self.accum else None)
        sample_jitter = j if self.base_gate and not self.carry_raw else key
        offsets = (self._store(torch.tensor(_phase_offsets(self, sample_jitter), dtype=torch.int32), torch.int32)
                   if self.nearest_sample else None)
        kernels = windows = None
        if self.base_gate:
            kernels, windows = _base_constants(self, j)
            kernels = self._store(kernels, torch.float32)
            windows = self._store(windows, torch.int32)
        result = weight, bias, None if phase is None else self._store(phase), offsets, kernels, windows
        if len(self.values) == self.capacity:
            self.values.pop(next(iter(self.values)))
        self.values[key] = result
        return result


def _shuffle_add_reference(up, skip):
    """Executable NHWC indexing specification of the in-place CUDA kernel."""
    _, c, H, W = skip.shape
    h, w = up.shape[-2:]
    assert (H, W) == (2 * h, 2 * w) and up.shape[:2] == (1, 4 * c)
    src = up.permute(0, 2, 3, 1).reshape(-1)
    t = torch.arange(H * W * c, device=up.device)
    channel, pixel = t % c, t // c
    x, y = pixel % W, pixel // W
    at = ((y // 2) * w + x // 2) * (4 * c) + channel * 4 + (y % 2) * 2 + x % 2
    skip.add_(src[at].reshape(1, H, W, c).permute(0, 3, 1, 2))
    return skip


def run_trunk(model, X, jitter, head=None):
    """Inference pyramid; pack supplies the stem input and FiLM folds into the head."""
    x = F.relu_(model.stem(X))
    skips = []
    for i, block in enumerate(model.enc):
        x = _run_enc(block, x)
        if i < len(model.down):
            skips.append(x)
            x = F.relu_(model.down[i](x))
    for i in range(len(model.down) - 1, -1, -1):
        x = F.pixel_shuffle(model.up[i](x), 2).add_(skips[i])
    x = F.relu_(model.fuse(x))
    if head is not None:
        return F.conv2d(x, *head)
    if model.film is None:
        return model.out(x)
    weight, bias = _fold_head(model, jitter, x.dtype, x.device)
    return F.conv2d(x, weight, bias)


def _run_enc(block, x):
    for residual in block:
        # Write into the fresh branch result; x may also be a saved skip.
        x = residual.b(F.relu_(residual.a(x))).add_(x)
    return x


class TrunkModule(torch.nn.Module):
    """Module entry point for optional trunk compilation, with resident packed X."""

    def __init__(self, model, weight=None, bias=None):
        super().__init__()
        self.model = model
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, X, jitter):
        head = None if self.weight is None else (self.weight, self.bias)
        return run_trunk(self.model, X, jitter, head)


class _TensorRTTrunkModule(torch.nn.Module):
    """NCHW pyramid with a runtime 1x1 head expressed as matrix multiplication."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, head_w, head_b):
        model = self.model
        x = F.relu(model.stem(x))
        skips = []
        for i, block in enumerate(model.enc):
            for residual in block:
                x = residual.b(F.relu(residual.a(x))) + x
            if i < len(model.down):
                skips.append(x)
                x = F.relu(model.down[i](x))
        for i in range(len(model.down) - 1, -1, -1):
            x = F.pixel_shuffle(model.up[i](x), 2) + skips[i]
        x = F.relu(model.fuse(x))
        h, w = x.shape[-2:]
        o = x.flatten(2).transpose(1, 2) @ head_w[:, :, 0, 0].T + head_b
        return o.transpose(1, 2).reshape(1, head_b.shape[0], h, w).contiguous()


def _state_views(history, feat, depth, frame_index, accum):
    color = history[..., :3].permute(2, 0, 1)[None]
    conf = history[..., 3:].permute(2, 0, 1)[None] if accum else None
    return FastState(color, feat, depth, frame_index, conf)


def materialize_state(state):
    """Copy borrowed state into independent contiguous NCHW storage."""
    new = FastState(state.color.clone(memory_format=torch.contiguous_format),
                    state.feat.clone(), state.depth.clone(), state.frame_index,
                    None if state.conf is None else
                    state.conf.clone(memory_format=torch.contiguous_format),
                    *(None if t is None else t.clone()
                      for t in (state.m1, state.m2, state.osc, state.luma, state.age)))
    if hasattr(state, "_display_rgb"):
        new._display_rgb = state._display_rgb.clone()
    return new


def state_rgb(state):
    """Return carry_raw's borrowed display, or materialise nonnegative history."""
    if hasattr(state, "_display_rgb"):
        return state._display_rgb
    return state.color[0].permute(1, 2, 0).clamp(min=0).contiguous()


def ref_resolve(model, o, lr_rgb, packed, jitter):
    """Resolve with explicit phase indexing; pack supplies the thin-aware clamp box."""
    h, w = model.render_size
    p, s = model.p, model.scale
    y, x = torch.meshgrid(torch.arange(h, device=o.device), torch.arange(w, device=o.device), indexing="ij")
    k = torch.arange(model.n_px, device=o.device)[:, None, None]
    px = o[0, k * 4 + (y % 2) * 2 + x % 2, y // 2, x // 2][None].to(lr_rgb.dtype)
    cov = (_coverage_coefficient(px[:, model._cov_offset:model._cov_offset + p], model.coverage_bias)
           if model.coverage else None)
    if model.nearest_sample:
        sample_jitter = (torch.as_tensor(jitter).float()
                         if model.base_gate and not model.carry_raw else jitter)
        base = torch.stack([lr_rgb[(y + ny).clamp(0, h - 1), (x + nx).clamp(0, w - 1)]
                            for ny, nx in _phase_offsets(model, sample_jitter)], dim=0).permute(3, 0, 1, 2)[None]
    else:
        base = lr_rgb.permute(2, 0, 1)[None, :, None]
    if model.base_gate:
        kernels, windows = _base_constants(model, jitter)
        kernels = kernels.to(device=o.device, dtype=lr_rgb.dtype).float()
        taps = torch.stack([lr_rgb[(y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1)]
                            for dy in range(-2, 2) for dx in range(-2, 2)])
        phases = []
        for q, (wy, wx) in enumerate(windows.tolist()):
            lanczos = torch.zeros_like(lr_rgb, dtype=torch.float32)
            for t in range(16):
                lanczos = lanczos + taps[t].float() * kernels[q, t]
            window = torch.stack([taps[(wy + dy) * 4 + wx + dx]
                                  for dy in range(3) for dx in range(3)])
            lanczos = lanczos.to(lr_rgb.dtype)
            phases.append(torch.minimum(torch.maximum(lanczos, window.amin(0)), window.amax(0)))
        lanczos = torch.stack(phases).permute(3, 0, 1, 2)[None]
        q = model._base_gate_offset
        gate = torch.sigmoid(px[:, q:q + p])[:, None]
        if model.coverage:
            gate = gate * (1 - cov)[:, None]
        base = torch.lerp(lanczos, base, gate)
    cur = base + px[:, :3 * p].reshape(1, 3, p, h, w)
    logit = px[:, 3 * p:4 * p]
    conf = None
    if model.accum:
        wc = _phase_weights(model, jitter, lr_rgb.dtype, o.device)
        if model.coverage:
            wc = wc * (1 - .9 * cov)
        wh = packed["conf_w"] * torch.sigmoid(px[:, -p:])
        if model.conf_consistent:
            wc = wc * torch.exp(logit.float().clamp(-2.0, 2.0)).to(lr_rgb.dtype)
            logit = (torch.log(wc) - torch.log(wh + 1e-3)).to(lr_rgb.dtype)
        else:
            logit = (logit + torch.log(wc) - torch.log(wh + 1e-3)).to(lr_rgb.dtype)
        conf = _phase_scatter((wh + wc).clamp(max=model.conf_max), s).float()
    current_weight = torch.sigmoid(logit)
    if model.history_age:
        limit = torch.maximum(1 / (1 + packed["age"]), model.alpha_min.clamp(0, 1)).to(lr_rgb.dtype)
        current_weight = torch.minimum(current_weight, limit)
    alpha = torch.where(packed["reset"] > 0.5, torch.ones_like(logit), current_weight)
    hist_b = packed["h_cl"].reshape(1, 3, p, h, w)
    if model.hist_residual:
        hr = torch.tanh(px[:, 4 * p:7 * p]).reshape(1, 3, p, h, w)
        hist_b = hist_b + hr * (packed["rng"].unsqueeze(2) * model.hres_gain.abs()).to(lr_rgb.dtype)
    if model.carry_raw:
        beta = torch.sigmoid(px[:, model._carry_beta_offset:model._carry_beta_offset + p])[:, None]
        h_mix = torch.lerp(packed["h_raw"].reshape(1, 3, p, h, w), hist_b, beta)
        out = torch.lerp(h_mix, base, alpha[:, None])
    else:
        out = torch.lerp(hist_b, cur, alpha[:, None])
    color = _phase_scatter(out.reshape(1, 3 * p, h, w), s)
    if model.carry_raw:
        display = out + px[:, :3 * p].reshape(1, 3, p, h, w)
        rgb = _phase_scatter(display.reshape(1, 3 * p, h, w), s)
        return rgb[0].permute(1, 2, 0).clamp(min=0).contiguous(), color, conf
    return color[0].permute(1, 2, 0).clamp(min=0).contiguous(), color, conf


# Explicit half round trips preserve the separate PyTorch elementwise operations.
# Disable contraction: grid normalisation must not become a different affine map.
_CUDA = r'''
#include <cuda_fp16.h>
extern "C" __global__ void shuffle_add(const half *up, half *skip, int h, int w, int c) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=4*h*w*c) return;
    int channel=t%c, pixel=t/c, x=pixel%(2*w), y=pixel/(2*w);
    int at=((y/2)*w+x/2)*(4*c)+channel*4+(y%2)*2+x%2;
    skip[t]=__hadd(up[at],skip[t]);
}
#ifndef K_STAGE
#define K_STAGE 0
#endif
__device__ float rh(float x) { return __half2float(__float2half_rn(x)); }
__device__ int bound(int x, int n) { return x < 0 ? 0 : (x >= n ? n - 1 : x); }
__device__ float c1(float x) { return ((1.25f*x-2.25f)*x)*x+1.0f; }
__device__ float c2(float x) { return ((-0.75f*x+3.75f)*x-6.0f)*x+3.0f; }
__device__ void cubic(float t, float *a) {
    a[0]=c2(t+1.0f); a[1]=c1(t); a[2]=c1(1.0f-t); a[3]=c2(2.0f-t);
}
template<bool Bicubic>
__device__ void sample4(const half *src, int H, int W,
                       float x, float y, float *result) {
    if (!Bicubic) { x=fminf(fmaxf(x,0.0f),W-1.0f); y=fminf(fmaxf(y,0.0f),H-1.0f); }
    int ix=(int)floorf(x), iy=(int)floorf(y);
    float tx=x-ix, ty=y-iy, wx[4], wy[4];
    const int count=Bicubic ? 4 : 2, first=Bicubic ? -1 : 0;
    if (Bicubic) { cubic(tx,wx); cubic(ty,wy); }
    else { wx[0]=1-tx; wx[1]=tx; wy[0]=1-ty; wy[1]=ty; }
    int xs[4], ys[4];
    #pragma unroll
    for(int i=0;i<count;i++) { xs[i]=bound(ix+i+first,W); ys[i]=bound(iy+i+first,H)*W; }
    #pragma unroll
    for(int c=0;c<4;c++) result[c]=0;
    #pragma unroll
    for(int j=0;j<count;j++) {
        float row[4]={0,0,0,0};
        #pragma unroll
        for(int i=0;i<count;i++) {
            // One aligned 8-byte load supplies RGB and confidence for this tap.
            unsigned long long tap=((const unsigned long long *)src)[ys[j]+xs[i]];
            #pragma unroll
            for(int c=0;c<4;c++)
                row[c] += __half2float(__ushort_as_half((unsigned short)(tap>>(16*c))))*wx[i];
        }
        #pragma unroll
        for(int c=0;c<4;c++) result[c] += row[c]*wy[j];
    }
}
__device__ float2 motion(const float *mv,int h,int w,float x,float y) {
    x=fmaxf(x,0.0f); y=fmaxf(y,0.0f);
    int ix=(int)floorf(x), iy=(int)floorf(y);
    float tx=x-ix, ty=y-iy;
    float wx0=1-tx, wy0=1-ty;
    int x0=bound(ix,w), x1=bound(ix+1,w), y0=bound(iy,h), y1=bound(iy+1,h);
    const float2 *src=(const float2 *)mv;
    float2 v00=src[y0*w+x0], v01=src[y0*w+x1], v10=src[y1*w+x0], v11=src[y1*w+x1];
    float ax=wx0*v00.x+tx*v01.x, bx=wx0*v10.x+tx*v11.x;
    float ay=wx0*v00.y+tx*v01.y, by=wx0*v10.y+tx*v11.y;
    float2 result;
    result.x=wy0*ax+ty*bx; result.y=wy0*ay+ty*by;
    return result;
}
__device__ int pack_index(int nchw,int ty,int tx,int c,int Ht,int Wt,int C) {
    return nchw ? ((c*Ht+ty)*Wt+tx) : ((ty*Wt+tx)*C+c);
}
__device__ void put(half *X,int base,int channel,int phase,float value,int stride) {
    X[base+(channel*4+phase)*stride]=__float2half_rn(value);
}
extern "C" __global__ void pack(
    const half *prev,const half *lr,const float *mv,const half *depth,
    const float *prev_depth,
    half *X,half *hcl,half *hraw,half *cw,half *reset,half *rng,const int *offsets,
    int h,int w,int s,int hp,int wp,float box,int first,int accum,int bicubic,
    int depth_test,int prev_depth_half,int conf_motion,float conf_m,int hist_residual,int nchw,
    int carry_raw,int nearest_sample) {
    int xp=blockIdx.x*blockDim.x+threadIdx.x, yp=blockIdx.y*blockDim.y+threadIdx.y;
    const bool active=yp<hp && xp<wp;
    if(!K_STAGE && !active) return;
    int y=bound(yp,h), x=bound(xp,w);
    int n=h*w, at=y*w+x, p=s*s, H=h*s, W=w*s;
    int channels=4*(9+3*p+(accum?p:0)+(carry_raw?p:0));
    int base=pack_index(nchw,yp/2,xp/2,0,hp/2,wp/2,channels), phase=(yp%2)*2+xp%2;
    int stride=nchw ? (hp/2)*(wp/2) : 1;
    int valid=yp<h && xp<w;
    // NHWC: the four threads of a trunk cell sit in two different warps, so direct 2-byte stores
    // scatter into a wide cell. Stage the block's cells in shared memory and copy them out in
    // contiguous 8-byte words instead.
    extern __shared__ half tile[];
    half *out_x=X;
    if(K_STAGE) {
        out_x=tile;
        base=((threadIdx.y>>1)*(blockDim.x>>1)+(threadIdx.x>>1))*channels;
    }
    float2 m=((const float2 *)mv)[at];
    float u=(x+0.5f)/w+m.x, v=(y+0.5f)/h+m.y;
    float rs=(first || !(u>=0 && u<1 && v>=0 && v<1)) ? 1.0f : 0.0f;
    if(depth_test) {
        float gx=u*2.0f-1.0f, gy=v*2.0f-1.0f;
        if(prev_depth_half) { gx=rh(gx); gy=rh(gy); }
        float sx=((gx+1.0f)*w-1.0f)/2.0f, sy=((gy+1.0f)*h-1.0f)/2.0f;
        int ix=(int)rintf(fminf(fmaxf(sx,0.0f),w-1.0f));
        int iy=(int)rintf(fminf(fmaxf(sy,0.0f),h-1.0f));
        float prev_d=prev_depth[iy*w+ix], mn=3.0e38f, mx=-3.0e38f;
        #pragma unroll
        for(int dy=-1;dy<=1;dy++) {
            #pragma unroll
            for(int dx=-1;dx<=1;dx++) {
                int yy=y+dy, xx=x+dx;
                if(yy>=0 && yy<h && xx>=0 && xx<w) {
                    float value=__half2float(depth[yy*w+xx]);
                    mn=fminf(mn,value); mx=fmaxf(mx,value);
                }
            }
        }
        if(prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f)) rs=1.0f;
    }
    float survive=1-rs;
    float lo[3], hi[3], ranges[3];
    int taps[9];
    #pragma unroll
    for(int dy=-1;dy<=1;dy++) {
        #pragma unroll
        for(int dx=-1;dx<=1;dx++) taps[(dy+1)*3+dx+1]=bound(y+dy,h)*w+bound(x+dx,w);
    }
    #pragma unroll
    for(int c=0;c<3;c++) {
        float mn=3.0e38f, mx=-3.0e38f;
        #pragma unroll
        for(int i=0;i<9;i++) {
            float value=__half2float(lr[c*n+taps[i]]);
            mn=fminf(mn,value); mx=fmaxf(mx,value);
        }
        float range=rh(mx-mn), slack=rh(range*fabsf(box));
        if(carry_raw) ranges[c]=range;
        // Reuse this reduction: three stores/loads replace 27 LR loads in resolve.
        if(valid && hist_residual) rng[c*n+at]=__float2half_rn(range);
        lo[c]=rh(mn-slack); hi[c]=rh(mx+slack);
        put(out_x,base,c,phase,__half2float(lr[c*n+at]),stride);
        put(out_x,base,3+3*p+c,phase,range,stride);
    }
    // Reciprocal and scalar multiply are separate half operations in forward.
    float d=fmaxf(__half2float(depth[at]),rh(0.01f));
    put(out_x,base,6+3*p,phase,rh(1.0f/d),stride);
    float vx=m.x*w, vy=m.y*h;
    float speed=sqrtf(vx*vx+vy*vy);
    put(out_x,base,7+3*p,phase,fminf(speed*0.1f,1.0f),stride);
    float conf_den=1.0f;
    if(conf_motion) conf_den=rh(1.0f+rh(fabsf(conf_m)*rh(fminf(speed,16.0f))));
    put(out_x,base,8+3*p,phase,rs,stride);
    if(valid) reset[at]=__float2half_rn(rs);
    int oy0=y*s, ox0=x*s;
    float luma_range=0;
    if(carry_raw) luma_range=rh(rh(rh(0.25f*ranges[0])+rh(0.5f*ranges[1]))+rh(0.25f*ranges[2]));
    for(int py=0;py<s;py++) for(int px=0;px<s;px++) {
        int q=py*s+px, oy=oy0+py, ox=ox0+px;
        float lx=(ox+0.5f)/s-0.5f, ly=(oy+0.5f)/s-0.5f;
        float2 mm=motion(mv,h,w,lx,ly);
        float gx=((ox+0.5f)/W+mm.x)*2.0f-1.0f;
        float gy=((oy+0.5f)/H+mm.y)*2.0f-1.0f;
        float sx=((gx+1.0f)*W-1.0f)/2.0f, sy=((gy+1.0f)*H-1.0f)/2.0f;
        float sampled[4];
        if(bicubic) sample4<true>(prev,H,W,sx,sy,sampled);
        else sample4<false>(prev,H,W,sx,sy,sampled);
        #pragma unroll
        for(int c=0;c<3;c++) {
            float hist=rh(sampled[c]);
            float cl=fminf(fmaxf(hist,lo[c]),hi[c]);
            put(out_x,base,3+c*p+q,phase,cl,stride);
            if(valid) hcl[(c*p+q)*n+at]=__float2half_rn(cl);
            if(valid && carry_raw) hraw[(c*p+q)*n+at]=__float2half_rn(hist);
        }
        if(accum) {
            float conf=rh(fmaxf(rh(sampled[3]),0.0f)*survive);
            if(conf_motion) conf=rh(conf/conf_den);
            put(out_x,base,9+3*p+q,phase,log1pf(conf),stride);
            if(valid) cw[q*n+at]=__float2half_rn(conf);
        }
        if(carry_raw) {
            int sample_at=nearest_sample ? bound(y+offsets[2*q],h)*w+bound(x+offsets[2*q+1],w) : at;
            float raw_luma=rh(rh(rh(0.25f*rh(sampled[0]))+rh(0.5f*rh(sampled[1])))+rh(0.25f*rh(sampled[2])));
            float base_luma=rh(rh(rh(0.25f*__half2float(lr[sample_at]))+
                                  rh(0.5f*__half2float(lr[n+sample_at])))+
                               rh(0.25f*__half2float(lr[2*n+sample_at])));
            float innovation=rh(rh(raw_luma-base_luma)/rh(luma_range+0.02f));
            innovation=rh(fminf(fmaxf(innovation,-4.0f),4.0f)*survive);
            put(out_x,base,9+4*p+q,phase,innovation,stride);
        }
    }
    if(K_STAGE) {
        __syncthreads();
        const int tx=blockDim.x>>1, ty=blockDim.y>>1, Wt=wp/2, Ht=hp/2;
        const int cx0=blockIdx.x*tx, cy0=blockIdx.y*ty;
        const int ncell=min(tx,Wt-cx0), tid=threadIdx.y*blockDim.x+threadIdx.x, nthreads=blockDim.x*blockDim.y;
        const int words=ncell*channels/4;
        for(int r=0;r<ty && cy0+r<Ht;r++) {
            const uint2 *src=(const uint2 *)(tile+(size_t)r*tx*channels);
            uint2 *dst=(uint2 *)(X+((size_t)(cy0+r)*Wt+cx0)*channels);
            for(int i=tid;i<words;i+=nthreads) dst[i]=src[i];
        }
    }
}
__device__ float sig(float x) { return rh(1.0f/(1.0f+expf(-x))); }
// Per-phase Lanczos base from the 4x4 register window. Only the 3x3 window at (WY, WX) has
// non-zero weight, so summing those nine taps in index order equals the sixteen-tap sum.
template<int WY,int WX>
__device__ __forceinline__ void lanczos_window(const float t[4][4][3], const float *k, float *out) {
    float kk[9];
    #pragma unroll
    for(int dy=0;dy<3;dy++) {
        #pragma unroll
        for(int dx=0;dx<3;dx++) kk[dy*3+dx]=rh(k[(WY+dy)*4+WX+dx]);
    }
    #pragma unroll
    for(int c=0;c<3;c++) {
        float acc=0, lo=t[WY][WX][c], hi=lo;
        #pragma unroll
        for(int dy=0;dy<3;dy++) {
            #pragma unroll
            for(int dx=0;dx<3;dx++) {
                float tap=t[WY+dy][WX+dx][c];
                acc=acc+tap*kk[dy*3+dx];
                lo=fminf(lo,tap); hi=fmaxf(hi,tap);
            }
        }
        out[c]=fminf(fmaxf(rh(acc),lo),hi);
    }
}
__device__ __forceinline__ void lanczos_phase(const float t[4][4][3], const float *k, int wy, int wx, float *out) {
    switch(wy*2+wx) {
        case 0: lanczos_window<0,0>(t,k,out); break;
        case 1: lanczos_window<0,1>(t,k,out); break;
        case 2: lanczos_window<1,0>(t,k,out); break;
        default: lanczos_window<1,1>(t,k,out); break;
    }
}
#ifndef K_S
#define K_S 0
#endif
#ifndef K_BASE
#define K_BASE -1
#endif
#ifndef K_CARRY
#define K_CARRY -1
#endif
#ifndef K_ACCUM
#define K_ACCUM -1
#endif
#ifndef K_HRES
#define K_HRES -1
#endif
#ifndef K_NEAREST
#define K_NEAREST -1
#endif
#ifndef K_CONS
#define K_CONS -1
#endif
#ifndef K_NCHW
#define K_NCHW -1
#endif
#define FLAG(name, arg) (K_##name >= 0 ? K_##name : (arg))
extern "C" __global__ void resolve(
    const half *o,const half *lr,const half *hcl,const half *hraw,const half *cw,const half *reset,
    const half *wc,const half *rng,const int *offsets,const float *kernels,const int *windows,half *history,half *rgb,
    int h,int w,int s_arg,int hp,int wp,int accum_arg,int hist_residual_arg,int nearest_sample_arg,int conf_consistent_arg,int nchw_arg,int carry_raw_arg,int base_gate_arg,
    float hres_gain,
    float conf_max,int write_rgb) {
    const int s=K_S>0 ? K_S : s_arg;
    const int accum=FLAG(ACCUM,accum_arg), hist_residual=FLAG(HRES,hist_residual_arg);
    const int nearest_sample=FLAG(NEAREST,nearest_sample_arg), conf_consistent=FLAG(CONS,conf_consistent_arg);
    const int nchw=FLAG(NCHW,nchw_arg), carry_raw=FLAG(CARRY,carry_raw_arg), base_gate=FLAG(BASE,base_gate_arg);
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(y>=h || x>=w) return;
    int at=y*w+x,n=h*w,p=s*s,W=w*s;
    int gate0=(4+3*hist_residual)*p;
    int beta0=gate0+base_gate*p;
    int keep0=beta0+carry_raw*p;
    int C=4*(4+3*hist_residual+accum+carry_raw+base_gate)*p;
    int base=pack_index(nchw,y>>1,x>>1,(y&1)*2+(x&1),hp/2,wp/2,C);
    int stride=nchw ? (hp/2)*(wp/2) : 1;
    int step=4*stride;
    float hres_scale[3];
    if(hist_residual) {
        #pragma unroll
        for(int c=0;c<3;c++) hres_scale[c]=rh(__half2float(rng[c*n+at])*fabsf(hres_gain));
    }
    float t[4][4][3];
    if(base_gate) {
        int xs[4], ys[4];
        #pragma unroll
        for(int i=0;i<4;i++) { xs[i]=bound(x+i-2,w); ys[i]=bound(y+i-2,h)*w; }
        #pragma unroll
        for(int ty=0;ty<4;ty++) {
            #pragma unroll
            for(int tx=0;tx<4;tx++) {
                int src=ys[ty]+xs[tx];
                #pragma unroll
                for(int c=0;c<3;c++) t[ty][tx][c]=__half2float(lr[c*n+src]);
            }
        }
    }
    const bool is_reset=__half2float(reset[at])>0.5f;
    const int dst0=y*s*W+x*s;
    const bool write_display=write_rgb || carry_raw;
    const bool pairs=(K_S==2);
    unsigned long long first_bits=0;
    unsigned short first_rgb[3]={0,0,0};
    #pragma unroll
    for(int py=0;py<s;py++) {
        #pragma unroll
        for(int px=0;px<s;px++) {
            int q=py*s+px, dst=dst0+py*W+px;
            unsigned long long bits=0;
            float logit=__half2float(o[base+(3*p+q)*step]);
            if(accum) {
                float keep=sig(__half2float(o[base+(keep0+q)*step]));
                float wh=rh(__half2float(cw[q*n+at])*keep), weight=__half2float(wc[q]);
                if(conf_consistent) {
                    weight=rh(weight*rh(expf(fminf(fmaxf(logit,-2.0f),2.0f))));
                    logit=rh(rh(logf(weight))-rh(logf(rh(wh+0.001f))));
                } else logit=rh(rh(logit+rh(logf(weight)))-rh(logf(rh(wh+0.001f))));
                bits=(unsigned long long)__half_as_ushort(__float2half_rn(fminf(rh(wh+weight),conf_max)))<<48;
            }
            float alpha=is_reset ? 1.0f : sig(logit);
            float beta=carry_raw ? sig(__half2float(o[base+(beta0+q)*step])) : 0;
            float gate=base_gate ? sig(__half2float(o[base+(gate0+q)*step])) : 0;
            int ny=0, nx=0;
            if(nearest_sample) { ny=offsets[2*q]; nx=offsets[2*q+1]; }
            float lan[3], nr[3];
            if(base_gate) {
                lanczos_phase(t,kernels+q*16,windows[2*q],windows[2*q+1],lan);
                if(ny>=-1 && ny<=1 && nx>=-1 && nx<=1) {
                    #pragma unroll
                    for(int c=0;c<3;c++) {
                        float a=ny<0 ? (nx<0 ? t[1][1][c] : (nx==0 ? t[1][2][c] : t[1][3][c]))
                               : (ny==0 ? (nx<0 ? t[2][1][c] : (nx==0 ? t[2][2][c] : t[2][3][c]))
                                        : (nx<0 ? t[3][1][c] : (nx==0 ? t[3][2][c] : t[3][3][c])));
                        nr[c]=a;
                    }
                } else {
                    int sample_at=bound(y+ny,h)*w+bound(x+nx,w);
                    #pragma unroll
                    for(int c=0;c<3;c++) nr[c]=__half2float(lr[c*n+sample_at]);
                }
            } else {
                int sample_at=nearest_sample ? bound(y+ny,h)*w+bound(x+nx,w) : at;
                #pragma unroll
                for(int c=0;c<3;c++) nr[c]=__half2float(lr[c*n+sample_at]);
            }
            unsigned short pix[3];
            #pragma unroll
            for(int c=0;c<3;c++) {
                float residual=__half2float(o[base+(c*p+q)*step]);
                float current=nr[c];
                if(base_gate) {
                    float lanczos=lan[c];
                    current=rh(gate<0.5f ? lanczos+gate*(current-lanczos) : current-(current-lanczos)*(1-gate));
                }
                float cur=carry_raw ? current : rh(current+residual);
                float hist=__half2float(hcl[(c*p+q)*n+at]);
                if(hist_residual) {
                    float hr=rh(tanhf(__half2float(o[base+((4+c)*p+q)*step])));
                    hist=rh(hist+rh(hr*hres_scale[c]));
                }
                if(carry_raw) {
                    float raw=__half2float(hraw[(c*p+q)*n+at]);
                    hist=rh(beta<0.5f ? raw+beta*(hist-raw) : hist-(hist-raw)*(1-beta));
                }
                // torch.lerp uses the numerically stable branch in opmath precision.
                float value=rh(alpha<0.5f ? hist+alpha*(cur-hist) : cur-(cur-hist)*(1-alpha));
                bits|=(unsigned long long)__half_as_ushort(__float2half_rn(value))<<(16*c);
                pix[c]=__half_as_ushort(__float2half_rn(fmaxf(carry_raw ? rh(value+residual) : value,0.0f)));
            }
            if(pairs) {
                // Two horizontally adjacent output pixels leave as one 16-byte history and three 4-byte display stores.
                if(px==0) {
                    first_bits=bits;
                    first_rgb[0]=pix[0]; first_rgb[1]=pix[1]; first_rgb[2]=pix[2];
                } else {
                    ((ulonglong2 *)history)[(dst-1)>>1]=make_ulonglong2(first_bits,bits);
                    if(write_display) {
                        unsigned int *out=(unsigned int *)rgb+((dst-1)>>1)*3;
                        out[0]=(unsigned int)first_rgb[0]|((unsigned int)first_rgb[1]<<16);
                        out[1]=(unsigned int)first_rgb[2]|((unsigned int)pix[0]<<16);
                        out[2]=(unsigned int)pix[1]|((unsigned int)pix[2]<<16);
                    }
                }
            } else {
                ((unsigned long long *)history)[dst]=bits;
                if(write_display) {
                    #pragma unroll
                    for(int c=0;c<3;c++) ((unsigned short *)rgb)[dst*3+c]=pix[c];
                }
            }
        }
    }
}
'''


def _cuda_source(model):
    """Keep the legacy source/ABI unchanged when robustness is disabled."""
    source = _CUDA
    if model.depth_soft:
        source = source.replace('    if(depth_test) {',
                                '    float reset_feature=rs;\n    if(depth_test) {')
        source = source.replace('        if(prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f)) rs=1.0f;',
                                '#if K_DEPTH_SOFT\n'
                                '        if(prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f)) reset_feature=1.0f;\n'
                                '#else\n'
                                '        if(prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f)) rs=1.0f;\n'
                                '#endif')
        source = source.replace('    put(out_x,base,8+3*p,phase,rs,stride);',
                                '    put(out_x,base,8+3*p,phase,reset_feature,stride);')
    if not (model.mv_dilate or model.depth_dilate or model.thin_lock):
        source = _coverage_cuda(source, float(model.coverage_bias)) if model.coverage else source
        source = _osc_cuda(source) if model.depth_soft_osc else source
        return _age_cuda(source) if model.history_age else source
    source = source.replace('__device__ float2 motion(', r'''
__device__ int nearest_depth(const half *depth,int h,int w,int y,int x) {
    int best=bound(y-1,h)*w+bound(x-1,w);
    float near=__half2float(depth[best]);
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
        int at=bound(y+dy,h)*w+bound(x+dx,w);
        float d=__half2float(depth[at]);
        if(d>near) { near=d; best=at; }
    }
    return best;
}
__device__ float2 selected_motion(const float2 *mv,const half *depth,int h,int w,int y,int x) {
    int at=y*w+x;
#if K_MV_DILATE
    at=nearest_depth(depth,h,w,y,x);
#endif
    return mv[at];
}
__device__ float thin_feature(const half *lr,int h,int w,int y,int x) {
    float l[9], lo=3e38f, hi=-3e38f;
    int n=h*w;
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
        int at=bound(y+dy,h)*w+bound(x+dx,w), q=(dy+1)*3+dx+1;
        l[q]=.25f*__half2float(lr[at])+.5f*__half2float(lr[n+at])+.25f*__half2float(lr[2*n+at]);
        lo=fminf(lo,l[q]); hi=fmaxf(hi,l[q]);
    }
    float ridge=0;
    for(int a=0;a<4;a++) {
        float bright=fminf(l[4]-l[a],l[4]-l[8-a]);
        float dark=fminf(l[a]-l[4],l[8-a]-l[4]);
        ridge=fmaxf(ridge,fmaxf(bright,dark));
    }
    return rh(fminf(fmaxf((ridge/fmaxf(hi-lo,.001f)-.25f)/.75f,0.f),1.f));
}
__device__ float2 motion(''')
    source = source.replace('motion(const float *mv,int h', 'motion(const float *mv,const half *depth,int h')
    source = source.replace('src[y0*w+x0], v01=src[y0*w+x1], v10=src[y1*w+x0], v11=src[y1*w+x1]',
                            'selected_motion(src,depth,h,w,y0,x0), v01=selected_motion(src,depth,h,w,y0,x1), '
                            'v10=selected_motion(src,depth,h,w,y1,x0), v11=selected_motion(src,depth,h,w,y1,x1)')
    source = source.replace('int carry_raw,int nearest_sample) {',
                            'int carry_raw,int nearest_sample,half *stored_depth,float thin_factor) {')
    source = source.replace('(carry_raw?p:0));', '(carry_raw?p:0)+K_THIN_LOCK);')
    source = source.replace('float2 m=((const float2 *)mv)[at];',
                            'float2 m=selected_motion((const float2 *)mv,depth,h,w,y,x);\n'
                            '#if K_DEPTH_DILATE\n'
                            '    if(valid) stored_depth[at]=depth[nearest_depth(depth,h,w,y,x)];\n'
                            '#endif')
    source = source.replace('    float lo[3], hi[3], ranges[3];',
                            '    float thin=0;\n#if K_THIN_LOCK\n'
                            '    thin=thin_feature(lr,h,w,y,x);\n'
                            '    put(out_x,base,9+3*p+(accum?p:0)+(carry_raw?p:0),phase,thin,stride);\n'
                            '#endif\n    float lo[3], hi[3], ranges[3];')
    source = source.replace('        if(carry_raw) ranges[c]=range;',
                            '#if K_THIN_LOCK\n'
                            '        slack=rh(slack*(thin>.5f ? thin_factor : 1.f));\n'
                            '#endif\n        if(carry_raw) ranges[c]=range;')
    source = source.replace('motion(mv,h,w,lx,ly)', 'motion(mv,depth,h,w,lx,ly)')
    source = _coverage_cuda(source, float(model.coverage_bias)) if model.coverage else source
    source = _osc_cuda(source) if model.depth_soft_osc else source
    return _age_cuda(source) if model.history_age else source


def _age_cuda(source):
    """Age state and input are compiled only for K_HISTORY_AGE specializations."""
    source = source.replace('int carry_raw,int nearest_sample',
                            'int carry_raw,int nearest_sample,const float *prev_age,float *next_age,float *warped_age')
    start = source.index('    int channels=4*(')
    end = source.index(';', start)
    source = source[:end-1] + '+K_HISTORY_AGE' + source[end-1:]
    source = source.replace('    int oy0=y*s, ox0=x*s;', r'''    float age_gx=u*2-1, age_gy=v*2-1;
    int age_ix=(int)rintf(fminf(fmaxf(((age_gx+1)*w-1)/2,0.f),w-1.f));
    int age_iy=(int)rintf(fminf(fmaxf(((age_gy+1)*h-1)/2,0.f),h-1.f));
    float age=rs ? 0:prev_age[age_iy*w+age_ix];
    put(out_x,base,channels/4-1,phase,log2f(1+age)/5,stride);
    if(valid) { warped_age[at]=age; next_age[at]=rs ? 0:fminf(age+1,32.f); }
    int oy0=y*s, ox0=x*s;''')
    source = source.replace('float conf_max,int write_rgb)',
                            'float conf_max,int write_rgb,const float *warped_age,float alpha_min)')
    source = source.replace('float alpha=is_reset ? 1.0f : sig(logit);', r'''float alpha=is_reset ? 1.0f : sig(logit);
            if(!is_reset) {
                float age=warped_age[at];
                alpha=fminf(alpha,rh(fmaxf(1/(1+age),alpha_min)));
            }''')
    return source


def _osc_cuda(source):
    """Hard inference gate; ABI and source remain unchanged with the option off."""
    source = source.replace('float *next_cov', 'float *next_cov,float soft_osc_threshold')
    source = source.replace('    float reset_feature=rs;',
                            '    bool mismatch=false, departure=false;\n    float reset_feature=rs;')
    source = source.replace('        if(prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f)) reset_feature=1.0f;',
                            '        mismatch=prev_d<rh(mn*0.9f) || prev_d>rh(mx*1.1f);\n'
                            '        departure=prev_d>rh(mx*1.25f);\n'
                            '        if(mismatch) reset_feature=1.0f;')
    # Gather pre-update evidence before any reset writes or confidence masking.
    start = source.index('    // Float32 nearest reprojection')
    end = source.index('    int cov0=9+3*p', start)
    preamble = source[start:end]
    source = source[:start] + source[end:]
    source = source.replace('    float survive=1-rs;\n', '')
    source = source.replace('    // Reciprocal and scalar multiply', preamble + r'''#if K_DEPTH_SOFT_OSC
    // Half-rounded coverage feature; threshold remains float32. Reversed-Z: larger is nearer.
    float osc_n=rs ? 0:rh(fminf(fmaxf(osc/span,0.f),4.f));
    bool dither=osc_n>fminf(fmaxf(soft_osc_threshold,.05f),4.f);
    if(departure || (mismatch && !dither)) rs=1.0f;
#endif
    float survive=1-rs;
    // Reciprocal and scalar multiply''')
    return source


def _coverage_cuda(source, coverage_bias):
    """Coverage-only ABI, eliminated entirely from legacy kernel specializations."""
    source = source.replace('int carry_raw,int nearest_sample',
                            'int carry_raw,int nearest_sample,const float *prev_cov,float *next_cov')
    source = source.replace('(carry_raw?p:0)', '(carry_raw?p:0)+2*K_COVERAGE', 1)
    source = source.replace('if(carry_raw) ranges[c]=range;', 'ranges[c]=range;')
    source = source.replace('    int oy0=y*s, ox0=x*s;', r'''    // Float32 nearest reprojection, independent of depth texture storage precision.
    float gx=u*2-1, gy=v*2-1;
    int ix=(int)rintf(fminf(fmaxf(((gx+1)*w-1)/2,0.f),w-1.f));
    int iy=(int)rintf(fminf(fmaxf(((gy+1)*h-1)/2,0.f),h-1.f));
    int src=iy*w+ix;
    float m1=prev_cov[src], m2=prev_cov[n+src], osc=prev_cov[2*n+src], last=prev_cov[3*n+src];
    float luma=.25f*__half2float(lr[at])+.5f*__half2float(lr[n+at])+.25f*__half2float(lr[2*n+at]);
    float span=.25f*ranges[0]+.5f*ranges[1]+.25f*ranges[2]+.02f;
    int cov0=9+3*p+(accum?p:0)+(carry_raw?p:0);
#ifdef K_THIN_LOCK
    cov0+=K_THIN_LOCK;
#endif
    put(out_x,base,cov0,phase,rs ? 0:fminf(fmaxf(m2-m1*m1,0.f)/span,4.f),stride);
    put(out_x,base,cov0+1,phase,rs ? 0:fminf(osc/span,4.f),stride);
    if(valid) {
        next_cov[at]=rs ? luma:m1+.25f*(luma-m1);
        next_cov[n+at]=rs ? luma*luma:m2+.25f*(luma*luma-m2);
        next_cov[2*n+at]=rs ? 0:osc+.25f*(fabsf(luma-last)-osc);
        next_cov[3*n+at]=luma;
    }
    int oy0=y*s, ox0=x*s;''')
    source = source.replace('    int keep0=beta0+carry_raw*p;',
                            '    int cov0=beta0+carry_raw*p;\n    int keep0=cov0+K_COVERAGE*p;')
    source = source.replace('4+3*hist_residual+accum+carry_raw+base_gate)',
                            '4+3*hist_residual+accum+carry_raw+base_gate+K_COVERAGE)')
    source = source.replace('            if(accum) {', r'''            float floor=sig(rh(COVERAGE_BIAS));
            float cov=rh(fminf(fmaxf(rh(sig(__half2float(o[base+(cov0+q)*step]))-floor)/rh(1-floor),0.f),1.f));
            if(accum) {''',1)
    source = source.replace('                if(conf_consistent) {',
                            '                weight=rh(weight*rh(1-rh(.9f*cov)));\n                if(conf_consistent) {',1)
    source = source.replace('            int ny=0, nx=0;',
                            '            gate=rh(gate*rh(1-cov));\n            int ny=0, nx=0;',1)
    return source.replace("COVERAGE_BIAS", f"{coverage_bias:.9e}f")


class FusedFast:
    """Inference-only, fixed-size runtime; one serial stream per instance.

    CPU construction and step_reference never import CuPy. The first CUDA step
    compiles kernels and captures or builds the trunk, excluded from timings.
    Inputs may be fp32; staging converts lr/depth to fp16 and motion to fp32.
    Host folding snapshots fp32 parameters and caches up to 64 jitter pairs.
    Supply jitter on the CPU to avoid a device-to-host synchronisation; GPU
    jitter tensors are supported. fold="gpu" retains the original fp16 fold.
    layout="nchw" uses contiguous trunk buffers; TensorRT requires this layout.
    carry_raw always writes/returns the display, independent of write_rgb;
    state colour holds the carry before the residual. state_rgb uses the display.
    CUDA_PATH is left to CuPy's normal toolkit discovery, without path overrides.
    """

    def __init__(self, model, fold="host", layout="nhwc", trunk="cudnn"):
        _validate(model)
        if fold not in ("host", "gpu"):
            raise ValueError("fold must be 'host' or 'gpu'")
        if layout not in ("nhwc", "nchw"):
            raise ValueError("layout must be 'nhwc' or 'nchw'")
        if trunk not in ("cudnn", "tensorrt"):
            raise ValueError("trunk must be 'cudnn' or 'tensorrt'")
        if trunk == "tensorrt" and layout != "nchw":
            raise ValueError("trunk='tensorrt' requires layout='nchw'")
        self.model = model
        self.fold = fold
        self.layout = layout
        self.trunk = trunk
        self._ready = False

    def init_state(self):
        if self._ready:
            for history in self._histories:
                history.zero_()
            if self.model.coverage:
                for cov in self._coverage_states:
                    cov.zero_()
            if self.model.history_age:
                for age in self._age_states:
                    age.zero_()
            self._depth.fill_(1)
            if self.model.depth_dilate:
                self._stored_depth.fill_(1)
            if self.model.carry_raw:
                self._rgb.zero_()
            return self._state(0, 0)
        device = next(self.model.parameters()).device
        return self.model.init_state(device)

    @torch.no_grad()
    def step_reference(self, state, lr_rgb, mv_lr, depth, jitter):
        jitter = _signed_jitter(self.model, jitter)
        packed = ref_pack(self.model, state, lr_rgb, mv_lr, depth, jitter)
        o = run_trunk(self.model, packed["X"], jitter)
        rgb, color, conf = ref_resolve(self.model, o, lr_rgb, packed, jitter)
        new = FastState(color, state.feat, packed["depth"],
                        state.frame_index + 1, conf, *packed["coverage_state"], packed["new_age"])
        if self.model.carry_raw:
            new._display_rgb = rgb
        return rgb, new

    @torch.no_grad()
    def _prepare(self, device):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FusedFast.step requires CUDA; use step_reference on CPU")
        try:
            import cupy
        except ImportError as exc:
            raise RuntimeError("FusedFast.step requires CuPy with CUDA support") from exc
        self._cp = cupy
        self._device = device
        memory_format = torch.channels_last if self.layout == "nhwc" else torch.contiguous_format
        self._net = copy.deepcopy(self.model).to(device=device, dtype=torch.float16,
                                                 memory_format=memory_format).eval()
        # Geometry and scalar parameters stay fp32, as in autocast forward.
        for name in ("box_slack", "acc_sharp", "_acc_px", "_acc_py", "conf_m", "hres_gain", "thin_slack"):
            if hasattr(self.model, name):
                getattr(self._net, name).data = getattr(self.model, name).detach().to(device).float()
        self._net.requires_grad_(False)
        if self._net.film is not None:
            self._net.film[1].inplace = True
        h, w = self.model.render_size
        H, W = self.model.output_size
        p = self.model.p
        hp, wp = h + self.model._pad[0], w + self.model._pad[1]
        def empty(shape, dtype=torch.float16):
            return torch.empty(shape, device=device, dtype=dtype)
        self._lr = empty((1, 3, h, w))
        self._lr_hwc = self._lr[0].permute(1, 2, 0)
        self._mv = empty((h, w, 2), torch.float32)
        self._depth = empty((1, 1, h, w))
        self._depth_hw = self._depth[0, 0]
        self._stored_depth = empty((1, 1, h, w)) if self.model.depth_dilate else self._depth
        self._prev_depth = empty((1, 1, h, w), torch.float32)
        self._jitter = torch.zeros(2, device=device, dtype=torch.float32)
        self._jitter_host = torch.zeros(2, dtype=torch.float32, pin_memory=True)
        self._feat = empty((1, 0, *self.model._tr_size))
        self._X = empty((1, self.model.stem.in_channels, hp // 2, wp // 2)).contiguous(memory_format=memory_format)
        self._X.zero_()
        self._hcl = empty((1, 3*p, h, w))
        self._hraw = empty((1, 3*p, h, w)) if self.model.carry_raw else None
        self._cw = empty((1, p, h, w))
        self._reset = empty((1, 1, h, w))
        self._rng = empty((1, 3, h, w)) if self.model.hist_residual else None
        self._histories = [torch.zeros((H, W, 4), device=device, dtype=torch.float16) for _ in range(2)]
        self._age_states = ([torch.zeros((1, 1, h, w), device=device) for _ in range(2)]
                            if self.model.history_age else None)
        self._warped_age = torch.zeros((1, 1, h, w), device=device) if self.model.history_age else None
        self._coverage_states = ([torch.zeros((1, 4, h, w), device=device) for _ in range(2)]
                                 if self.model.coverage else None)
        self._colors = [t[..., :3].permute(2, 0, 1)[None] for t in self._histories]
        self._confs = [t[..., 3:].permute(2, 0, 1)[None] for t in self._histories]
        self._rgb = empty((H, W, 3))
        self._wc = torch.ones(p, device=device, dtype=torch.float16)
        self._offsets = empty((p, 2), torch.int32)
        self._base_kernels = empty((p, 16), torch.float32) if self.model.base_gate else None
        self._base_windows = empty((p, 2), torch.int32) if self.model.base_gate else None
        self._fold_weight = torch.empty_like(self._net.out.weight)
        self._fold_bias = torch.empty_like(self._net.out.bias)
        self._host_frames = _HostFrameCache(self.model, pin_memory=True, fold=self.fold == "host")
        self._update_frame((0.0, 0.0))
        self.run_trunk_module = TrunkModule(self._net, self._fold_weight, self._fold_bias)
        import numpy as np
        self._np = np
        with cupy.cuda.Device(device.index if device.index is not None else torch.cuda.current_device()):
            options = self._kernel_options()
            source = _cuda_source(self.model)
            self._pack_kernel = cupy.RawKernel(source, "pack", options=options)
            self._resolve_kernel = cupy.RawKernel(source, "resolve", options=options)
            self._pack_kernel.compile()
            self._resolve_kernel.compile()
            if self.layout == "nhwc":
                self._shuffle_kernel = cupy.RawKernel(_CUDA, "shuffle_add", options=options)
                self._shuffle_kernel.compile()
        if self.trunk == "tensorrt":
            self._o = empty((1, self.model.n_px * 4, hp // 2, wp // 2))
            self._prepare_tensorrt()
        else:
            # Capture allocates convolution work buffers once; replay reuses them.
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            benchmark = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = True
            try:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._middle()
                current.wait_stream(stream)
                current.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=stream):
                    self._middle()
            finally:
                torch.backends.cudnn.benchmark = benchmark
            current.wait_stream(stream)
        # asarray is zero-copy through the tensor's __cuda_array_interface__.
        self._arrays = {}
        with cupy.cuda.Device(device.index if device.index is not None else torch.cuda.current_device()):
            for name in ("_lr", "_mv", "_depth", "_prev_depth", "_X", "_hcl", "_cw", "_reset", "_wc", "_offsets", "_o", "_rgb"):
                self._arrays[name] = cupy.asarray(getattr(self, name))
            self._arrays["_stored_depth"] = cupy.asarray(self._stored_depth)
            self._arrays["_rng"] = (cupy.asarray(self._rng) if self._rng is not None
                                    else self._arrays["_lr"])
            self._arrays["_hraw"] = (cupy.asarray(self._hraw) if self._hraw is not None
                                     else self._arrays["_lr"])
            for name in ("_base_kernels", "_base_windows"):
                value = getattr(self, name)
                self._arrays[name] = cupy.asarray(value) if value is not None else self._arrays["_lr"]
            self._history_arrays = [cupy.asarray(t) for t in self._histories]
            self._age_arrays = ([cupy.asarray(t) for t in self._age_states]
                                if self.model.history_age else None)
            self._warped_age_array = cupy.asarray(self._warped_age) if self.model.history_age else None
            self._coverage_arrays = ([cupy.asarray(t) for t in self._coverage_states]
                                     if self.model.coverage else None)
        self._pack_sizes = tuple(np.int32(v) for v in (h, w, self.model.scale, hp, wp))
        self._resolve_sizes = tuple(np.int32(v) for v in (h, w, self.model.scale, hp, wp, self.model.accum,
                                                        self.model.hist_residual, self.model.nearest_sample,
                                                        self.model.conf_consistent, self.layout == "nchw",
                                                        self.model.carry_raw, self.model.base_gate))
        self._nchw = np.int32(self.layout == "nchw")
        self._carry_raw = np.int32(self.model.carry_raw)
        self._nearest_sample = np.int32(self.model.nearest_sample)
        # The model multiplies half tensors by these 0-dim fp32 CUDA parameters, which PyTorch
        # converts to half before the product; the kernels must round them the same way.
        half = lambda v: np.float32(np.float16(v))
        self._box = half(self.model.box_slack.detach().abs().item())
        self._age_resolve_args = ((self._warped_age_array, np.float32(self.model.alpha_min.detach().clamp(0, 1).item()))
                                  if self.model.history_age else ())
        self._osc_args = ((np.float32(self.model.soft_osc_threshold.detach().clamp(.05, 4).item()),)
                          if self.model.depth_soft_osc else ())
        self._robust_args = ((self._arrays["_stored_depth"],
                              half((1 + self.model.thin_slack.detach()).item()
                                   if self.model.thin_lock else 1))
                             if self.model.mv_dilate or self.model.depth_dilate or self.model.thin_lock else ())
        self._conf_max = np.float32(self.model.conf_max)
        self._conf_m = half(self.model.conf_m.detach().abs().item() if self.model.conf_motion else 0)
        self._flags = tuple(np.int32(v) for v in (self.model.accum, self.model.hist_filter == "bicubic",
                                                 self.model.depth_test))
        self._prev_depth_half = np.int32(0)
        self._conf_motion = np.int32(self.model.conf_motion)
        self._hist_residual = np.int32(self.model.hist_residual)
        self._hres_gain = half(self.model.hres_gain.detach().abs().item()
                               if self.model.hist_residual else 0)
        self._first_flags = (np.int32(0), np.int32(1))
        self._write_rgb_flags = (np.int32(0), np.int32(1))
        self._block = (32, 8)
        self._pack_grid = ((wp+31)//32, (hp+7)//8)
        self._resolve_grid = self._pack_grid
        self._ready = True

    def _kernel_options(self):
        # Model-static switches are compile-time constants, so each kernel carries only its own
        # code path and fixed-size loops unroll into registers.
        m = self.model
        channels = 4 * (9 + 3 * m.p + m.accum * m.p + m.carry_raw * m.p + m.thin_lock + 2 * m.coverage + m.history_age)
        smem = 16 * 4 * channels * 2          # the (32, 8) block holds 16 x 4 trunk cells
        self._stage_bytes = smem if self.layout == "nhwc" and smem <= 40960 else 0
        flags = dict(STAGE=bool(self._stage_bytes), S=m.scale, BASE=m.base_gate, CARRY=m.carry_raw, ACCUM=m.accum, HRES=m.hist_residual,
                     NEAREST=m.nearest_sample, CONS=m.conf_consistent, NCHW=self.layout == "nchw")
        if m.history_age:
            flags.update(HISTORY_AGE=True)
        if m.coverage:
            flags.update(COVERAGE=True)
        if m.depth_soft:
            flags.update(DEPTH_SOFT=True)
        if m.depth_soft_osc:
            flags.update(DEPTH_SOFT_OSC=True)
        if m.mv_dilate or m.depth_dilate or m.thin_lock:
            flags.update(MV_DILATE=m.mv_dilate, DEPTH_DILATE=m.depth_dilate, THIN_LOCK=m.thin_lock)
        return ("--fmad=false",) + tuple(f"-DK_{k}={int(v)}" for k, v in flags.items())

    def _prepare_tensorrt(self):
        try:
            import tensorrt as trt
            import onnx
        except (ImportError, OSError) as exc:
            raise RuntimeError("TensorRT trunk requires TensorRT >= 10 and ONNX for export") from exc
        if int(trt.__version__.split(".")[0]) < 10:
            raise RuntimeError("TensorRT trunk requires TensorRT >= 10")
        exported = io.BytesIO()
        try:
            with torch.cuda.device(self._device):
                torch.onnx.export(_TensorRTTrunkModule(self._net).eval(),
                                  (self._X, self._fold_weight, self._fold_bias), exported,
                                  input_names=["x", "head_w", "head_b"], output_names=["o"],
                                  opset_version=17, dynamo=False)
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
                builder = trt.Builder(self._trt_logger)
                flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
                network = builder.create_network(0 if flag is None else 1 << int(flag))
                parser = trt.OnnxParser(network, self._trt_logger)
                if not parser.parse(exported.getvalue()):
                    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")
                # LINEAR half I/O binds directly to contiguous NCHW torch buffers.
                for i in range(network.num_inputs):
                    network.get_input(i).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                for i in range(network.num_outputs):
                    network.get_output(i).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
                config.max_aux_streams = 0
                plan = builder.build_serialized_network(network, config)
                if plan is None:
                    raise RuntimeError("TensorRT engine build failed")
                self._trt_serialized = bytes(plan)
                self._trt_runtime = trt.Runtime(self._trt_logger)
                self._trt_engine = self._trt_runtime.deserialize_cuda_engine(self._trt_serialized)
                if self._trt_engine is None:
                    raise RuntimeError("TensorRT engine deserialization failed")
                self._trt_context = self._trt_engine.create_execution_context()
                if self._trt_context is None:
                    raise RuntimeError("TensorRT execution context creation failed")
                buffers = dict(x=self._X, head_w=self._fold_weight,
                               head_b=self._fold_bias, o=self._o)
                names = {self._trt_engine.get_tensor_name(i)
                         for i in range(self._trt_engine.num_io_tensors)}
                if names != set(buffers):
                    raise RuntimeError(f"Unexpected TensorRT I/O tensors: {names}")
                for name, tensor in buffers.items():
                    if (tuple(self._trt_engine.get_tensor_shape(name)) != tuple(tensor.shape)
                            or self._trt_engine.get_tensor_dtype(name) != trt.float16
                            or self._trt_engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                        raise RuntimeError(f"TensorRT tensor {name} requires contiguous half I/O")
                    if not self._trt_context.set_tensor_address(name, tensor.data_ptr()):
                        raise RuntimeError(f"TensorRT failed to bind {name}")
        except Exception as exc:
            raise RuntimeError(f"TensorRT trunk setup failed: {exc}") from exc

    def _execute_trunk(self):
        if self.trunk == "cudnn":
            self._graph.replay()
        elif not self._trt_context.execute_async_v3(self._cp.cuda.get_current_stream().ptr):
            raise RuntimeError("TensorRT trunk execution failed")

    def _middle(self):
        model = self._net
        x = F.relu_(model.stem(self._X))
        self._work = []
        skips = []
        for i, block in enumerate(model.enc):
            work = [x]
            for residual in block:
                a = F.relu_(residual.a(x))
                b = residual.b(a)
                x.add_(b)
                work.extend((a, b))
            self._work.append(work)
            if i < len(model.down):
                skips.append(x)
                x = F.relu_(model.down[i](x))
        for i in range(len(model.down) - 1, -1, -1):
            up = model.up[i](x)
            self._work[i].append(up)
            self._shuffle_add(up, skips[i])
            x = skips[i]
        x = F.relu_(model.fuse(x))
        self._o = F.conv2d(x, self._fold_weight, self._fold_bias)

    def _shuffle_add(self, up, skip):
        if self.layout == "nchw":
            skip.add_(F.pixel_shuffle(up, 2))
            return
        h, w = up.shape[-2:]
        c = skip.shape[1]
        np = self._np
        # Raw pointer arguments avoid creating array wrappers during capture.
        with self._cp.cuda.Device(self._device.index), self._stream_context():
            self._shuffle_kernel(((skip.numel() + 255) // 256,), (256,),
                                 (np.uint64(up.data_ptr()), np.uint64(skip.data_ptr()),
                                  np.int32(h), np.int32(w), np.int32(c)))

    def _update_frame(self, jitter):
        weight, bias, phase, offsets, kernels, windows = self._host_frames.get(jitter)
        if self.fold == "host":
            self._fold_weight.copy_(weight, non_blocking=True)
            self._fold_bias.copy_(bias, non_blocking=True)
        else:
            if torch.is_tensor(jitter):
                self._jitter.copy_(jitter.reshape(2))
            else:
                self._jitter_host.copy_(torch.as_tensor(jitter, dtype=torch.float32))
                self._jitter.copy_(self._jitter_host)
            if self._net.film is None:
                weight, bias = self._net.out.weight, self._net.out.bias
            else:
                weight, bias = _fold_head(self._net, self._jitter, torch.float16, self._device)
            self._fold_weight.copy_(weight)
            self._fold_bias.copy_(bias)
        if phase is not None:
            self._wc.copy_(phase, non_blocking=True)
        if offsets is not None:
            self._offsets.copy_(offsets, non_blocking=True)
        if kernels is not None:
            self._base_kernels.copy_(kernels, non_blocking=True)
            self._base_windows.copy_(windows, non_blocking=True)

    def _stage(self, state, lr_rgb, mv_lr, depth, jitter):
        h, w = self.model.render_size
        if lr_rgb.shape != (h, w, 3) or mv_lr.shape != (h, w, 2) or depth.shape != (h, w):
            raise ValueError("Expected render-resolution lr (h,w,3), motion (h,w,2), depth (h,w)")
        if any(t.device != self._device for t in (lr_rgb, mv_lr, depth, state.color)):
            raise ValueError("All frame tensors and state must be on the runtime CUDA device")
        if state.color.shape != (1, 3, *self.model.output_size):
            raise ValueError("Expected output-resolution state.color (1,3,H,W)")
        if self.model.accum:
            if state.conf is None:
                raise ValueError("Accumulation requires state.conf")
            if state.conf.device != self._device or state.conf.shape != (1, 1, *self.model.output_size):
                raise ValueError("Expected output-resolution state.conf (1,1,H,W) on the runtime CUDA device")
        if self.model.depth_test:
            if state.depth.device != self._device or state.depth.shape != (1, 1, h, w):
                raise ValueError("Expected render-resolution state.depth (1,1,h,w) on the runtime CUDA device")
            # Preserve borrowed depth before staging the next frame into its buffer.
            self._prev_depth.copy_(state.depth)
            self._prev_depth_half = self._np.int32(state.depth.dtype == torch.float16)
        self._lr_hwc.copy_(lr_rgb)
        self._mv.copy_(mv_lr)
        self._depth_hw.copy_(depth)
        self._update_frame(jitter)
        if state.color.data_ptr() == self._colors[1].data_ptr():
            source = 1
        else:
            source = 0
            if state.color.data_ptr() != self._colors[0].data_ptr():
                self._colors[0].copy_(state.color)
        if self.model.accum:
            if state.conf.data_ptr() != self._confs[source].data_ptr():
                self._confs[source].copy_(state.conf)
        if self.model.coverage:
            for i, tensor in enumerate((state.m1, state.m2, state.osc, state.luma)):
                if tensor is None or tensor.shape != (1, 1, h, w) or tensor.device != self._device:
                    raise ValueError("Expected float32 render-resolution coverage state")
                target = self._coverage_states[source][:, i:i+1]
                if target.data_ptr() != tensor.data_ptr():
                    target.copy_(tensor)
        if self.model.history_age:
            tensor = state.age
            if tensor is None or tensor.shape != (1, 1, h, w) or tensor.device != self._device:
                raise ValueError("Expected float32 render-resolution age state")
            if self._age_states[source].data_ptr() != tensor.data_ptr():
                self._age_states[source].copy_(tensor)
        return source

    def _pack(self, source, first):
        a = self._arrays
        self._pack_kernel(self._pack_grid, self._block, (
            self._history_arrays[source], a["_lr"], a["_mv"], a["_depth"], a["_prev_depth"],
            a["_X"], a["_hcl"], a["_hraw"], a["_cw"], a["_reset"], a["_rng"], a["_offsets"], *self._pack_sizes, self._box,
            self._first_flags[bool(first)], *self._flags, self._prev_depth_half,
            self._conf_motion, self._conf_m, self._hist_residual, self._nchw,
            self._carry_raw, self._nearest_sample,
            *((self._age_arrays[source], self._age_arrays[1-source], self._warped_age_array)
              if self.model.history_age else ()),
            *((self._coverage_arrays[source], self._coverage_arrays[1-source]) if self.model.coverage else ()),
            *self._osc_args,
            *self._robust_args), shared_mem=self._stage_bytes)

    def _resolve(self, target, write_rgb=False):
        a = self._arrays
        self._resolve_kernel(self._resolve_grid, self._block, (
            a["_o"], a["_lr"], a["_hcl"], a["_hraw"], a["_cw"], a["_reset"], a["_wc"], a["_rng"], a["_offsets"],
            a["_base_kernels"], a["_base_windows"],
            self._history_arrays[target], a["_rgb"],
            *self._resolve_sizes, self._hres_gain, self._conf_max, self._write_rgb_flags[bool(write_rgb)],
            *self._age_resolve_args))

    def _state(self, target, frame_index):
        state = _state_views(self._histories[target], self._feat, self._stored_depth,
                             frame_index, self.model.accum)
        if self.model.history_age:
            state.age = self._age_states[target]
        if self.model.coverage:
            state.m1, state.m2, state.osc, state.luma = self._coverage_states[target].chunk(4, 1)
        if self.model.carry_raw:
            state._display_rgb = self._rgb
        return state

    def _stream_context(self):
        return self._cp.cuda.ExternalStream(torch.cuda.current_stream(self._device).cuda_stream,
                                             device_id=self._device.index)

    @torch.no_grad()
    def step(self, state, lr_rgb, mv_lr, depth, jitter, *, write_rgb=False):
        """Return (rgb, FastState); rgb is None unless write_rgb or carry_raw.

        Colour/confidence are borrowed NCHW views of interleaved history. Use
        state_rgb(new) for display or materialize_state(new) to retain a frame.
        """
        jitter = _signed_jitter(self.model, jitter)
        if not self._ready:
            self._prepare(lr_rgb.device)
        source = self._stage(state, lr_rgb, mv_lr, depth, jitter)
        self._last_inputs = (lr_rgb, mv_lr, depth, jitter)
        with self._cp.cuda.Device(self._device.index), self._stream_context():
            self._pack(source, state.frame_index == 0)
            self._execute_trunk()
            self._resolve(1 - source, write_rgb)
        new = self._state(1 - source, state.frame_index + 1)
        return self._rgb if write_rgb or self.model.carry_raw else None, new

    @torch.no_grad()
    def timings(self, n=200, warmup=50, *, write_rgb=False):
        """Separate synchronised passes; total includes staging and state handling.

        Returns median_ms/p95_ms for each stage. Stage passes use fixed resident
        inputs; total advances recurrence. Trunk includes per-frame FiLM/weights.
        """
        if n < 1 or warmup < 0:
            raise ValueError("n must be positive and warmup nonnegative")
        if not self._ready:
            raise RuntimeError("Call step once with representative inputs before timings")
        inputs = self._last_inputs
        state = self.init_state()
        self._stage(state, *inputs)
        result = {}
        with self._cp.cuda.Device(self._device.index), self._stream_context():
            self._pack(0, False)
            self._execute_trunk()
            self._resolve(1, write_rgb)
            def total():
                nonlocal state
                _, state = self.step(state, *inputs, write_rgb=write_rgb)
            for name, fn in (("pack", lambda: self._pack(0, False)),
                             ("trunk", lambda: (self._update_frame(inputs[-1]), self._execute_trunk())),
                             ("resolve", lambda: self._resolve(1, write_rgb)), ("total", total)):
                values = []
                for i in range(n + warmup):
                    torch.cuda.synchronize(self._device)
                    start = time.perf_counter()
                    fn()
                    torch.cuda.synchronize(self._device)
                    if i >= warmup:
                        values.append((time.perf_counter() - start) * 1000)
                t = torch.tensor(values, dtype=torch.float64)
                result[name] = dict(median_ms=t.quantile(0.5).item(), p95_ms=t.quantile(0.95).item())
        return result
