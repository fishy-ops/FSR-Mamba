"""Frozen pre-robustness pack/resolve arithmetic for exact legacy regression."""
import torch
import torch.nn.functional as F

from fsrmamba.fused import (_validate, _sample, _sample_nearest, _phase_gather,
                           _phase_scatter, _phase_offsets, _phase_weights, _base_constants)


def ref_pack(model, state, lr_rgb, mv_lr, depth, jitter):
    """Slow executable specification of pack; no grid_sample/pixel_unshuffle."""
    _validate(model)
    h, w = model.render_size
    s, p = model.scale, model.p
    dt, dev = lr_rgb.dtype, lr_rgb.device
    lr = lr_rgb.permute(2, 0, 1)[None]
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
    d = depth[None, None].to(dt)
    if model.depth_test:
        dgx = (u * 2 - 1).to(state.depth.dtype).float()
        dgy = (v * 2 - 1).to(state.depth.dtype).float()
        prev_d = _sample_nearest(state.depth, ((dgx + 1) * w - 1) / 2,
                                 ((dgy + 1) * h - 1) / 2)
        d_max = F.max_pool2d(d, 3, stride=1, padding=1)
        d_min = -F.max_pool2d(-d, 3, stride=1, padding=1)
        reset = reset | (prev_d < d_min * 0.9) | (prev_d > d_max * 1.1)
    reset = reset.to(dt)
    if state.frame_index == 0:
        reset.fill_(1)
    taps = [lr[..., (y.long() + dy).clamp(0, h - 1), (x.long() + dx).clamp(0, w - 1)]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    mn, mx = torch.stack(taps).amin(0), torch.stack(taps).amax(0)
    rng = mx - mn
    slack = rng * model.box_slack.abs()
    hist = _phase_gather(hist, s).reshape(1, 3, p, h, w)
    h_cl = torch.minimum(torch.maximum(hist, (mn - slack)[:, :, None]),
                         (mx + slack)[:, :, None]).reshape(1, 3 * p, h, w)
    motion_lr = mv_lr.float()
    speed_pixels = ((motion_lr[..., 0] * w).square() + (motion_lr[..., 1] * h).square()).sqrt()
    speed = (speed_pixels * 0.1).clamp(0, 1)[None, None].to(dt)
    parts = [lr, h_cl, rng, 1.0 / d.clamp(min=1e-2), speed, reset]
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
    cat = torch.cat(parts, 1)
    hp, wp = h + model._pad[0], w + model._pad[1]
    # Each padded pixel owns one channel phase; its source is edge-clamped.
    yp = torch.arange(hp, device=dev).clamp(max=h - 1)
    xp = torch.arange(wp, device=dev).clamp(max=w - 1)
    padded = cat[..., yp[:, None], xp[None, :]]
    X = _phase_gather(padded, 2).contiguous(memory_format=torch.channels_last)
    return dict(X=X, h_cl=h_cl.contiguous(), conf_w=conf_w, reset=reset, rng=rng,
                h_raw=hist.reshape(1, 3 * p, h, w).contiguous() if model.carry_raw else None)



def ref_resolve(model, o, lr_rgb, packed, jitter):
    """Slow executable specification of resolve, using explicit phase indexing."""
    h, w = model.render_size
    p, s = model.p, model.scale
    y, x = torch.meshgrid(torch.arange(h, device=o.device), torch.arange(w, device=o.device), indexing="ij")
    k = torch.arange(model.n_px, device=o.device)[:, None, None]
    px = o[0, k * 4 + (y % 2) * 2 + x % 2, y // 2, x // 2][None].to(lr_rgb.dtype)
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
        base = torch.lerp(lanczos, base, gate)
    cur = base + px[:, :3 * p].reshape(1, 3, p, h, w)
    logit = px[:, 3 * p:4 * p]
    conf = None
    if model.accum:
        wc = _phase_weights(model, jitter, lr_rgb.dtype, o.device)
        wh = packed["conf_w"] * torch.sigmoid(px[:, -p:])
        if model.conf_consistent:
            wc = wc * torch.exp(logit.float().clamp(-2.0, 2.0)).to(lr_rgb.dtype)
            logit = (torch.log(wc) - torch.log(wh + 1e-3)).to(lr_rgb.dtype)
        else:
            logit = (logit + torch.log(wc) - torch.log(wh + 1e-3)).to(lr_rgb.dtype)
        conf = _phase_scatter((wh + wc).clamp(max=model.conf_max), s).float()
    alpha = torch.where(packed["reset"] > 0.5, torch.ones_like(logit), torch.sigmoid(logit))
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

