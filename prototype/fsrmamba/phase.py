"""Phase accumulator: FSR's own decisions, predicted per sub-pixel at render resolution.

`fast.py` showed that moving the network to half render resolution is affordable, and
that it is not enough: with a free RGB residual on a nearest (or Lanczos) base and a hard
RGB min/max clamp it lands 0.6 dB and 0.006 SSIM below FSR full-frame, and neither more
capacity nor a recurrent state moves that. The deficit sits on edges. The diagnosis is
that the small trunk was asked to *synthesise colour*, and to do it without seeing the
evidence FSR's heuristics are built on.

This module keeps the cost structure (all convolutions at 1/2 render resolution or below,
output assembled per sub-pixel phase) and changes what is predicted. Nothing here predicts
colour by default. Per output sub-pixel the network outputs three gates:

    q      which current-frame resolve to use: FSR's Lanczos-2, or a sharper one (the same
           taps with kernel bias 1.8 -- `upsample.h`'s own lever)
    k      how wide the history rectification box is, in [1, box_max] neighbourhood stddevs
           (FSR's box scale, as `mamba.py` learns it)
    alpha  how much of the current frame to take

and the output is ``lerp(rectify(history, k), lerp(C0, C1, q), alpha)`` -- a combination of
real, deringed samples, so it cannot drift through the recurrence however it is driven.

Everything those gates need is computed from the same 4x4 low-res tap window with
per-frame, per-phase kernels (three tiny convolutions), exactly as `baseline._upsample`
does it: the resolve with the window rule of upsample.h:307, deringing against the taps
actually in each phase's window, and the Gaussian-weighted YCoCg mean/stddev box.

The network sees the *raw* warped history, the signed normalised luma innovation
``(Y(history) - Y(C0)) / sigma`` per phase, signed motion and per-phase validity -- the
evidence, rather than a history that was already clamped.

``residual=True`` adds a colour residual bounded by the local stddev (the discipline
`mamba.py`'s LR upsampler needed to stay stable), for measuring what colour synthesis
adds on top of the gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .colorspace import rgb_to_ycocg, ycocg_to_rgb
from .fast import _Res, _uv_grid
from .resolve import phase_kernels, window_bounds

__all__ = ["PhaseState", "PhaseAccumulator"]


@dataclass
class PhaseState:
    color: torch.Tensor   # (1, 3, H, W) history, YCoCg, output resolution
    feat: torch.Tensor    # (1, n_state, h/2, w/2) optional learned state
    frame_index: int = 0

    def detach(self) -> "PhaseState":
        return PhaseState(self.color.detach(), self.feat.detach(), self.frame_index)


class PhaseAccumulator(nn.Module):
    def __init__(self, render_size: tuple[int, int], output_size: tuple[int, int],
                 widths: tuple[int, ...] = (32, 64), depths: tuple[int, ...] = (1, 2),
                 n_state: int = 0, film: bool = True, residual: bool = False,
                 box_max: float = 3.0, sharp_bias: float = 1.8,
                 device: torch.device | str = "cpu", jitter_sign: float = 1.0) -> None:
        super().__init__()
        h, w = render_size
        s = output_size[0] // h
        if s * h != output_size[0] or s * w != output_size[1] or s < 1:
            raise ValueError(f"need an integer uniform scale, got {render_size} -> {output_size}")
        assert len(widths) == len(depths)
        self.render_size, self.output_size, self.scale = render_size, output_size, s
        self.jitter_sign = float(jitter_sign)
        if self.jitter_sign != 1.0:
            self.register_buffer("_jitter_sign", torch.tensor(self.jitter_sign))
        self.n_state, self.residual = n_state, residual
        self.box_max, self.sharp_bias = box_max, sharp_bias
        self.state_channels = n_state
        self.dev = torch.device(device)
        p = s * s
        self.p = p
        m = 2 ** len(widths)
        self._pad = ((-h) % m, (-w) % m)
        hp, wp = h + self._pad[0], w + self._pad[1]

        # lr 3 | raw history 3p | luma innovation p | motion 2 | inverse depth | validity p
        in_ch = 3 + 3 * p + p + 2 + 1 + p
        self.stem = nn.Conv2d(in_ch * 4 + n_state, widths[0], 1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, (c, d) in enumerate(zip(widths, depths)):
            self.enc.append(nn.Sequential(*[_Res(c) for _ in range(d)]))
            if i + 1 < len(widths):
                self.down.append(nn.Conv2d(c, widths[i + 1], 3, stride=2, padding=1))
                self.up.append(nn.Conv2d(widths[i + 1], c * 4, 1))
        self.fuse = nn.Conv2d(widths[0], widths[0], 3, padding=1)

        self.film = None
        if film:
            self.film = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2 * widths[0]))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)

        # Per render pixel: alpha p | q p | box scale p | (residual 3p). Four per trunk cell.
        self.n_px = (6 if residual else 3) * p
        self.gates = nn.Conv2d(widths[0], self.n_px * 4 + 2 * n_state, 1)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            b = torch.zeros(self.n_px * 4 + 2 * n_state)
            b[0: p * 4] = -3.05              # alpha ~0.045, FSR's converged blend rate
            b[p * 4: 2 * p * 4] = -3.0       # q ~0.05: start on the plain Lanczos resolve
            b[2 * p * 4: 3 * p * 4] = -0.4   # box scale toward the tight end, as mamba.py
            self.gates.bias.copy_(b)
        if residual:
            self.res_gain = nn.Parameter(torch.tensor(0.5))
        # How strongly a phase with a sample close to it this frame leans on the current
        # frame (FSR's per-frame Lanczos weight, learned scale).
        self.jit_gain = nn.Parameter(torch.tensor(1.0))

        ph = (torch.arange(s, dtype=torch.float32) + 0.5) / s
        py, px = torch.meshgrid(ph, ph, indexing="ij")
        tap = torch.arange(-2.0, 2.0)
        ty, tx = torch.meshgrid(tap, tap, indexing="ij")
        for name, t in (("_ph_x", px.reshape(p, 1, 1)), ("_ph_y", py.reshape(p, 1, 1)),
                        ("_tap_x", tx.reshape(1, 4, 4)), ("_tap_y", ty.reshape(1, 4, 4)),
                        ("_aniso", torch.tensor([1.7, 1.0, 1.0]).view(1, 3, 1, 1, 1)),
                        ("_uv_hr", _uv_grid(*output_size, "cpu")),
                        ("_uv_tr", _uv_grid(hp // 2, wp // 2, "cpu")
                         * torch.tensor([wp / w, hp / h])),
                        ("_tr_fac", torch.tensor([w / wp, h / hp])),
                        ("_wh", torch.tensor([float(w), float(h)]))):
            self.register_buffer(name, t, persistent=False)
        self._tr_size = (hp // 2, wp // 2)
        self.to(self.dev)

    def init_state(self, device=None) -> PhaseState:
        device = device or self.dev
        return PhaseState(torch.zeros(1, 3, *self.output_size, device=device),
                          torch.zeros(1, self.n_state, *self._tr_size, device=device))

    # -- per-frame, per-phase tap kernels (upsample.h:301-641) ---------------------------

    def _kernels(self, jitter, device):
        return phase_kernels(jitter, device, self._ph_x, self._ph_y,
                             self._tap_x, self._tap_y, self.sharp_bias)

    def reproject(self, state: PhaseState, mv_hr: torch.Tensor):
        """History moved along the motion vectors, and whether each output pixel's source
        was on screen. Coordinates stay float32: in half precision a 1920-wide grid is off
        by up to 0.2 px."""
        uv = self._uv_hr + mv_hr.float()
        hist = F.grid_sample(state.color.float(), uv * 2 - 1, mode="bilinear",
                             padding_mode="border", align_corners=False)
        valid = ((uv >= 0) & (uv < 1)).all(dim=-1)[:, None]
        return hist, valid

    def forward(self, state: PhaseState, lr_rgb: torch.Tensor, mv: torch.Tensor,
                depth: torch.Tensor, jitter, prev_depth_hr=None, hist=None, valid=None):
        """lr_rgb (h,w,3); mv (.,.,2) and depth (.,.) at render OR output resolution."""
        h, w = self.render_size
        s, p = self.scale, self.p
        if self.jitter_sign != 1.0:
            jitter = (jitter * self.jitter_sign if torch.is_tensor(jitter)
                      else (float(jitter[0]) * self.jitter_sign, float(jitter[1]) * self.jitter_sign))
        dt = torch.float16 if torch.is_autocast_enabled() else lr_rgb.dtype
        lr = rgb_to_ycocg(lr_rgb).permute(2, 0, 1).unsqueeze(0).to(dt)
        mv = mv.unsqueeze(0)
        if mv.shape[1] == h:
            mv_lr, mv_hr = mv, None
        else:
            mv_hr = mv
            mv_lr = F.avg_pool2d(mv.permute(0, 3, 1, 2), s).permute(0, 2, 3, 1)
        d = depth[None, None].to(dt)
        if d.shape[2] != h:
            d = d[:, :, ::s, ::s]

        if hist is None:
            if mv_hr is None:
                mv_hr = F.interpolate(mv_lr.permute(0, 3, 1, 2), scale_factor=s, mode="bilinear",
                                      align_corners=False).permute(0, 2, 3, 1)
            hist, valid = self.reproject(state, mv_hr)
        if state.frame_index == 0:
            valid = torch.zeros_like(valid)
        valid = F.pixel_unshuffle(valid.to(dt), s)                       # (1, p, h, w)
        h_raw = F.pixel_unshuffle(hist.to(dt), s).reshape(1, 3, p, h, w)

        # --- everything the current frame offers, from one 4x4 tap window -----------------
        j, k0, k1, g, prior, win = self._kernels(jitter, lr.device)
        taps = F.pad(lr.reshape(3, 1, h, w), (2, 1, 2, 1), mode="replicate")
        c0 = F.conv2d(taps, k0.to(dt)).unsqueeze(0)                       # (1, 3, p, h, w)
        c1 = F.conv2d(taps, k1.to(dt)).unsqueeze(0)
        mu = F.conv2d(taps, g.to(dt)).unsqueeze(0)
        ex2 = F.conv2d(taps * taps, g.to(dt)).unsqueeze(0)
        sigma = torch.sqrt((ex2 - mu * mu).abs() + 1e-8)
        # Dering each phase against the taps in ITS window (upsample.h:23-26).
        mn, mx = window_bounds(taps, win, h, w)
        c0 = torch.minimum(torch.maximum(c0, mn), mx)
        c1 = torch.minimum(torch.maximum(c1, mn), mx)

        # --- evidence for the network -------------------------------------------------
        vv = valid.unsqueeze(1)
        h_in = torch.where(vv > 0.5, h_raw, c0)
        innov = ((h_in[:, 0] - c0[:, 0]) / (sigma[:, 0] + 0.02)).clamp(-4, 4)
        mv_px = (mv_lr * self._wh).permute(0, 3, 1, 2).clamp(-8, 8).to(dt) * 0.125
        x = torch.cat((lr, h_in.reshape(1, 3 * p, h, w), innov, mv_px,
                       1.0 / d.clamp(min=1e-2), valid), dim=1)
        ph_, pw_ = self._pad
        if ph_ or pw_:
            x = F.pad(x, (0, pw_, 0, ph_), mode="replicate")
        x = F.pixel_unshuffle(x, 2)
        if self.n_state:
            mv_p, va_p = mv_lr.permute(0, 3, 1, 2), valid.amin(dim=1, keepdim=True)
            if ph_ or pw_:
                mv_p = F.pad(mv_p, (0, pw_, 0, ph_), mode="replicate")
                va_p = F.pad(va_p, (0, pw_, 0, ph_), mode="replicate")
            g_tr = (self._uv_tr + F.avg_pool2d(mv_p, 2).permute(0, 2, 3, 1).float()) * self._tr_fac * 2 - 1
            feat_prev = F.grid_sample(state.feat.float(), g_tr, mode="bilinear",
                                      padding_mode="border", align_corners=False).to(dt)
            feat_prev = feat_prev * -F.max_pool2d(-va_p, 2)
            x = torch.cat((x, feat_prev), dim=1)
        x = F.relu(self.stem(x))

        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = F.relu(self.down[i](x))
        for i in range(len(self.down) - 1, -1, -1):
            x = skips[i] + F.pixel_shuffle(self.up[i](x), 2)
        x = F.relu(self.fuse(x))
        if self.film is not None:
            ga, be = self.film(j.to(x.dtype).view(1, 2)).chunk(2, dim=1)
            x = x * (1 + ga.view(1, -1, 1, 1)) + be.view(1, -1, 1, 1)
        o = self.gates(x)

        # --- the three decisions, per sub-pixel phase -------------------------------------
        px = F.pixel_shuffle(o[:, : self.n_px * 4], 2)[..., :h, :w].to(dt)   # (1, n_px, h, w)
        alpha = torch.sigmoid(px[:, :p] + (self.jit_gain * prior).to(dt))
        alpha = torch.where(valid > 0.5, alpha, torch.ones_like(alpha)).unsqueeze(1)
        q = torch.sigmoid(px[:, p: 2 * p]).unsqueeze(1)
        k = (1.0 + (self.box_max - 1.0) * torch.sigmoid(px[:, 2 * p: 3 * p])).unsqueeze(1)

        cur = torch.lerp(c0, c1, q)
        if self.residual:
            res = torch.tanh(px[:, 3 * p:]).reshape(1, 3, p, h, w)
            cur = cur + res * sigma * self.res_gain.abs()
        half = sigma * self._aniso.to(dt) * k
        h_rect = torch.minimum(torch.maximum(h_in, mu - half), mu + half)
        out_un = torch.lerp(h_rect, cur, alpha)
        self._last_reset = 1 - valid.amin(dim=1, keepdim=True)
        self._last_alpha, self._last_q, self._last_k = alpha, q, k
        out = F.pixel_shuffle(out_un.reshape(1, 3 * p, h, w), s)             # YCoCg, (1,3,H,W)

        feat = state.feat
        if self.n_state:
            n = self.n_state
            so = o[:, self.n_px * 4:].float()
            a = torch.exp(-F.softplus(so[:, :n]))
            feat = (a * feat_prev.float() + (1 - a) * torch.tanh(so[:, n:])).to(state.feat.dtype)
        rgb = ycocg_to_rgb(out[0].permute(1, 2, 0)).clamp(min=0.0)
        return rgb, PhaseState(out, feat, state.frame_index + 1)
