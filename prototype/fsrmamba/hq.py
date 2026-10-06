"""Temporal kernel prediction with a recurrent shifted-window U-Net.

Research basis: Xiao et al., Neural Supersampling (SIGGRAPH 2020);
Mildenhall et al., Burst Denoising with Kernel Prediction Networks (CVPR 2018);
Liu et al., Swin Transformer (ICCV 2021). This is a new combination, not a
reproduction of those models or a claim about a proprietary upscaler.

The tensor-only trunk is exportable separately from packing and resolve.
Widths not divisible by 32 project attention to ceil(width/32) 32-wide heads;
in particular the default width 48 uses two heads, then projects back to 48.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
import torch.nn.functional as F

from .colorspace import rgb_to_ycocg
from .fast import _coverage_evidence, _uv_grid


@dataclass
class HQState:
    color: torch.Tensor
    feat: torch.Tensor
    depth: torch.Tensor
    age: torch.Tensor
    m1: torch.Tensor
    m2: torch.Tensor
    osc: torch.Tensor
    luma: torch.Tensor
    frame_index: int = 0

    def detach(self):
        return HQState(**{f.name: (v.detach() if torch.is_tensor(v) else v)
                          for f in fields(self) for v in (getattr(self, f.name),)})


def _windows(x):
    b, h, w, c = x.shape
    return x.reshape(b, h // 8, 8, w // 8, 8, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, 64, c)


class _WindowBlock(nn.Module):
    def __init__(self, width, size, shifted):
        super().__init__()
        self.size = size
        h, w = size
        self.padded = (h + (-h) % 8, w + (-w) % 8)
        ph, pw = self.padded
        self.shift = (4 if shifted and h > 8 else 0,
                      4 if shifted and w > 8 else 0)
        self.heads = math.ceil(width / 32)
        self.inner = self.heads * 32
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * self.inner)
        self.proj = nn.Linear(self.inner, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.relative_bias = nn.Parameter(torch.zeros(225, self.heads))
        y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        coords = torch.stack((y.flatten(), x.flatten()), -1)
        delta = coords[:, None] - coords[None, :]
        self.register_buffer("relative_index", (delta[..., 0] + 7) * 15 + delta[..., 1] + 7,
                             persistent=False)
        # Rolled coordinates expose cyclic wrap boundaries without region slicing.
        yy, xx = torch.meshgrid(torch.arange(ph), torch.arange(pw), indexing="ij")
        sy, sx = self.shift
        coords = torch.stack((yy, xx), -1)[None]
        coords = torch.roll(coords, (-sy, -sx), (1, 2))
        tokens = _windows(coords)
        wrap = ((tokens[:, :, None] - tokens[:, None, :]).abs() >= 8).any(-1)
        valid = (tokens[..., 0] < h) & (tokens[..., 1] < w)
        mask = wrap | ~valid[:, None, :]
        self.register_buffer("mask", torch.zeros_like(mask, dtype=torch.float32).masked_fill(
            mask, -10000.)[:, None], persistent=False)

    def forward(self, x):
        h, w = self.size
        ph, pw = self.padded
        sy, sx = self.shift
        shortcut = x.permute(0, 2, 3, 1)
        z = F.pad(self.norm1(shortcut), (0, 0, 0, pw - w, 0, ph - h))
        z = _windows(torch.roll(z, (-sy, -sx), (1, 2)))
        qkv = self.qkv(z).reshape(-1, 64, 3, self.heads, 32).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)[None]
        # Float32 softmax protects the reference path under autocast.
        attn = ((q * (32 ** -.5)) @ k.transpose(-1, -2)).float()
        attn = (attn + bias.float() + self.mask).softmax(-1).to(v.dtype)
        z = self.proj((attn @ v).transpose(1, 2).reshape(-1, 64, self.inner))
        z = z.reshape(1, ph // 8, pw // 8, 8, 8, -1).permute(
            0, 1, 3, 2, 4, 5).reshape(1, ph, pw, -1)
        z = shortcut + torch.roll(z, (sy, sx), (1, 2))[:, :h, :w]
        return (z + self.mlp(self.norm2(z))).permute(0, 3, 1, 2)


class _ConvBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return x + self.pointwise(F.gelu(self.depthwise(x)))


class _Merge(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm = nn.LayerNorm(4 * cin)
        self.proj = nn.Conv2d(4 * cin, cout, 1)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2).permute(0, 2, 3, 1)
        return self.proj(self.norm(x).permute(0, 3, 1, 2))


class HQTrunk(nn.Module):
    """Static NCHW inputs: padded R-grid pack, signed jitter (1,2), warped R/8 state.

    Returns eight parameter planes per output phase at padded R, and new R/8 state.
    Encoder/decoder attention counts match at R/4 and R/8; R/16 is the bottleneck.
    """
    def __init__(self, padded_size, in_channels, scale, c0, widths, depths, n_state):
        super().__init__()
        channels = (c0, *widths)
        self.sizes = [tuple(v // (2 ** (i + 1)) for v in padded_size) for i in range(4)]
        self.stem = nn.Conv2d(4 * in_channels, c0, 1)
        self.enc0 = nn.Sequential(_ConvBlock(c0), _ConvBlock(c0))
        self.dec0 = nn.Sequential(_ConvBlock(c0), _ConvBlock(c0))
        self.down = nn.ModuleList([_Merge(a, b) for a, b in zip(channels, channels[1:])])
        self.up = nn.ModuleList([nn.Conv2d(b, 4 * a, 1) for a, b in zip(channels, channels[1:])])
        def stage(i):
            return nn.Sequential(*[_WindowBlock(widths[i], self.sizes[i + 1], j % 2 == 1)
                                   for j in range(depths[i])])
        self.enc = nn.ModuleList([stage(i) for i in range(3)])
        self.dec = nn.ModuleList([stage(i) for i in range(2)])
        self.hidden_in = nn.Conv2d(widths[1] + n_state, widths[1], 1)
        self.hidden_out = nn.Conv2d(widths[1], n_state, 1)
        self.film = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 2 * c0))
        self.head = nn.Conv2d(c0, 4 * 8 * scale * scale, 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, packed, jitter, hidden):
        x0 = self.stem(F.pixel_unshuffle(packed, 2))
        gain, bias = self.film(jitter).chunk(2, -1)
        x0 = self.enc0(x0 * (1 + .1 * gain[:, :, None, None]) + .1 * bias[:, :, None, None])
        x1 = self.enc[0](self.down[0](x0))
        x2 = self.enc[1](self.hidden_in(torch.cat((self.down[1](x1), hidden), 1)))
        x3 = self.enc[2](self.down[2](x2))
        x2 = self.dec[1](x2 + F.pixel_shuffle(self.up[2](x3), 2))
        next_hidden = torch.tanh(self.hidden_out(x2))
        x1 = self.dec[0](x1 + F.pixel_shuffle(self.up[1](x2), 2))
        x0 = self.dec0(x0 + F.pixel_shuffle(self.up[0](x1), 2))
        return F.pixel_shuffle(self.head(x0), 2), next_hidden

    def macs(self):
        """Analytic multiply-accumulates (one multiply+add = one MAC), batch one.

        Includes padded attention tokens, QK/AV, conv/linear layers and FiLM MLP.
        Excludes bias, norm, activation, softmax, reshapes and elementwise operations.
        """
        total = 0
        def conv(layer, size):
            return math.prod(size) * layer.weight.numel()
        total += conv(self.stem, self.sizes[0]) + conv(self.head, self.sizes[0])
        for stage in (self.enc0, self.dec0):
            for block in stage:
                total += conv(block.depthwise, self.sizes[0]) + conv(block.pointwise, self.sizes[0])
        for i in range(3):
            total += conv(self.down[i].proj, self.sizes[i + 1])
            total += conv(self.up[i], self.sizes[i + 1])
        total += conv(self.hidden_in, self.sizes[2]) + conv(self.hidden_out, self.sizes[2])
        for layer in self.film:
            if isinstance(layer, nn.Linear):
                total += layer.weight.numel()
        for block in self.modules():
            if isinstance(block, _WindowBlock):
                c, a = block.norm1.normalized_shape[0], block.inner
                total += math.prod(block.padded) * (4 * c * a + 2 * 64 * a)
                total += math.prod(block.size) * 8 * c * c
        return total


class HQAccumulator(nn.Module):
    """Reinhard RGB, UV motion (current -> previous), reversed Z, Fast jitter convention.

    State age is an integer frame count (nearest reprojected), capped at 32;
    invalid history has age zero before update and one after the current sample.
    Frame mean Y (floor .05) normalizes both current and historical trunk inputs;
    resolve and stored RGB remain in the original Reinhard space. No exposure
    metadata is inferred, and no extra tonemapping is applied.
    """
    def __init__(self, render_size, output_size, c0=24, widths=(48, 96, 128),
                 depths=(2, 2, 2), n_state=16, jitter_sign=1.0, device="cpu"):
        super().__init__()
        h, w = render_size
        s = output_size[0] // h
        if min(h, w, s) < 1 or tuple(output_size) != (s * h, s * w):
            raise ValueError("HQ requires a positive integer uniform upscale")
        if (len(widths) != 3 or len(depths) != 3 or min(c0, n_state, *widths) < 1
                or min(depths) < 1 or jitter_sign not in (-1., 1.)):
            raise ValueError("HQ requires three positive widths/depths, positive C0/H and jitter_sign +/-1")
        self.render_size, self.output_size = tuple(render_size), tuple(output_size)
        self.scale, self.p = s, s * s
        self.c0, self.widths, self.depths = c0, tuple(widths), tuple(depths)
        self.n_state = self.state_channels = n_state
        self.jitter_sign = float(jitter_sign)
        self.padded_size = (h + (-h) % 16, w + (-w) % 16)
        hp, wp = self.padded_size
        self.in_channels = 12 + 4 * self.p
        self.trunk = HQTrunk(self.padded_size, self.in_channels, s, c0, widths, depths, n_state)
        self.register_buffer("_hq_spec", torch.tensor([c0, *widths, *depths, n_state], dtype=torch.int64))
        self.register_buffer("_jitter_sign", torch.tensor(self.jitter_sign))
        self.register_buffer("_uv_lr", _uv_grid(h, w, device), persistent=False)
        self.register_buffer("_uv_hr", _uv_grid(*output_size, device), persistent=False)
        self.register_buffer("_uv_hidden", _uv_grid(hp // 8, wp // 8, device)
                             * torch.tensor([wp / w, hp / h], device=device), persistent=False)
        self.to(device)

    def init_state(self, device=None):
        device = device or self._uv_lr.device
        def zero(c, size):
            return torch.zeros(1, c, *size, device=device)
        return HQState(zero(3, self.output_size),
                       zero(self.n_state, tuple(v // 8 for v in self.padded_size)),
                       torch.ones(1, 1, *self.render_size, device=device),
                       zero(1, self.output_size), *(zero(1, self.render_size) for _ in range(4)))

    def _pad(self, x):
        h, w = self.render_size
        hp, wp = self.padded_size
        return F.pad(x, (0, wp - w, 0, hp - h), mode="replicate")

    @staticmethod
    def _warp(x, uv, mode="bilinear"):
        return F.grid_sample(x.float(), uv.float() * 2 - 1, mode=mode,
                             padding_mode="border", align_corners=False)

    def reproject(self, state, mv_hr):
        return self._warp(state.color, self._uv_hr + mv_hr, "bicubic")

    def _kernel(self, params, jitter):
        """25 normalized weights per output pixel, centred on its nearest LR sample.

        Coordinates are in LR pixels; positive signed jitter places samples at
        centre-jitter, matching FastAccumulator. A zero scale is the delta limit.
        """
        h, w = self.render_size
        xy = self._uv_hr * self._uv_hr.new_tensor([w, h]) - .5 + jitter.reshape(1, 1, 1, 2)
        centre = torch.floor(xy + .5)
        theta = math.pi * torch.tanh(params[:, 1:2])
        scales = .15 + 1.85 * torch.sigmoid(params[:, 2:4])
        return self.gaussian_weights(xy, centre, theta, scales), centre

    @staticmethod
    def gaussian_weights(xy, centre, theta, scales):
        dx = (centre[..., 0] - xy[..., 0])[:, None]
        dy = (centre[..., 1] - xy[..., 1])[:, None]
        cs, sn = theta.cos(), theta.sin()
        a, b = scales[:, :1].clamp(min=1e-4), scales[:, 1:].clamp(min=1e-4)
        logits = []
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                u = cs * (dx + ox) + sn * (dy + oy)
                v = -sn * (dx + ox) + cs * (dy + oy)
                logits.append(-.5 * ((u / a).square() + (v / b).square()))
        weights = torch.cat(logits, 1).softmax(1)
        delta = torch.zeros_like(weights)
        delta[:, 12] = 1
        return torch.where((scales <= 0).all(1, keepdim=True), delta, weights)

    def resolve(self, lr, history, blend, weights, centre, slack, residual, reset):
        """Reference dynamic filter. blend=1 preserves history bit-for-bit.

        Slack/residual operate on the current-weighted path so the exact history
        endpoint is retained even when history is outside the local clamp box.
        """
        h, w = self.render_size
        current = torch.zeros_like(history)
        for i, (oy, ox) in enumerate((y, x) for y in range(-2, 3) for x in range(-2, 3)):
            ix = (centre[..., 0].long() + ox).clamp(0, w - 1)
            iy = (centre[..., 1].long() + oy).clamp(0, h - 1)
            samples = lr[0, :, iy[0], ix[0]][None]
            current = current + weights[:, i:i + 1] * samples
        lo = F.interpolate(-F.max_pool2d(-lr, 5, 1, 2), size=self.output_size, mode="nearest")
        hi = F.interpolate(F.max_pool2d(lr, 5, 1, 2), size=self.output_size, mode="nearest")
        span = hi - lo
        rectified = torch.minimum(torch.maximum(history, lo - slack * span), hi + slack * span)
        blend = torch.where(reset, 0., blend)
        rectified = history + (1 - blend) * (rectified - history)
        candidate = (current + residual * (span + .01)).clamp(0, 1 - 1e-6)
        out = blend * rectified + (1 - blend) * candidate
        return torch.where(blend == 1, history, out.clamp(0, 1 - 1e-6))

    def forward(self, state, lr_rgb, mv, depth, jitter, prev_depth_hr=None, hist=None):
        h, w = self.render_size
        lr = lr_rgb.permute(2, 0, 1)[None].float()
        motion = mv.permute(2, 0, 1)[None].float()
        ml = F.interpolate(motion, size=(h, w), mode="bilinear", align_corners=False)
        mh = F.interpolate(motion, size=self.output_size, mode="bilinear", align_corners=False)
        d = F.interpolate(depth[None, None].float(), size=(h, w), mode="nearest")
        uv, uvh = self._uv_lr + ml.permute(0, 2, 3, 1), self._uv_hr + mh.permute(0, 2, 3, 1)
        old_depth = self._warp(state.depth, uv, "nearest")
        dmin, dmax = -F.max_pool2d(-d, 3, 1, 1), F.max_pool2d(d, 3, 1, 1)
        mismatch = (old_depth < .9 * dmin) | (old_depth > 1.1 * dmax)
        reset = mismatch | ~((uv >= 0) & (uv < 1)).all(-1)[:, None]
        if state.frame_index == 0:
            reset = torch.ones_like(reset)
            mismatch = torch.ones_like(mismatch)
        reset_hr = F.interpolate(reset.float(), size=self.output_size, mode="nearest").bool()
        reset_hr = reset_hr | ~((uvh >= 0) & (uvh < 1)).all(-1)[:, None]
        history = self.reproject(state, mh.permute(0, 2, 3, 1)) if hist is None else hist.float()
        history = history.clamp(0, 1 - 1e-6)
        history = torch.where(reset_hr, 0., history)
        age = torch.where(reset_hr, 0., self._warp(state.age, uvh, "nearest")).clamp(0, 32)
        previous = self._warp(torch.cat((state.m1, state.m2, state.osc, state.luma), 1), uv, "nearest").chunk(4, 1)
        rng = F.max_pool2d(lr, 3, 1, 1) + F.max_pool2d(-lr, 3, 1, 1)
        evidence, moments = _coverage_evidence(lr, rng, previous, reset.float())
        yc = rgb_to_ycocg(lr.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        exposure = yc[:, :1].mean((2, 3), keepdim=True).detach().clamp(min=.05)
        yh = rgb_to_ycocg(history.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        speed = (ml * ml.new_tensor([w, h])[None, :, None, None]).square().sum(1, keepdim=True).sqrt()
        packed = self._pad(torch.cat((yc / exposure, F.pixel_unshuffle(yh / exposure, self.scale),
                                     F.pixel_unshuffle(age / 32, self.scale), mismatch.float(),
                                     evidence, d, (speed / 16).clamp(0, 1), reset.float(), rng / exposure), 1))
        hidden_mv = F.interpolate(self._pad(ml), size=state.feat.shape[-2:], mode="bilinear", align_corners=False)
        hidden_uv = self._uv_hidden + hidden_mv.permute(0, 2, 3, 1)
        hp, wp = self.padded_size
        hidden = self._warp(state.feat, hidden_uv * hidden_uv.new_tensor([w / wp, h / hp]))
        hidden_reset = F.max_pool2d(self._pad(reset.float()), 8, 8).bool()
        hidden = torch.where(hidden_reset, 0., hidden)
        signed = torch.as_tensor(jitter, device=lr.device, dtype=torch.float32).reshape(1, 2) * self.jitter_sign
        params, feat = self.trunk(packed, signed, hidden)
        params = F.pixel_shuffle(params[:, :, :h, :w], self.scale).float()
        blend = torch.sigmoid(params[:, :1])
        weights, centre = self._kernel(params, signed)
        out = self.resolve(lr, history, blend, weights, centre, F.softplus(params[:, 4:5]),
                           .05 * torch.tanh(params[:, 5:8]), reset_hr)
        self._last_reset = reset_hr.float()
        self._last_alpha = torch.where(reset_hr, 1., 1 - blend)
        new = HQState(out, feat, d, (age + 1).clamp(max=32), *moments, state.frame_index + 1)
        return out[0].permute(1, 2, 0), new

    def cost(self):
        """Trunk MACs plus the 25-tap RGB weighted sum; other resolve math excluded."""
        trunk = self.trunk.macs()
        resolve = math.prod(self.output_size) * 25 * 3
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "trunk_macs": trunk, "filter_macs": resolve,
                "trunk_and_filter_macs": trunk + resolve}
