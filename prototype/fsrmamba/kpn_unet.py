"""Temporal reconstruction from real jittered samples and rectified output history.

Deployment: export ``trunk`` independently. Its DirectML operations are conv,
max-pool, bilinear resample, add and ReLU. Replicate-pad the render extent
to multiples of 16 * trunk_stride, then crop the shuffled parameter map.
Packing, cubic history reprojection, depth rejection, the 9/25-tap Gaussian and
blend belong in compute shaders. Coordinates/filter arithmetic stay float32.
There is no learned recurrent state: retain output RGB, render-grid age and
previous depth (reprojection metadata), plus a frame/reset counter.
This is a kernel-prediction design, not a reproduction of a proprietary model.

The 16 render-grid inputs, in order, are:
  0:3   current YCoCg (Reinhard RGB is supplied by the caller; Y is also the
        separate luma view used for range/disagreement, not a duplicate input);
  3:6   reprojected YCoCg averaged over each 2x2 output phase block;
  6:10  reprojected Y for phases (top-left, top-right, bottom-left, bottom-right);
  10    signed history-minus-current Y / (local 3x3 Y range + .02), clipped +/-4;
  11    depth mismatch or offscreen/first-frame disocclusion;
  12    motion length in render pixels / 10, clipped to [0,1];
  13    log2(1 + reprojected age) / 5 (age capped at 32);
  14:16 signed jitter x,y, constant over the frame.
Inputs already occupy model space: no second tonemap or exposure normalization.

The seven output parameters are log sigma along the two rotated axes, angle
(radians), current alpha, clamp slack, residual luma and residual red/blue chroma.
The last two produce RGB logits (Y+Co, Y, Y-Co), followed by tanh * .05.
An independent three-channel RGB residual would require an eight-channel head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .colorspace import rgb_to_ycocg
from .fast import _nearest_depth, _uv_grid


DEFAULT_WIDTHS = (24, 32, 64, 96, 128, 192)
LITE_WIDTHS = (16, 24, 48, 64, 96, 128)


def catmull_sample(x, uv):
    """Separable Catmull-Rom (a=-.5), texel-centred UVs and clamped border taps."""
    x, uv = x.float(), uv.float()
    n, c, h, w = x.shape
    p = uv * uv.new_tensor([w, h]) - .5
    base = p.floor()
    t = p - base
    def weights(t):
        return (-.5*t + t*t - .5*t*t*t, 1 - 2.5*t*t + 1.5*t*t*t,
                .5*t + 2*t*t - 1.5*t*t*t, -.5*t*t + .5*t*t*t)
    wx, wy = weights(t[..., 0]), weights(t[..., 1])
    ix, iy = base[..., 0].long(), base[..., 1].long()
    out = x.new_zeros(n, c, *uv.shape[1:3])
    for j in range(4):
        row = torch.zeros_like(out)
        for i in range(4):
            index = ((iy+j-1).clamp(0, h-1)*w + (ix+i-1).clamp(0, w-1)).flatten(1)
            sample = x.flatten(2).gather(2, index[:, None].expand(-1, c, -1)).reshape_as(out)
            row = row + sample * wx[i][:, None]
        out = out + row * wy[j][:, None]
    return out


@dataclass
class KPNState:
    color: torch.Tensor
    age: torch.Tensor
    depth: torch.Tensor
    frame_index: int = 0

    def detach(self):
        return KPNState(self.color.detach(), self.age.detach(), self.depth.detach(), self.frame_index)


class KPNTrunk(nn.Module):
    def __init__(self, widths, lite, parameters, trunk_stride=1):
        super().__init__()
        self.trunk_stride = trunk_stride
        channels = (16 * trunk_stride ** 2, *widths[:5])
        self.enc = nn.ModuleList([nn.Conv2d(a, b, k, padding=k // 2)
                                  for a, b, k in zip(channels, channels[1:], (3, 1, 3, 3, 3))])
        self.bottleneck = nn.Conv2d(widths[4], widths[5], 1)
        self.skip = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = widths[5]
        for cout in reversed(widths[:4]):
            self.skip.append(nn.Conv2d(cout, cin, 1) if cin != cout else nn.Identity())
            layers = [nn.Conv2d(cin, cout, 1), nn.ReLU()]
            if not lite:
                layers.extend((nn.Conv2d(cout, cout, 1), nn.ReLU()))
            self.dec.append(nn.Sequential(*layers))
            cin = cout
        phases = (2 * trunk_stride) ** 2
        self.head = nn.Conv2d(widths[0], phases * parameters, 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        # Start anisotropic so the angle head receives a gradient immediately.
        with torch.no_grad():
            self.head.bias[:phases].fill_(math.log(.9))
            self.head.bias[phases:2 * phases].fill_(math.log(.6))

    def forward(self, packed):
        skips = []
        x = F.pixel_unshuffle(packed, self.trunk_stride) if self.trunk_stride == 2 else packed
        for i, layer in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = F.relu(layer(x))
            skips.append(x)
        x = F.relu(self.bottleneck(x))
        for skip, project, stage in zip(reversed(skips[:4]), self.skip, self.dec):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = stage(x + project(skip))
        return F.pixel_shuffle(self.head(x), 2 * self.trunk_stride)

    def macs(self, padded_size):
        """Conv MACs, including skip projections and head; one multiply-add = 1 MAC."""
        sizes = [math.prod(padded_size) // (self.trunk_stride ** 2 * 4 ** i) for i in range(5)]
        total = sum(size * layer.weight.numel() for size, layer in zip(sizes, self.enc))
        total += sizes[4] * self.bottleneck.weight.numel()
        for size, project, stage in zip(reversed(sizes[:4]), self.skip, self.dec):
            if isinstance(project, nn.Conv2d):
                total += size * project.weight.numel()
            total += size * sum(layer.weight.numel() for layer in stage if isinstance(layer, nn.Conv2d))
        return total + sizes[0] * self.head.weight.numel()


class KPNAccumulator(nn.Module):
    """2x accumulator with UV current-to-previous motion and reversed-Z depth.

    Age counts valid transitions since reset: reset stores zero, then increments
    to 32. Soft depth mode still clears age on mismatch, but preserves color;
    offscreen and first-frame resets always force current weight to one.
    """
    def __init__(self, render_size, output_size, widths=None, lite=False, residual=True,
                 mv_dilate=True, depth_soft=False, jitter_sign=1.0, sigma_min=.3, proximity=0.,
                 proximity_gain=2., device="cpu", trunk_stride=1, taps=5, history_filter="bicubic"):
        super().__init__()
        widths = tuple(widths if widths is not None else LITE_WIDTHS if lite else DEFAULT_WIDTHS)
        if (len(widths) != 6 or any(not isinstance(c, int) or c < 1 for c in widths)
                or jitter_sign not in (-1., 1.)):
            raise ValueError("KPN requires six positive integer widths and jitter_sign +/-1")
        h, w = render_size
        if min(h, w) < 1 or tuple(output_size) != (2 * h, 2 * w):
            raise ValueError("KPN requires a positive render size and exactly 2x output")
        if trunk_stride not in (1, 2) or taps not in (3, 5) or history_filter not in ("bicubic", "catmull"):
            raise ValueError("KPN requires stride 1/2, taps 3/5 and bicubic/catmull history")
        self.trunk_stride, self.taps, self.history_filter = trunk_stride, taps, history_filter
        if trunk_stride != 1:
            self.register_buffer("_kpn_stride", torch.tensor(trunk_stride))
        if taps != 5:
            self.register_buffer("_kpn_taps", torch.tensor(taps))
        if history_filter != "bicubic":
            self.register_buffer("_kpn_catmull", torch.tensor(1))
        self.render_size, self.output_size = tuple(render_size), tuple(output_size)
        self.widths, self.lite, self.residual = widths, bool(lite), bool(residual)
        self.mv_dilate, self.depth_soft = bool(mv_dilate), bool(depth_soft)
        self.jitter_sign = float(jitter_sign)
        self.sigma_min = float(sigma_min)
        if self.sigma_min != .3:   # stored only when non-default so older checkpoints load unchanged
            self.register_buffer("_sigma_min", torch.tensor(self.sigma_min))
        # Proximity-weighted accumulation (as FSR 2 / DLSS accumulate): a frame updates an
        # output pixel in proportion to how close its nearest jittered sample landed (as odds
        # against the history), so
        # averaging over frames converges to output-pixel detail instead of a box blur two
        # output pixels wide. ``proximity`` is that Gaussian's sigma in output pixels; 0 = off.
        self.proximity, self.proximity_gain = float(proximity), float(proximity_gain)
        if self.proximity:
            self.register_buffer("_proximity", torch.tensor([self.proximity, self.proximity_gain]))
        self.scale, self.state_channels, self.in_channels = 2, 0, 16
        multiple = 16 * trunk_stride
        self.padded_size = (h + (-h) % multiple, w + (-w) % multiple)
        self.trunk = KPNTrunk(widths, lite, 7, trunk_stride)
        self.register_buffer("_kpn_spec", torch.tensor([*widths, lite, residual, mv_dilate, depth_soft],
                                                       dtype=torch.int64))
        self.register_buffer("_jitter_sign", torch.tensor(self.jitter_sign))
        self.register_buffer("_uv_lr", _uv_grid(h, w, device), persistent=False)
        self.register_buffer("_uv_hr", _uv_grid(*output_size, device), persistent=False)
        yy, xx = torch.meshgrid(torch.arange(2*h, device=device, dtype=torch.float32),
                                torch.arange(2*w, device=device, dtype=torch.float32), indexing="ij")
        self.register_buffer("_sample_xy", (torch.stack((xx, yy), -1)[None]+.5)*.5-.5, persistent=False)
        self.to(device)

    def init_state(self, device=None):
        device = device or self._uv_lr.device
        return KPNState(torch.zeros(1, 3, *self.output_size, device=device),
                        torch.zeros(1, 1, *self.render_size, device=device),
                        torch.ones(1, 1, *self.render_size, device=device))

    @staticmethod
    def _warp(x, uv, mode="nearest"):
        return F.grid_sample(x.float(), uv.float() * 2 - 1, mode=mode,
                             padding_mode="border", align_corners=False)

    def reproject(self, state, mv_hr):
        uv = self._uv_hr + mv_hr.float()
        return (catmull_sample(state.color, uv) if self.history_filter == "catmull"
                else self._warp(state.color, uv, "bicubic"))

    def _kernel(self, params, signed_jitter):
        """Log-sigmas, angle -> weights at true sample positions in render pixels.

        Fast's signed convention places sample i at i+.5-signed_jitter; thus
        jitter_sign=-1 places it at texel centre + the capture's supplied jitter.
        Centre the kernel support on the nearest sample to each output pixel.
        """
        h, w = self.render_size
        xy = self._sample_xy + signed_jitter.reshape(1, 1, 1, 2)
        centre = torch.floor(xy + .5)
        centre = torch.minimum(centre.clamp(min=0), centre.new_tensor([w - 1, h - 1]))
        valid = []
        for oy in range(-(self.taps // 2), self.taps // 2 + 1):
            for ox in range(-(self.taps // 2), self.taps // 2 + 1):
                pos = centre + centre.new_tensor([ox, oy])
                valid.append(((pos[..., 0] >= 0) & (pos[..., 0] < w)
                              & (pos[..., 1] >= 0) & (pos[..., 1] < h))[:, None])
        sigma = params[:, :2].clamp(math.log(self.sigma_min), math.log(2.5)).exp()
        return self.gaussian_weights(xy, centre, params[:, 2:3], sigma, torch.cat(valid, 1), self.taps), centre

    @staticmethod
    def gaussian_weights(xy, centre, angle, sigma, valid=None, taps=5):
        """Stable normalized Gaussian; accepts tiny sigmas for delta-limit tests."""
        dx, dy = ((centre - xy)[..., i][:, None] for i in range(2))
        cs, sn = angle.cos(), angle.sin()
        a, b = sigma[:, :1].clamp(min=1e-4), sigma[:, 1:2].clamp(min=1e-4)
        logits = []
        for oy in range(-(taps // 2), taps // 2 + 1):
            for ox in range(-(taps // 2), taps // 2 + 1):
                u, v = cs * (dx + ox) + sn * (dy + oy), -sn * (dx + ox) + cs * (dy + oy)
                logits.append(-.5 * ((u / a).square() + (v / b).square()))
        logits = torch.cat(logits, 1)
        if valid is not None:
            logits = logits.masked_fill(~valid, -torch.inf)
        weights = (logits - logits.amax(1, keepdim=True)).exp()
        return weights / weights.sum(1, keepdim=True).clamp(min=1e-8)

    def resolve(self, lr, history, alpha, weights, centre, slack, residual, reset):
        h, w = self.render_size
        current = torch.zeros_like(history)
        ix, iy = centre[0, ..., 0].long(), centre[0, ..., 1].long()
        radius = self.taps // 2
        for i, (oy, ox) in enumerate((y, x) for y in range(-radius, radius + 1)
                                    for x in range(-radius, radius + 1)):
            samples = lr[0, :, (iy + oy).clamp(0, h - 1), (ix + ox).clamp(0, w - 1)][None]
            current = current + weights[:, i:i + 1] * samples
        lo = -F.max_pool2d(-lr, 3, 1, 1)
        hi = F.max_pool2d(lr, 3, 1, 1)
        # Clamp bounds use the same nearest-sample centre as the reconstruction.
        lo = lo[0, :, iy.clamp(0, h - 1), ix.clamp(0, w - 1)][None]
        hi = hi[0, :, iy.clamp(0, h - 1), ix.clamp(0, w - 1)][None]
        span = hi - lo
        rectified = torch.minimum(torch.maximum(history, lo - slack * span), hi + slack * span)
        alpha = torch.where(reset, 1., alpha)
        return (alpha * current + (1 - alpha) * rectified + residual).clamp(0, 1 - 1e-6)

    def forward(self, state, lr_rgb, mv, depth, jitter, prev_depth_hr=None, hist=None):
        h, w = self.render_size
        lr = lr_rgb.permute(2, 0, 1)[None].float()
        motion = mv.permute(2, 0, 1)[None].float()
        ml = motion if motion.shape[-2:] == (h, w) else F.avg_pool2d(motion, 2)
        d = depth[None, None].float()
        if d.shape[-2:] != (h, w):
            d = d[..., ::2, ::2]
        if self.mv_dilate:
            _, ml = _nearest_depth(d, ml.permute(0, 2, 3, 1))
            ml = ml.permute(0, 3, 1, 2)
        mh = (motion if not self.mv_dilate and motion.shape[-2:] == self.output_size else
              F.interpolate(ml, size=self.output_size, mode="bilinear", align_corners=False))
        uv = self._uv_lr + ml.permute(0, 2, 3, 1)
        uvh = self._uv_hr + mh.permute(0, 2, 3, 1)
        old_depth = self._warp(state.depth, uv)
        dmin, dmax = -F.max_pool2d(-d, 3, 1, 1), F.max_pool2d(d, 3, 1, 1)
        mismatch = (old_depth < .9 * dmin) | (old_depth > 1.1 * dmax)
        offscreen = ~((uv >= 0) & (uv < 1)).all(-1)[:, None]
        invalid = mismatch | offscreen
        reset = offscreen if self.depth_soft else invalid
        if state.frame_index == 0:
            reset, invalid = torch.ones_like(reset), torch.ones_like(invalid)
        reset_hr = F.interpolate(reset.float(), scale_factor=2, mode="nearest").bool()
        offscreen_hr = ~((uvh >= 0) & (uvh < 1)).all(-1)[:, None]
        reset_hr = reset_hr | offscreen_hr
        invalid = invalid | F.max_pool2d(offscreen_hr.float(), 2).bool()
        history = self.reproject(state, mh.permute(0, 2, 3, 1)) if hist is None else hist.float()
        history = torch.where(reset_hr, 0., history.clamp(0, 1 - 1e-6))
        age = torch.where(invalid, 0., self._warp(state.age, uv)).clamp(0, 32)
        yc = rgb_to_ycocg(lr.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        yh = rgb_to_ycocg(history.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        luma = yc[:, :1]
        mean_history = F.avg_pool2d(yh, 2)
        luma_range = F.max_pool2d(luma, 3, 1, 1) + F.max_pool2d(-luma, 3, 1, 1)
        disagreement = ((mean_history[:, :1] - luma) / (luma_range + .02)).clamp(-4, 4)
        speed = torch.linalg.vector_norm(ml * ml.new_tensor([w, h])[None, :, None, None], dim=1, keepdim=True)
        signed = torch.as_tensor(jitter, device=lr.device, dtype=torch.float32) * self.jitter_sign
        packed = torch.cat((yc, mean_history, F.pixel_unshuffle(yh[:, :1], 2), disagreement,
                            invalid.float(), (speed / 10).clamp(0, 1), torch.log2(1 + age) / 5,
                            signed.reshape(1, 2, 1, 1).expand(1, 2, h, w)), 1)
        hp, wp = self.padded_size
        packed = F.pad(packed, (0, wp - w, 0, hp - h), mode="replicate")
        params = self.trunk(packed)[..., :2 * h, :2 * w].float()
        weights, centre = self._kernel(params, signed)
        alpha = params[:, 3:4].sigmoid()
        if self.proximity:
            xy = self._sample_xy + signed.reshape(1, 1, 1, 2)
            r2 = ((centre - xy) * 2).square().sum(-1)[:, None]   # squared distance, output pixels
            # Log-odds form scales the current frame's weight with sample proximity without
            # dividing underflowed weights, so history can still be discarded at frame edges.
            alpha = (params[:, 3:4] + math.log(self.proximity_gain)
                     - r2 / (2 * self.proximity ** 2)).sigmoid()
        alpha = torch.where(reset_hr, 1., alpha)
        # K=7 uses two residual controls; tanh bounds each RGB correction by .05.
        if self.residual:
            y, co = params[:, 5:7].chunk(2, 1)
            residual = .05 * torch.cat((y + co, y, y - co), 1).tanh()
        else:
            residual = 0.
        out = self.resolve(lr, history, alpha, weights, centre, params[:, 4:5].sigmoid(),
                           residual, reset_hr)
        self._last_reset, self._last_alpha = reset_hr.float(), alpha
        new_age = torch.where(invalid, 0., (age + 1).clamp(max=32))
        new = KPNState(out, new_age, d.clone(), state.frame_index + 1)
        return out[0].permute(1, 2, 0), new

    def cost(self):
        """Exclude pooling/resampling, bias, activation, Gaussian exp/normalization,
        rectification and blend math; filter_macs counts the taps-by-taps RGB sum only.
        """
        trunk = self.trunk.macs(self.padded_size)
        filtered = math.prod(self.output_size) * self.taps ** 2 * 3
        return dict(parameters=sum(p.numel() for p in self.parameters()), trunk_macs=trunk,
                    filter_macs=filtered, trunk_and_filter_macs=trunk + filtered)
