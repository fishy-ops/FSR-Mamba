"""Per-phase Lanczos kernels and deringing windows from the FSR resolve."""

import torch
import torch.nn.functional as F

from .baseline import _lanczos2

_BOX_CURVE = -2.3


def phase_kernels(jitter, device, ph_x, ph_y, tap_x, tap_y, sharp_bias=None):
    j = jitter if torch.is_tensor(jitter) else torch.tensor(jitter, device=device)
    j = j.float()
    ph_x, ph_y = ph_x.float(), ph_y.float()
    tap_x, tap_y = tap_x.float(), tap_y.float()
    ox = tap_x + 0.5 - j[0] - ph_x
    oy = tap_y + 0.5 - j[1] - ph_y
    # The 3x3 window starts at -2 when the base sample lies beyond the output point.
    sx = torch.where(0.5 - j[0] - ph_x > 0, -2.0, -1.0)
    sy = torch.where(0.5 - j[1] - ph_y > 0, -2.0, -1.0)
    inwin = ((tap_x >= sx) & (tap_x <= sx + 2)
             & (tap_y >= sy) & (tap_y <= sy + 2)).float()
    r2 = ox * ox + oy * oy
    r = torch.sqrt(r2)
    k0 = _lanczos2(r) * inwin
    k1 = _lanczos2(r * sharp_bias) * inwin if sharp_bias is not None else None
    g = torch.exp(_BOX_CURVE * r2) * inwin if sharp_bias is not None else None
    s0 = k0.sum(dim=(1, 2), keepdim=True)
    s1 = k1.sum(dim=(1, 2), keepdim=True) if k1 is not None else None
    # The sharp kernel can lose all its weight when no tap is near; fall back to k0.
    if k1 is not None:
        k1 = torch.where(s1 > 1e-3, k1 / s1.clamp(min=1e-3), k0 / s0.clamp(min=1e-4))
    k0 = k0 / s0.clamp(min=1e-4)
    if g is not None:
        g = g / g.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    prior = torch.log(s0.clamp(min=1e-4) / s0.mean()).reshape(1, ph_x.shape[0], 1, 1)
    win = [(int(sy[i].item()) + 2, int(sx[i].item()) + 2) for i in range(ph_x.shape[0])]
    return (j, k0.unsqueeze(1), k1.unsqueeze(1) if k1 is not None else None,
            g.unsqueeze(1) if g is not None else None, prior, win)


def window_bounds(taps, win, h, w):
    mxp = F.max_pool2d(taps, 3, stride=1)
    mnp = -F.max_pool2d(-taps, 3, stride=1)
    mx = torch.cat([mxp[:, :, a: a + h, b: b + w] for a, b in win], dim=1).unsqueeze(0)
    mn = torch.cat([mnp[:, :, a: a + h, b: b + w] for a, b in win], dim=1).unsqueeze(0)
    return mn, mx
