"""Depthwise-separable convolution -- the standard fix when you are FLOP-bound.

A dense KxK conv with C_in -> C_out costs K*K*C_in*C_out MACs per pixel. Factoring it into
a KxK *depthwise* conv (one filter per input channel, no cross-channel mixing) followed by a
1x1 *pointwise* conv (all mixing, no spatial extent) costs K*K*C_in + C_in*C_out. For the
64->64 3x3 convs that dominate this model's cost:

    dense      : 9 * 64 * 64 = 36,864 MAC/px
    separable  : 9 * 64 + 64 * 64 = 4,672 MAC/px      -> 7.9x cheaper

Capacity does not fall anywhere near 7.9x, because spatial filtering and channel mixing are
largely independent jobs and doing them separately loses little -- this is the observation
MobileNet is built on and it is why every mobile/real-time vision network uses it.

Why it matters here specifically: this project measured 592 GFLOP per 1080p frame against
FSR 3.1.4's few hundred FLOP/pixel, i.e. ~476x more arithmetic for comparable quality. The
FLOP breakdown put 53% of that in `lr_up`'s four residual blocks, which are exactly the
64->64 dense 3x3 convs above. Nothing else in the model is as cheap to fix.

The depthwise conv is initialised near-identity (centre tap 1, ring 0) so a separable model
starts out behaving like a pointwise-only network rather than like noise, which matters when
distilling into it from a dense teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SepConv2d", "conv2d"]


class SepConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d(in, out, k, padding=k//2)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int | None = None,
                 identity_init: bool = True):
        super().__init__()
        pad = k // 2 if padding is None else padding
        self.dw = nn.Conv2d(in_ch, in_ch, k, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        if identity_init and k > 1:
            with torch.no_grad():
                self.dw.weight.zero_()
                self.dw.weight[:, 0, k // 2, k // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

    @property
    def weight(self):
        """So callers that zero-init `.weight` (the model does this in several places
        to make a module start as a no-op) still hit something meaningful -- the
        pointwise stage, which gates the whole block's output."""
        return self.pw.weight

    @property
    def bias(self):
        return self.pw.bias


def conv2d(in_ch: int, out_ch: int, k: int = 3, padding: int | None = None,
           separable: bool = False) -> nn.Module:
    """Dense or separable conv, chosen by flag, so call sites stay identical."""
    pad = k // 2 if padding is None else padding
    if separable and k > 1:
        return SepConv2d(in_ch, out_ch, k, pad)
    return nn.Conv2d(in_ch, out_ch, k, padding=pad)
