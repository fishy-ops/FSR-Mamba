"""U-Net trunk: put the parameters where pixels are cheap.

Why this exists
---------------
This project's model was *flat* -- every convolution ran at render or output resolution,
with no downsampling anywhere. Measured, that is 592 GFLOP per 1080p frame (285,306 FLOP per
output pixel) against FSR 3.1.4's few hundred, and ~35 ms even with a perfect fp16 port.
The instinct was to fix it by shrinking channel widths, which trades quality directly for
speed and produced v54: 7k parameters, 1.7 ms, and 1.5 dB worse than FSR.

That was optimising the wrong dimension. A convolution costs ``channels^2 x pixels``, so
what matters is not how many parameters you have but *what resolution you run them at*:

    48->48 3x3 at 1080p   =  86 GFLOP
    the same conv at 1/4  =   5.4 GFLOP
    the same conv at 1/8  =   1.3 GFLOP      <- 64x cheaper, identical parameters

A U-Net exploits this exactly. Each level halves the spatial dimensions and doubles the
channels, so ``channels^2`` grows 4x while ``pixels`` falls 4x and **every level costs the
same**. Capacity grows exponentially with depth while cost grows linearly. Measured for this
design: 3.1M parameters -- 8.4x the flat model -- for 167 GFLOP, which is 3.5x *less*.

This is why DLSS 4.5 and FSR 4.1 fit million-parameter networks in a frame budget. They are
not doing something exotic; they are U-Nets whose weight sits at 1/4 and 1/8 resolution,
running on matrix hardware this GPU does not have, at fp8/int8. Architecture, hardware and
precision each contribute roughly 64x, 12x and 2x.

Design notes
------------
- **The pyramid starts at RENDER resolution, not output.** The expensive thing about the old
  design was convolutions at 1080p; here the only full-resolution work is a PixelShuffle and
  elementwise blending, both trivial. This mirrors FSR itself, which does nearly all of its
  work at render resolution.
- **Skip connections** carry the high-frequency detail that survives downsampling; without
  them a U-Net bottleneck destroys exactly the thin geometry this project cares about.
- **Bilinear upsampling + conv rather than transposed conv**, which avoids the checkerboard
  artifacts transposed convolutions produce -- relevant here because those artifacts would be
  written into the recurrent history and compound.
- **Separable convs optional**, composing with the ~7.9x factorisation win in sepconv.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sepconv import conv2d

__all__ = ["UNetTrunk"]


class _Block(nn.Module):
    """Two convs at one pyramid level, group-normalised.

    GroupNorm (not BatchNorm) because batch size here is 1 -- batch statistics over a
    single crop are meaningless. Without normalisation a 4-level pyramid compounds its
    activation scale level over level; measured on the first attempt, the trunk emitted
    features at std 20.5 where the flat encoder produced ~1, which saturated the
    downstream softmax within a few epochs and pinned the model (see the class docstring).
    """

    def __init__(self, in_ch: int, out_ch: int, separable: bool = False):
        super().__init__()
        g = max(1, min(8, out_ch // 8))
        self.body = nn.Sequential(
            conv2d(in_ch, out_ch, 3, separable=separable), nn.GroupNorm(g, out_ch), nn.GELU(),
            conv2d(out_ch, out_ch, 3, separable=separable), nn.GroupNorm(g, out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class UNetTrunk(nn.Module):
    """(B, in_ch, H, W) -> (B, out_ch, H, W), with most work done far below H x W.

    ``levels`` counts downsampling steps. With base=32 and levels=3 the pyramid runs
    32 / 64 / 128 / 256 channels at 1 / 1/2 / 1/4 / 1/8 resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, base: int = 32, levels: int = 3,
                 separable: bool = False):
        super().__init__()
        self.levels = levels
        chans = [base * (2 ** i) for i in range(levels + 1)]

        self.enc = nn.ModuleList()
        c_prev = in_ch
        for c in chans:
            self.enc.append(_Block(c_prev, c, separable))
            c_prev = c

        self.dec = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            # input = upsampled deeper features + the skip from this level
            self.dec.append(_Block(chans[i + 1] + chans[i], chans[i], separable))

        # Final GELU so the trunk's output distribution matches what the flat encoder
        # produced (it ended in GELU). Without it the raw conv output was unbounded and
        # ~20x too large, which saturated the kernel-prediction softmax: measured weights
        # collapsed to 0.998 on the raw taps and 0.001 on history, i.e. the model threw
        # away temporal accumulation entirely and became a per-frame resampler.
        self.out = nn.Sequential(conv2d(chans[0], out_ch, 3, separable=separable), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.levels:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        for j, blk in enumerate(self.dec):
            skip = skips[self.levels - 1 - j]
            # Bilinear + conv, not transposed conv: transposed convolutions produce
            # checkerboard artifacts, and anything periodic here is written into the
            # recurrent history and compounds frame over frame.
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat((x, skip), dim=1))
        return self.out(x)
