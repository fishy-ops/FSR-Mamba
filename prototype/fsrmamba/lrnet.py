"""Learned upsampler that reads the RAW low-res image (not the Lanczos output).

Why this exists
---------------
Every refiner tried so far (dilated-conv v12, Swin v15) is a residual on the
*already-resolved* Lanczos image, fed only Lanczos-derived signals (the upsample
plus its neighbourhood box). None of them can see the raw LR samples, so none can
recover detail the fixed 3x3 Lanczos taps discarded -- they can only polish what
survived. That is a strong candidate for why every variant pinned to ~0.9425 SSIM,
exactly the score of the hand-written FSR reimplementation: they all inherit the
same spatial front-end.

This module does what an actual super-resolution network does: convolve at *low*
resolution (cheap -- 4x fewer pixels than HR), then expand with PixelShuffle,
producing a residual on the resolve. It therefore has access to the full LR signal.

Jitter is fed in as two constant planes. The LR frame is sub-pixel jittered and the
Lanczos resolve accounts for that when placing samples; an upsampler that did not
know the offset would produce a residual misregistered by a fraction of a pixel, so
it is told explicitly.

The final conv is zero-initialised: the module starts as an exact no-op, so it can
only move the output away from plain Lanczos if that reduces the loss -- important
because this feeds a recurrent accumulation that can otherwise diverge.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .sepconv import conv2d

__all__ = ["LRUpsampler"]


class _ResBlock(nn.Module):
    def __init__(self, dim: int, separable: bool = False):
        super().__init__()
        # These four blocks are 53% of the whole model's per-frame cost (measured:
        # 316 of 592 GFLOP at 1080p). Separable convs cut them ~7.9x, which is the
        # single largest efficiency lever available anywhere in the network.
        self.body = nn.Sequential(
            conv2d(dim, dim, 3, separable=separable), nn.GELU(),
            conv2d(dim, dim, 3, separable=separable)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class LRUpsampler(nn.Module):
    """LR-domain conv trunk + PixelShuffle -> HR residual on the resolve.

    ``film`` conditions the trunk output on the jitter offset multiplicatively
    instead of relying only on the two constant input planes. The planes can only
    shift features; what jitter actually controls is *where the LR sample sits
    inside the output pixel*, i.e. the resampling geometry, and that enters the
    correct mapping multiplicatively. A jitter-conditioned modulation lets the same
    LR neighbourhood produce a different sub-pixel reconstruction per phase, which
    is the mechanism by which accumulated jittered samples turn into detail --
    exactly the thing the model is measurably short of ("detail is absent, not
    damped": every post-hoc sharpening attempt lowered both PSNR and SSIM).

    It is deliberately tiny (~4k parameters on a ~350k model). Every capacity
    increase tried so far has lost, so this is a change of *mechanism*, not size,
    and it is initialised to the exact identity (gamma=1, beta=0) so the module
    still starts as a no-op residual.
    """

    def __init__(self, scale: int, dim: int = 64, depth: int = 4, film: bool = False,
                 separable: bool = False):
        super().__init__()
        self.scale = scale
        # 3 LR YCoCg + 2 jitter planes
        self.head = conv2d(5, dim, 3, separable=separable)
        self.trunk = nn.Sequential(*[_ResBlock(dim, separable) for _ in range(depth)])
        self.tail = conv2d(dim, 3 * scale * scale, 3, separable=separable)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

        self.film = None
        if film:
            # 2 -> 32 -> (gamma, beta) per feature channel.
            self.film = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 2 * dim))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)  # gamma = 1 + 0, beta = 0 at init

    def forward(self, lr_ycocg: torch.Tensor, jitter: torch.Tensor) -> torch.Tensor:
        """lr_ycocg: (1,3,rh,rw), jitter: (2,) -> (1,3,rh*scale,rw*scale)"""
        b, _, h, w = lr_ycocg.shape
        j = jitter.to(lr_ycocg.dtype).view(1, 2, 1, 1).expand(b, 2, h, w)
        x = self.head(torch.cat((lr_ycocg, j), dim=1))
        x = self.trunk(x)
        if self.film is not None:
            g, s = self.film(jitter.to(x.dtype).view(1, 2)).chunk(2, dim=1)
            x = x * (1.0 + g.view(1, -1, 1, 1)) + s.view(1, -1, 1, 1)
        return self.shuffle(self.tail(x))
