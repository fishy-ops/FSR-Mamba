"""Kernel-prediction output head: predict per-pixel *filter weights*, not colour.

Why this exists (the ideology change)
-------------------------------------
Every model in this project so far composes its output as

    out = lerp(warped_history, resolve, alpha) + residual

i.e. it **predicts colour**. That single choice is the source of the project's
longest-running problem. Anything added to the resolve is fed back into the
recurrent history and compounds frame over frame, so an unbounded residual
destroys the accumulation (v15, v16 and v17 all collapsed exactly this way). The
fix used since v18 -- `tanh(raw) * box_stddev * gain` -- is a *hand-tuned bound on
a fundamentally unbounded operator*, and it caps how much the network is allowed
to change, which is precisely the thing that shows up as "detail is absent, not
damped".

What the current production upscalers and the denoising literature do instead is
predict **filtering weights** and apply them to real samples:

- Intel's HPG 2022 joint denoiser/supersampler shares one feature extractor across
  "multiple higher-precision **filter stages**".
- AMD's I3D 2025 neural supersampler "predict[s] multiple filtering weights".
- The Monte-Carlo denoising line (KPCN and successors) is built on predicting a
  spatially varying per-pixel kernel rather than a colour.

The property that matters is that the output is a **convex combination of colours
that actually exist in the inputs**. With weights from a softmax, the result lies
inside the convex hull of its candidate samples, so:

- it cannot drift or ring, no matter how many frames it is recurred through;
- no tanh clamp, no `box_stddev` scaling, no `gain` parameter is needed;
- the network is free to change the output *completely* (pick a different sample)
  without ever being able to invent an out-of-range colour. Expressiveness and
  stability stop being in tension, which is the trade every previous run fought.

It is also a strict generalisation of FSR, which is the design rule that made this
project work in the first place (see the architecture lesson in HANDOFF.md). FSR's
resolve is itself a weighted sum: a sliced Lanczos over the low-res taps at
~0.046 weight per frame against history at up to 1.0. If this head puts 0.954 on
the history candidate and spreads 0.046 over the tap candidates in Lanczos ratios,
it *is* FSR 3.1.4. The biases below initialise it to almost exactly that point, so
training starts from a working accumulator rather than from noise.

Candidates per output pixel
---------------------------
    9  raw low-res taps -- the 3x3 LR neighbourhood around the source pixel,
       nearest-expanded to output resolution. These are the actual rendered
       samples, not a resampling of them, so the head can reconstruct detail the
       fixed 3x3 Lanczos discarded instead of only polishing what survived.
    1  warped (and optionally rectified) history colour.
    1  the Lanczos resolve itself, so the incumbent behaviour stays reachable.

All in YCoCg, matching the rest of the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sepconv import conv2d as _conv

__all__ = ["KernelPredictHead", "lr_taps_at_output_res"]

N_TAPS = 9  # 3x3 low-res neighbourhood


def lr_taps_at_output_res(lr_ycocg: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """(1,3,rh,rw) -> (1, N_TAPS, 3, uh, uw): 9 phase-correct bilinear LR samples.

    Each output pixel is mapped to its true continuous position in low-res space and
    the 9 taps are sampled at that position plus (-1,0,+1) offsets, bilinearly.

    The obvious cheaper construction -- unfold the 3x3 neighbourhood at low res and
    nearest-expand it -- is what this originally did, and it is **broken for this
    purpose**: it gives every output pixel in a 2x2 quad the *same* nine candidate
    colours, so a weighted combination of them is constant across the quad and can
    express no sub-pixel structure at all. It can only add blockiness. Measured on the
    v42 checkpoint, the head duly learned to ignore those candidates, pushing their
    total weight from 0.008 at init down to 0.0047 while leaving history at 0.94 --
    i.e. the whole new capability sat unused and the model stayed FSR.

    Sampling at the output pixel's own sub-pixel phase is the fix: neighbouring output
    pixels get genuinely different tap values, so the predicted weights can reconstruct
    detail between low-res samples instead of merely re-mixing them.
    """
    b, _, rh, rw = lr_ycocg.shape
    uh, uw = out_hw
    dev, dt = lr_ycocg.device, lr_ycocg.dtype

    # Output pixel centres expressed in low-res pixel coordinates.
    ys = (torch.arange(uh, device=dev, dtype=dt) + 0.5) * (rh / uh) - 0.5
    xs = (torch.arange(uw, device=dev, dtype=dt) + 0.5) * (rw / uw) - 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    taps = []
    for dy in (-1.0, 0.0, 1.0):
        for dx in (-1.0, 0.0, 1.0):
            # grid_sample wants normalised [-1,1] coords with align_corners=False.
            nx = ((gx + dx + 0.5) / rw) * 2.0 - 1.0
            ny = ((gy + dy + 0.5) / rh) * 2.0 - 1.0
            grid = torch.stack((nx, ny), dim=-1).unsqueeze(0).expand(b, uh, uw, 2)
            taps.append(F.grid_sample(lr_ycocg, grid, mode="bilinear",
                                      padding_mode="border", align_corners=False))
    return torch.stack(taps, dim=1)  # (b, N_TAPS, 3, uh, uw)


def phase_planes(out_hw: tuple[int, int], render_hw: tuple[int, int],
                 jitter, device, dtype) -> torch.Tensor:
    """(1, 4, uh, uw): sub-pixel phase within the low-res pixel, plus the jitter offset.

    A convolution is translation-invariant, so it **cannot tell which sub-pixel of a
    2x2 quad it is looking at** -- it sees the same neighbourhood statistics at every
    phase. Without this the head can only apply one kernel per quad, which is the same
    limitation that made the nearest-expanded taps useless. Handing it the phase makes
    the predicted kernel phase-dependent, which is what turns accumulated jittered
    samples into sub-pixel detail.

    Jitter is included for the same reason it is fed to the LR upsampler: it moves the
    sample grid, so the correct kernel for a given phase depends on it.
    """
    uh, uw = out_hw
    rh, rw = render_hw
    ys = (torch.arange(uh, device=device, dtype=dtype) + 0.5) * (rh / uh) - 0.5
    xs = (torch.arange(uw, device=device, dtype=dtype) + 0.5) * (rw / uw) - 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    px = (gx - torch.floor(gx)) * 2.0 - 1.0
    py = (gy - torch.floor(gy)) * 2.0 - 1.0
    if not torch.is_tensor(jitter):
        jitter = torch.tensor(jitter, device=device, dtype=dtype)
    jx = jitter[0].to(dtype).expand(uh, uw)
    jy = jitter[1].to(dtype).expand(uh, uw)
    return torch.stack((px, py, jx, jy), dim=0).unsqueeze(0)


class KernelPredictHead(nn.Module):
    """Predicts softmax weights over (LR taps + history + resolve)."""

    def __init__(self, in_ch: int, hidden: int = 32, separable: bool = False):
        super().__init__()
        self.n_cand = N_TAPS + 2  # taps, history, resolve
        # +4: sub-pixel phase (x,y) and jitter (x,y); see phase_planes.
        self.net = nn.Sequential(
            _conv(in_ch + 4, hidden, 3, separable=separable),
            nn.GELU(),
            _conv(hidden, self.n_cand, 3, separable=separable),
        )
        nn.init.zeros_(self.net[2].weight)
        # Initialise to FSR's converged blend: softmax([-4]*9 + [3.0, 0.0]) gives
        # ~0.945 on history, ~0.047 on the resolve and ~0.008 spread over the raw
        # taps -- within a hair of FSR's 0.954 / 0.046. The model therefore starts
        # as a working temporal accumulator, which is the lesson that rescued the
        # very first learned version of this project.
        bias = torch.full((self.n_cand,), -4.0)
        bias[N_TAPS] = 3.0      # warped history
        bias[N_TAPS + 1] = 0.0  # Lanczos resolve
        with torch.no_grad():
            self.net[2].bias.copy_(bias)

    def forward(self, feats: torch.Tensor, taps: torch.Tensor, hist: torch.Tensor,
                resolve: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """feats (1,C,H,W); taps (1,9,3,H,W); hist/resolve (1,3,H,W); phase (1,4,H,W)."""
        cands = torch.cat((taps, hist.unsqueeze(1), resolve.unsqueeze(1)), dim=1)
        w = torch.softmax(self.net(torch.cat((feats, phase), dim=1)), dim=1).unsqueeze(2)
        self._last_weights = w
        return (cands * w).sum(dim=1)
