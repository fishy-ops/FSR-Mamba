"""Temporal PatchGAN discriminator -- a learned perceptual loss for texture.

Why a GAN here
--------------
L1 and PSNR ask for the *conditional mean* of all outputs consistent with the input, and
the mean of many plausible textures is a blur. That is not a tuning failure, it is what the
objective mathematically requests, and it is why this project measures 0.798 of
ground-truth texture energy where FSR reaches 1.061 and where every fixed loss tried so far
(spectral, VGG-perceptual, gradient, edge-weighted gradient) failed to move it. A
discriminator supplies a *learned* notion of "looks real" that no hand-written loss term can
express.

It is also free where this project is most constrained: the discriminator is discarded after
training, so it adds **zero inference cost**. That makes it strictly preferable to adding
generator capacity, which we cannot afford -- the real-time budget is ~1.7 ms and the
current 372k model already needs ~35 ms.

Why TEMPORAL, and not a plain PatchGAN
--------------------------------------
A per-frame discriminator rewards any texture that looks locally real, whether or not it is
consistent with the previous frame. Fed into a recurrent accumulator, invented detail is
written into history and compounds -- it reads as shimmer and crawling, which is precisely
the failure mode this project exists to avoid, and it would destroy the temporal metric that
is currently our only outright win over FSR.

So the discriminator sees a *pair*: the current frame and the motion-warped previous frame,
concatenated. To be judged real, a texture must be plausible AND stable under the true
motion. Hallucinating something that flickers is then penalised rather than rewarded. This
is the TecoGAN construction, and the temporal pairing is the part that makes it usable for
video rather than stills.

Design notes
------------
- **Spectral normalisation** on every conv. GAN training on 2,040 frames is easily unstable,
  and constraining the discriminator's Lipschitz constant is the cheapest reliable fix.
- **PatchGAN** (fully convolutional, no global pooling): it judges local texture statistics
  over ~70x70 receptive fields rather than whole-image semantics, which is what we want --
  the complaint is about texture and edges, not about content being wrong.
- **Hinge loss**: better behaved than BCE when the discriminator gets strong early.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["TemporalPatchDiscriminator", "d_hinge_loss", "g_adv_loss"]


class TemporalPatchDiscriminator(nn.Module):
    """(current, warped-previous) -> per-patch real/fake logits."""

    def __init__(self, in_ch: int = 6, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers = [spectral_norm(nn.Conv2d(in_ch, base, 4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        c = base
        for i in range(n_layers):
            nc = min(c * 2, 256)
            layers += [spectral_norm(nn.Conv2d(c, nc, 4, stride=2, padding=1)),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = nc
        layers += [spectral_norm(nn.Conv2d(c, 1, 4, stride=1, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, cur: torch.Tensor, prev_warped: torch.Tensor) -> torch.Tensor:
        """cur/prev_warped: (1,3,H,W) -> (1,1,h,w) patch logits."""
        return self.net(torch.cat((cur, prev_warped), dim=1))


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss. Wants d_real > +1 and d_fake < -1."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_adv_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator's non-saturating adversarial term: push fake logits up."""
    return -d_fake.mean()
