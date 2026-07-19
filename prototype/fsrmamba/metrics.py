"""Quality metrics for upscaler comparison.

Three numbers matter for this project, and they are not interchangeable:

* **PSNR** -- raw fidelity to ground truth. Easy to compute, easy to game, and
  weakly correlated with what the eye notices. Report it, don't trust it alone.
* **SSIM** -- structural similarity. Better aligned with perception than PSNR,
  still a single-frame measure.
* **Temporal instability** -- how much the output flickers between frames once
  motion is compensated for. This is the metric upscalers actually live or die
  on, and it is the one a single-frame model cannot optimise for. Lower is
  better.

The temporal metric is the interesting one here: a learned accumulator that
wins on PSNR but loses on temporal instability has not beaten FSR, it has
overfit to still frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["psnr", "ssim", "temporal_instability"]


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Inputs (H, W, 3) in [0, max_val]."""
    mse = F.mse_loss(pred.clamp(0, max_val), target.clamp(0, max_val)).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over the image, averaged across colour channels."""
    x = pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y = target.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    c = x.shape[1]
    w = _gaussian_window(window, sigma, x.device).expand(c, 1, window, window)

    mu_x = F.conv2d(x, w, padding=window // 2, groups=c)
    mu_y = F.conv2d(y, w, padding=window // 2, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sig_x = F.conv2d(x * x, w, padding=window // 2, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=window // 2, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=window // 2, groups=c) - mu_xy

    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return s.mean().item()


def temporal_instability(
    curr: torch.Tensor, prev: torch.Tensor, mv: torch.Tensor
) -> float:
    """Mean absolute motion-compensated frame difference. Lower is more stable.

    Warps `prev` forward along `mv` (FSR convention: reprojected_uv = uv + mv)
    and measures what is left. On static content this goes to zero; under motion
    it captures shimmer, ghosting, and flicker that PSNR-per-frame misses.

    Pixels whose reprojection lands off-screen are excluded rather than counted
    as instability -- they are disocclusions, which are a different problem.
    """
    h, w, _ = curr.shape
    ys = (torch.arange(h, device=curr.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=curr.device, dtype=torch.float32) + 0.5) / w
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    uv = torch.stack((gx, gy), dim=-1) + mv

    valid = (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
    if not bool(valid.any()):
        return 0.0

    warped = F.grid_sample(
        prev.permute(2, 0, 1).unsqueeze(0),
        (uv * 2 - 1).unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)

    diff = (curr - warped).abs().mean(dim=-1)
    return diff[valid].mean().item()
