"""Excess temporal-change loss, motion alignment, masks, resets and gradients."""
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import flicker_loss, warp_prev


def main():
    h, w = 8, 12
    zero = torch.zeros(h, w, 3)
    mv = torch.zeros(h, w, 2)
    reset = torch.zeros(1, 1, h//2, w//2)
    out = torch.full_like(zero, .4, requires_grad=True)
    loss = flicker_loss(out, zero, torch.full_like(zero, .1), zero, mv, reset)
    assert torch.allclose(loss, torch.tensor(.3))
    loss.backward()
    assert out.grad.gt(0).all()
    assert flicker_loss(zero, out.detach(), zero, torch.full_like(zero, .5), mv, reset) == 0
    assert flicker_loss(out, zero, zero, zero, mv, torch.ones_like(reset)) == 0
    assert flicker_loss(out, zero, zero, zero, mv, reset, torch.zeros(h, w)) == 0
    assert flicker_loss(out, zero, zero, zero, mv, reset, previous_mask=torch.zeros(h, w)) == 0
    mask = torch.zeros(h, w)
    mask[2:6, 3:9] = 1
    assert torch.allclose(flicker_loss(out, zero, zero, zero, mv, reset, mask, mask, 1), torch.tensor(.4))
    previous = torch.rand(h, w, 3)
    mv[..., 0] = 1/w
    aligned = warp_prev(previous, mv)
    assert flicker_loss(aligned, previous, aligned, previous, mv, reset) == 0
    assert flicker_loss(aligned, previous, zero, zero, mv, reset) == 0
    # The previous invalid texel must invalidate its new motion-warped location.
    mask.fill_(1)
    mask[:, 5] = 0
    changed = aligned.clone()
    changed[:, 4] += .2
    assert flicker_loss(changed, previous, aligned, previous, mv, reset, previous_mask=mask) == 0
    mv[..., 0] = -1
    assert flicker_loss(out, zero, zero, zero, mv, reset) == 0
    help_text = subprocess.run([sys.executable, "train.py", "--help"], check=True,
                               capture_output=True, text=True).stdout
    assert "--flicker-weight" in help_text
    for value in ("-1", "nan", "inf"):
        result = subprocess.run([sys.executable, "train.py", "--flicker-weight", value],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "--flicker-weight must be" in result.stderr
    print("test_flicker_loss passed")


if __name__ == "__main__":
    main()
