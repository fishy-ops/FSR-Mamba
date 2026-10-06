"""Exposure gains, clip consistency, and disabled augmentation."""

from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.augment import augment_exposure, exposure_gain, exposure_range


def main():
    torch.manual_seed(31)
    image = torch.rand(12, 18, 3) * 0.97
    snapshot = image.clone()
    torch.testing.assert_close(exposure_gain(image, 1), image, atol=1e-6, rtol=0)
    for gain in (0.05, 0.3, 3, 20):
        restored = exposure_gain(exposure_gain(image, gain), 1 / gain)
        torch.testing.assert_close(restored, image, atol=1e-4, rtol=0)
    clipped = exposure_gain(torch.ones(2, 3, 3), 1)
    assert torch.isfinite(clipped).all() and (clipped == 1 - 1 / 1024).all()

    geometry = torch.rand(12, 18, 2)
    frames = [dict(lr=image, gt=image, fsr_out=image, mv=geometry,
                   depth=geometry[..., 0], jitter=(0.1, -0.2)) for _ in range(4)]
    rng = random.Random(19)
    state = rng.getstate()
    assert augment_exposure(frames, None, rng) is frames
    assert rng.getstate() == state
    bounds = exposure_range("0.05,20")
    augmented = augment_exposure(frames, bounds, rng)
    gain = (augmented[0]["lr"] / (1 - augmented[0]["lr"])) / (image / (1 - image))
    assert gain.min() >= bounds[0] and gain.max() <= bounds[1]
    torch.testing.assert_close(gain, torch.full_like(gain, gain.mean()), atol=1e-5, rtol=1e-5)
    for original, out in zip(frames, augmented):
        for key in ("lr", "gt", "fsr_out"):
            assert torch.equal(out[key], augmented[0]["lr"])
            assert out[key] is not original[key]
        for key in ("mv", "depth", "jitter"):
            assert out[key] is original[key]
        assert original["lr"] is image and torch.equal(original["lr"], snapshot)
    for value in ("0,1", "-1,2", "2,1", "nan,2", "1,inf", "1", "1,2,3", "x,2"):
        try:
            exposure_range(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid bounds accepted: {value}")
    print("Exposure identity, inverse, clip consistency, and default-off tests passed")


if __name__ == "__main__":
    main()
