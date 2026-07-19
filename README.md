# FSR-Mamba

Replacing the hand-tuned temporal accumulator in AMD's FidelityFX Super
Resolution 3.1.4 with a **learned recurrent (state-space) accumulator**, plus
content-adaptive expert routing.

## The idea in one paragraph

Modern upscalers render at low resolution and reconstruct detail *over time*:
the camera is jittered by a sub-pixel amount each frame, so several blurry frames
collectively contain enough information for one sharp one. The component that
decides how to combine them — "how much do I trust my memory of this pixel
versus what I'm seeing now?" — is FSR's accumulation pass, and it is a few
hundred lines of hand-tuned heuristics with constants earned through years of
artifact-chasing. This project replaces those rules with a learned recurrence:
same per-pixel state, same motion-vector reprojection, but a wider learned state
and a learned update.

## Status

Phase one works. On held-out synthetic scenes, at 2× upscale:

| | PSNR (dB) | SSIM | temporal instability ↓ |
|---|---|---|---|
| FSR baseline (ported) | 19.04 | 0.786 | 0.0281 |
| **learned accumulator** (11k params) | **20.44** | **0.831** | **0.0063** |

Temporal instability — motion-compensated frame-to-frame difference — is the
metric upscalers actually live or die on, and it is 4.5× better.

**Three caveats that belong next to those numbers:**

1. The ported baseline is **weaker than real FSR 3.1.4** — the lock pass, luma
   instability, and reactive masks are not ported. This is a floor, not AMD's
   shipping quality.
2. Training data is **synthetic**, so motion vectors are perfectly truthful.
   Real engine motion vectors lie (shadows, reflections, VFX) and learning when
   they lie is the actual thesis. This is the easy case by construction.
3. **No performance measurement yet.** Parameter count is not milliseconds.

**Expert routing is implemented but shows no measurable benefit yet**, and the
synthetic testbed is too homogeneous to say whether that is a real null result.
See `prototype/README.md` for the router diagnostics.

## Layout

```
ORIENTATION.md        Findings on the FSR source: version map, the graft point,
                      state layout, and the bugs that cost the most time
prototype/            PyTorch harness — synthetic data, baseline port, model
prototype/README.md   Results, ablations, and caveats in detail
fsr-upstream/         AMD FidelityFX SDK @ v1.1.4 (not vendored — see below)
```

## Setup

```bash
git clone --branch v1.1.4 \
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git fsr-upstream

python3 -m venv .venv
# On a CUDA machine, get the right torch build from https://pytorch.org
.venv/bin/pip install numpy torch pillow

cd prototype
../.venv/bin/python run_baseline.py --frames 24 --dump out   # FSR baseline
../.venv/bin/python train.py --epochs 10                     # train the model
../.venv/bin/python diagnose_router.py --num-experts 4       # router probe
```

Nothing here needs Windows or a GPU — the synthetic data pipeline exists
specifically so phase one runs anywhere.

## Attribution

`prototype/fsrmamba/baseline.py` and `colorspace.py` are derived from AMD's
FidelityFX SDK (MIT licence, Copyright © 2024 Advanced Micro Devices, Inc.),
with line references to the original HLSL throughout. FSR 3.1.4 is the newest
version AMD published source for; SDK v2.x ships the upscaler as signed DLLs
only.
