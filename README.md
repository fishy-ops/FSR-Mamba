# FSR-Mamba

A **learned temporal accumulator** that replaces the hand-tuned heuristics in
AMD's FidelityFX Super Resolution 3.1.4 with a selective state-space model
(SSM), kernel-prediction output head, and learned low-resolution upsampler.

Trained on 4x SSAA ground truth captured from Unreal Engine 5, evaluated at
full frame (1280x710) on held-out scenes.

## Current best (v64)

Full-frame evaluation on 3 held-out UE5 scenes (bistro, brutalism, chess),
72 frames, 640x355 render → 1280x710 output:

| | PSNR (dB) | SSIM | Temporal instability ↓ |
|---|---|---|---|
| AMD FSR 3.1.4 | 30.87 | 0.9549 | 0.02153 |
| **FSR-Mamba v64** (372k params) | **31.38** (+0.51) | **0.9546** (−0.0003) | 0.02296 (+0.00143) |

PSNR: **+0.51 dB** ahead of FSR. SSIM gap is **0.0003** — visually
indistinguishable. Temporal stability is the one remaining metric to close.

## Architecture

The model keeps FSR's overall pipeline shape — Lanczos resolve, motion-vector
reprojection, history rectification, alpha blending — but replaces the
hand-tuned accumulation rules with learned components.

**Yes, the selective state-space model (SSM) is still the core temporal mechanism
in the final v64 checkpoint.** The SSM was never replaced — later additions
(kernel prediction head, LR upsampler) work *alongside* it, not instead of it.
The SSM's per-pixel state `h` carries temporal memory across frames; its output
`y` feeds into both the alpha blend and the KPN head as their primary feature
signal.

```
LR input (540p) ─────────────────────────────┐
  │                                           │
  ├── FSR Lanczos resolve ── upsampled (1080p)│
  │                                           │
  ├── Flat encoder (2×conv3×3)                │
  │     └── SSM projections (Δ, B, C, X)      │
  │           └── Selective state update ──────┤  ← temporal backbone
  │                 │                          │    (h carries memory
  │                 │                          │     across frames)
  │                 ├── Learned alpha blend    │
  │                 │     └── History rectification (learned box scale)
  │                 │
  │                 └── Kernel prediction head (KPN)
  │                       ├── LR taps at output res
  │                       ├── Rectified history
  │                       └── Current resolve
  │                             ↓
  └── LR Upsampler (4×ResBlock + PixelShuffle)── high-freq residual
                                                    ↓
                                              Output (1080p)
```

### How the SSM and KPN work together

The **SSM** handles the *temporal* question: for each pixel, how much to trust
the accumulated history vs the current frame. It maintains a hidden state `h`
that is warped along motion vectors each frame and updated via input-dependent
gates (the selective scan from Mamba):

```
delta = softplus(to_delta(enc))       # per-pixel forget rate
a_bar = exp(-delta)                   # decay gate
h = a_bar * h_prev + (1-a_bar) * B(enc) * X(enc)   # state update
y = C(enc) * h                        # readout
```

The **KPN** handles the *spatial* question: given the SSM's temporal readout
`y` and the encoder features, predict per-pixel softmax weights over 11 real
candidate colours (9 LR taps + warped history + Lanczos resolve). The output
is a convex combination — bounded by definition, so it cannot diverge through
the recurrence. This solved the stability collapse that killed v15–v17.

### Other key components

- **History rectification**: FSR's anisotropic YCoCg colour box clamp, but with a
  *learned* per-pixel box scale in [1, box_max] instead of FSR's velocity-based
  heuristic. Prevents stale smooth history from overriding sharp current-frame detail.

- **LR Upsampler**: 4-layer ResNet at render resolution with PixelShuffle 2×
  expansion. Recovers high-frequency detail the fixed Lanczos 3×3 taps discard.
  FiLM conditioning on sub-pixel jitter offset for phase-aware reconstruction.
  82% of model parameters, but cheap (runs at 1/4 pixel count).

- **Robust disocclusion**: Neighbourhood depth envelope test (3×3 min/max + tolerance)
  instead of FSR's per-pixel depth test, preventing false disocclusion on geometric
  edges that would kill temporal anti-aliasing.

## Iteration history

67 experiments across 4 phases of development. Key milestones:

### Phase 1: Synthetic data (v1–v10)
Trained on procedurally generated motion + known ground truth. Proved the
SSM-based accumulator concept works. Best synthetic result: PSNR 20.44, SSIM
0.831, temporal instability 0.0063 (4.5× better than the ported FSR baseline).

### Phase 2: Real engine captures (v11–v38)
Switched to 4× SSAA captures from Unreal Engine 5 (10 scenes, 7 train / 3 val).
This changed everything — real motion vectors, real disocclusion patterns, real
texture complexity. Explored: expert routing (no benefit on this data), Swin
transformer refiners, perceptual/frequency losses, learned spatial resolve.

### Phase 3: Architecture search (v39–v56)
Systematic exploration of the quality–latency frontier:
- **LR Upsampler** (v17+): Broke the Lanczos-resolve ceiling at ~0.9425 SSIM
  by giving the model direct access to raw LR samples.
- **Kernel prediction** (v39+): Solved the recurrent divergence problem. Output
  is a convex combination of real colours, so it cannot compound through the
  accumulation.
- **U-Net encoder** (v56): 28.9 GFLOP vs 592 GFLOP (20× cheaper) by running
  convolutions at 1/8 resolution. 1.69 ms measured on GPU — fits the real-time
  budget. Quality gap remains (SSIM −0.0135 vs FSR) — the path forward for
  latency.
- **Separable convolutions** (v58–v59): 7.9× cheaper per conv layer, but
  representation-limited at small channel counts. Non-separable wins.

### Phase 4: Quality push (v61–v67)
Closed the texture sharpness gap to FSR:
- **v61** (32ch non-sep, crop 128): SSIM 0.9502 on crops but 0.9392 at full
  frame — crop-to-full generalization gap.
- **v63** (32ch non-sep, crop 256): Fixed the generalization gap (0.0044 vs
  0.0110) but 32ch encoder is capacity-limited. Full-frame SSIM 0.9394.
- **v64** (48ch non-sep, warm-start from v52 + SSIM/gradient/edge-gradient
  losses): **New best.** Full-frame SSIM 0.9546 (−0.0003 vs FSR), PSNR +0.51.
  Proved the improvement comes from better training signal, not more capacity.
- **v65** (3× SSIM weight): Confirmed the SSIM ceiling is architectural, not
  loss-driven. Identical result to v64.
- **v67** (temporal-focused): Alpha penalty + high temporal weight. Marginal
  crop-temporal improvement, but best-selection criterion saved epoch-0 weights.

### Key findings

1. **Encoder channel count sets the SSIM ceiling.** 32ch caps at ~0.9438 on
   crops regardless of crop size, loss weights, or training length. 48ch reaches
   ~0.9618. This is architectural, not tunable.

2. **Warm-starting is 90% of quality.** v64 started at SSIM 0.9618 on epoch 0
   (from v52's weights). Training from scratch starts at 0.9311 and takes 30+
   epochs to reach ~0.94.

3. **RCAS post-processing hurts.** FSR's Robust Contrast Adaptive Sharpening
   makes both PSNR and SSIM worse at every strength. The texture gap is internal
   to the model's representation, not fixable with post-processing.

4. **The flat encoder is the latency bottleneck.** Two 48×48 3×3 convs at full
   output resolution cost 592 GFLOP/frame at 1080p. The U-Net path (v56) runs
   the same capacity at 1/8 resolution for 28.9 GFLOP — the path to real-time.

## Full-frame scoreboard

All numbers: full-frame 1280×710, 3 held-out UE5 scenes, 72 frames.
Ground truth: 4× SSAA at 2560×1420, area-downsampled.

| Version | Architecture | Params | PSNR gap | SSIM gap | Temporal gap |
|---------|-------------|--------|----------|----------|-------------|
| v52 | 48ch flat, lr_dim 64 | 372k | +0.48 | −0.0005 | +0.00143 |
| **v64** | **48ch flat, lr_dim 64, sharp losses** | **372k** | **+0.51** | **−0.0003** | **+0.00143** |
| v56 | U-Net (3 levels), lr_dim 64 | 924k | −0.74 | −0.0135 | +0.00191 |
| v63 | 32ch flat, crop 256 | 349k | −0.61 | −0.0155 | +0.00353 |
| v61 | 32ch flat, crop 128 | 124k | −0.61 | −0.0157 | +0.00408 |

## Training

### Data

Training uses 4× supersampled captures from Unreal Engine 5. Each frame
includes: LR colour (540p), motion vectors, depth, and the SSAA ground truth
(1080p area-downsampled from 2160p).

10 scenes total:
- **Train** (7): hangar, hybridrefl, locomotive, spaceship, sponza, table, toyshop
- **Val** (3): bistro, brutalism, chess

The capture pipeline (`engine_capture/`) patches UE5's FSR integration to dump
per-frame data during gameplay.

### Training a model

```bash
cd prototype

# From scratch (48ch, 60 epochs, ~35 min on RTX 3070 Ti)
python train.py \
  --engine-data /path/to/captures_ssaa \
  --crop 256 --epochs 60 --bptt 5 --device cuda \
  --val-scenes 3 --eval-crops 3 --eval-every 6 \
  --state-channels 24 --feature-channels 48 \
  --lr 3e-4 --rectify --robust-disocc --kernel-predict \
  --lr-upsampler --lr-dim 64 \
  --ssim-weight 1.0 --grad-weight 0.3 --edge-grad-weight 0.15 \
  --temporal-weight 0.3 \
  --save out/my_model.pt

# Fine-tune from an existing checkpoint (recommended)
python train.py \
  --engine-data /path/to/captures_ssaa \
  --crop 256 --epochs 30 --lr 1e-4 \
  --init-from out/beat_fsr_v64_sharp.pt \
  --distill-from out/beat_fsr_v64_sharp.pt --distill-weight 0.5 \
  ... # same architecture flags as above
  --save out/my_finetuned.pt
```

### Key training flags

| Flag | Description |
|------|-------------|
| `--feature-channels` | Encoder width (24/32/48). Sets the quality ceiling. |
| `--encoder-depth` | Number of encoder conv layers (default 2). |
| `--crop` | Training crop size. 256 recommended for full-frame generalization. |
| `--kernel-predict` | Use KPN output head (convex combination, no divergence). |
| `--lr-upsampler` | Enable learned LR-domain upsampler. |
| `--lr-dim` | LR upsampler width (64 recommended). |
| `--rectify` | Enable learned history rectification. |
| `--init-from` | Warm-start weights (strict=False, cross-architecture OK). |
| `--distill-from` | Teacher checkpoint for knowledge distillation. |
| `--ssim-weight` | Weight on (1 − SSIM) loss. |
| `--grad-weight` | Weight on gradient-L1 loss (edge sharpness). |
| `--edge-grad-weight` | Weight on edge-weighted gradient loss. |
| `--temporal-weight` | Weight on temporal consistency loss. |

## Layout

```
README.md                     This file
ORIENTATION.md                Findings on the FSR source code
HANDOFF.md                    Cross-machine context
prototype/
  train.py                    Training script with all flags
  fsrmamba/
    mamba.py                  Core model: SSM accumulator + all heads
    kpn.py                    Kernel prediction output head
    lrnet.py                  Learned LR-domain upsampler
    unet.py                   U-Net encoder (latency-optimised path)
    swin.py                   Swin transformer resolve refiner
    sepconv.py                Depthwise-separable convolution wrapper
    metrics.py                PSNR, SSIM, temporal, gradient, perceptual, freq losses
    engine_data.py            UE5 SSAA capture loader
    baseline.py               Ported FSR 3.1.4 accumulator (reference)
    colorspace.py             RGB ↔ YCoCg conversion
    cropbank.py               Pre-cropped training bank
    capture.py                Frame capture utilities
    synth.py                  Synthetic data generator (phase 1)
engine_capture/
  capture_scenes.py           Multi-scene UE5 capture orchestrator
  fsrapi_capture.patch        UE5 FSR integration patch for data dumping
```

## Attribution

`prototype/fsrmamba/baseline.py` and `colorspace.py` are derived from AMD's
FidelityFX SDK (MIT licence, Copyright (c) 2024 Advanced Micro Devices, Inc.),
with line references to the original HLSL throughout. FSR 3.1.4 is the newest
version AMD published source for.

## License

MIT
