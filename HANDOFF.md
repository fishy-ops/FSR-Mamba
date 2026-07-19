# Handoff — start here

Written 2026-07-18 at the end of the macOS session, for whoever (or whatever)
picks this up on the Windows machine. The prior chat does not transfer; this
file plus `ORIENTATION.md` is the context.

## What this project is

A fork of AMD's FidelityFX Super Resolution **3.1.4** that replaces its
hand-tuned temporal accumulator with a **learned recurrent (state-space)
accumulator**, plus content-adaptive expert routing.

Primary motivation is **portfolio/resume**, not publication. That shapes
priorities: a working, measured result beats a novel-but-broken one, and honest
framing beats an inflated novelty claim.

**Novelty, stated accurately** (this matters — the inflated version does not
survive an interview): mixture-of-experts for super-resolution is well covered
already (ClassSR, MoESR, SkipVSR, SP-MoMamba). Mamba for video SR likewise. What
no prior work was found for is *temporally-stable recurrent expert routing inside
a real-time rendered upscaler* — because all the prior art is single-frame and
therefore never had to solve router flicker or GPU wave divergence. That narrow
claim is the defensible one.

## Hardware and constraints on the target machine

- **RTX 2070 Super (8 GB VRAM)**, i7-9700, 64 GB RAM
- **Only ~80–90 GB free storage** — this is the binding constraint
- Cannot benchmark against FSR 4 (AMD-only, RDNA4/RX7000). This is fine: the
  meaningful comparison is against FSR 3.1.4 in the same pipeline, one variable
  changed. DLSS does run on this card if an external reference point is wanted.

Storage plan that fits: Visual Studio **Build Tools** (not the full IDE, saves
~20 GB) ≈ 8–10 GB, shallow FSR clone ≈ 430 MB, torch+CUDA ≈ 8 GB. Leaves ~55 GB
for capture data.

**8 GB VRAM means training must use crops** (256×256) with BPTT over 8–16
frames. Full-frame backprop at 1080p will not fit. Evaluate on full frames.

## What already exists and works

| | |
|---|---|
| `prototype/fsrmamba/synth.py` | Analytic synthetic scenes: paired LR/GT with **exact** motion vectors. No engine or GPU needed. |
| `prototype/fsrmamba/baseline.py` | FSR 3.1.4's accumulator ported to PyTorch, line-referenced to the HLSL. The number to beat. |
| `prototype/fsrmamba/mamba.py` | Learned recurrent accumulator + optional expert routing. |
| `prototype/fsrmamba/metrics.py` | PSNR, SSIM, motion-compensated temporal instability. |
| `prototype/diagnose_router.py` | Distinguishes a real routing null result from a collapsed router. |

Result on held-out synthetic scenes, 2× upscale: **20.44 dB / 0.831 SSIM /
0.0063 temporal instability**, versus baseline **19.04 / 0.786 / 0.0281**. That
is +1.4 dB and 4.5× better temporal stability from an 11k-parameter model.

## The four things most likely to trip up the next session

**1. The version map is confusing and easy to get wrong.**
`v1.1.4` is the git tag containing FSR Upscaler **3.1.4** — the newest version
AMD published source for. SDK `v2.x` and `main` ship FSR 4.x as **signed DLLs
with no upscaler source**. FSR 3.1.5 exists in docs but its source was never
released. Do not go looking for a `3.1.x` tag; there isn't one.

**2. The recurrent state is only 4 channels wide, and that is the graft point.**
`accumulate.h:165` stores `float4(historyColor, lock)`. The whole project is
widening that to N channels and learning the update rule. The entry point is
`Accumulate()` at `accumulate.h:141` — 172 lines, and the entire thesis lives
there.

**3. FSR's resolve is a *sliced* Lanczos filter.** Each frame contributes only
`fAverageLanczosWeightPerFrame` ≈ 0.046 of weight against a history weight of up
to 1.0 (`common.h:47`, applied at `upsample.h:629`). Miss that one constant and
temporal accumulation silently stops working while output still looks plausible.
This cost the most time in the last session. Related traps: the Lanczos is
**radial**, not separable; the rectification box uses a **different** kernel
(`exp(-2.3·d²)`); lock constants are 1.0 / 2.0.

**4. Hold out *scenes*, not frames.** Consecutive frames are near-duplicates and
a recurrent model carries state across them, so a frame-wise split leaks almost
everything. `random_scene(seed)` exists for this. An earlier evaluation
accidentally trained on the test set and produced a meaningless win.

## Architecture lesson worth keeping

The first learned model used a fully abstract state and predicted a residual on
the resolve. It started at *spatial-only* quality and barely improved — it had to
rediscover temporal accumulation from nothing.

Making it a **strict generalisation of FSR** fixed it: history colour in the
first 3 state channels, output = `lerp(warped_history, resolved, alpha) +
residual`, with `alpha`'s bias initialised to FSR's converged blend rate
(≈0.045). It then started as a working accumulator and beat baseline within one
epoch. Generalise the incumbent; don't reinvent it. As a bonus, it makes the
comparison fair — FSR's heuristic lies inside the model's hypothesis space.

## Open items, in priority order

1. **Port the lock pass** (`ffx_fsr3upscaler_lock.h`). The current baseline omits
   it, plus luma instability and the reactive masks, so it is **weaker than real
   FSR 3.1.4**. The +1.4 dB win overstates until this is fixed. Highest-value
   correctness work, and it is unblocked.
2. **Engine capture.** Now needed for the *headline* claim, not just robustness:
   routing showed no benefit and the synthetic testbed was diagnosed as too
   homogeneous to test it (every scene from one 4-layer template). Capture from
   the FSR sample — it ships scenes with every buffer already wired.
   Capture at 1080p output / 540p render, motion vectors and depth at **render**
   resolution (`FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS`), short
   sequences of 16–24 frames, compressed. ≈7 MB/frame → ~11 GB for 1600 frames.
3. **Measure cost at 4K.** Parameter count is not milliseconds.
4. **Scope decision, still open and needs the user:** 1080p quality-only proof
   (~2 months) versus full HLSL shader port (6–12 months).

## Routing status: built, no benefit yet, and correctly diagnosed

Ablation at matched state budget: 1 expert 20.32 dB / 0.0055 TI, 4 experts
20.33 / 0.0053. No effect, well inside noise.

`diagnose_router.py` showed the router fails in two opposite ways: without a
load-balancing loss it starves 2 of 4 experts (usage `[0.554, 0.032, 0.004,
0.411]`); with one at weight 0.05 it goes near-uniform (entropy 0.998). Neither
helps. Tuning between them would be fishing, not testing.

One supporting finding: **router flip rate is 7–16% of pixels per frame**, so
router churn is real and measurable — the thing single-frame prior work never
faced. The hysteresis term is justified; it just isn't buying quality yet on
homogeneous data.

**Do not read this as the idea failing.** The experiment is underpowered to
detect the effect. Fixing it needs content diversity, which means engine capture.

## First commands on the new machine

```bash
git clone --depth 1 --branch v1.1.4 \
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git fsr-upstream
python -m venv .venv
# torch with CUDA from https://pytorch.org — do not guess the wheel
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"

cd prototype
..\.venv\Scripts\python run_baseline.py --frames 24 --height 360 --width 640
```

That last command is the canary: FSR should beat `lanczos` on temporal
instability by roughly 5×. If it doesn't, something broke in transit.
