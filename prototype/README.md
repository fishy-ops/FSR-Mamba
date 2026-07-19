# prototype — PyTorch harness

Phase-one workspace. Everything here runs on the Mac; nothing needs Windows or a
discrete GPU yet.

```
../.venv/bin/python run_baseline.py --frames 24 --height 360 --width 640 --dump out
```

## What's here

| File | Purpose |
|---|---|
| `fsrmamba/synth.py` | Analytic synthetic scenes — paired LR/GT data with exact motion vectors, no engine required |
| `fsrmamba/baseline.py` | FSR 3.1.4's temporal accumulator ported to PyTorch — the number to beat |
| `fsrmamba/colorspace.py` | FSR's exact YCoCg transform |
| `fsrmamba/metrics.py` | PSNR, SSIM, motion-compensated temporal instability |
| `run_baseline.py` | Runs a sequence, compares bilinear / spatial-only / full FSR |

## Phase-one result: the learned accumulator beats the baseline

```
../.venv/bin/python train.py --epochs 10 --seq-len 10 --bptt 5 --height 144 --width 256
```

Trained on scene seeds 0–5, evaluated on **unseen** seeds 1000–1002:

| | PSNR (dB) | SSIM | temporal instability |
|---|---|---|---|
| FSR baseline | 19.04 | 0.786 | 0.0281 |
| **learned (11k params)** | **20.44** | **0.831** | **0.0063** |

+1.4 dB, +0.044 SSIM, and **4.5× more temporally stable** — with an 11,000-parameter
model after 10 epochs. The temporal number is the one that matters.

### Why it works: the model is a strict generalisation of FSR

The first version kept the state fully abstract and predicted a residual on the
Lanczos resolve. It started at *spatial-only* quality and crawled — it had to
rediscover temporal accumulation from scratch.

The fix was to keep FSR's structure: first 3 state channels hold history colour
in YCoCg, output is `lerp(warped_history, resolved, alpha) + residual`, with
`alpha` predicted per pixel and its bias initialised to ≈0.045 to match FSR's
converged blend rate. The model now *starts* as a working temporal accumulator
and spends capacity on improving the blend decision — which is the actual thesis.

It also makes the comparison fair in the other direction: FSR's heuristic lies
inside this model's hypothesis space, so a win can't be an artifact of the two
having different architectures.

### Four caveats, all of which matter

1. **The baseline is weakened.** No lock pass, no luma instability, no
   reactive/shading-change masks. Beating it is *not* beating real FSR 3.1.4.
   This is the single biggest asterisk on the result.
2. **Motion vectors are perfectly truthful.** Synthetic scenes have no shadows,
   reflections, or VFX, so the MVs never lie — and learning when they lie is the
   whole point of the project. This is the easy case by construction.
3. **Scene diversity is narrow.** All scenes come from one 4-layer template with
   randomised parameters, so "generalisation" here means within-distribution.
4. **No performance measurement.** 11k parameters is small, but nobody has
   measured what these convs cost at 4K. Quality headroom means nothing until
   the millisecond budget is checked.

## Phase two: expert routing shows no benefit yet — and the testbed can't tell why

Routing is implemented (`--num-experts K`), with the routing decision carried in
the hidden state and reprojected along motion vectors, which is the novel part
of the design. The ablation, same state budget (12 channels), same schedule:

| | PSNR | SSIM | temporal instability |
|---|---|---|---|
| 1 expert | 20.32 | 0.829 | 0.0055 |
| 4 experts | 20.33 | 0.830 | 0.0053 |

**No effect.** The difference is far inside run-to-run noise — the 1-expert run
alone swung 20.01 → 20.60 → 20.32 across epochs.

### The router was probed, not assumed

`diagnose_router.py` distinguishes "routing doesn't help" from "the router never
routed", which an aggregate metric cannot:

| | mean max weight | entropy ratio | per-expert usage | flip rate |
|---|---|---|---|---|
| no balance loss | 0.444 | 0.762 | `[0.554, 0.032, 0.004, 0.411]` | 0.073 |
| balance loss 0.05 | 0.266 | 0.998 | `[0.007, 0.296, 0.242, 0.455]` | 0.160 |

Both configurations fail, in opposite directions. Without a load-balancing loss
the router specialises but starves two of four experts — a 2-expert model paying
4 experts' compute. With one, usage balances but the weights go near-uniform,
which is four experts averaged into one. Tuning the balance weight between these
might find a middle, but that is fishing for a positive result, not testing a
hypothesis.

One finding does support the design premise: the **router flip rate is 7-16% of
pixels per frame**. Router churn is real and measurable here, which is exactly
the flicker that single-frame routing work never had to confront. The hysteresis
term is doing something; it just isn't buying quality yet.

### Why this is not evidence against the idea

The experiment is underpowered to detect the effect it is designed to test:

1. **The scenes are too homogeneous.** Every scene comes from one 4-layer
   template with jittered parameters. Routing's entire premise is that different
   *kinds* of content want different treatment — and there are only really four
   kinds here, present in every scene in similar proportion. There is not much
   for experts to divide up.
2. **The model is underfit.** 20k parameters, 10 epochs, loss still falling.
   Mixtures help when a single model is capacity-limited; this one is not yet.
3. **Dense combination hides the sparse case.** Quality is measured with all
   experts mixed, which is the ceiling, not what a tile-routed shader would do.

The honest conclusion: **testing the routing hypothesis properly needs content
diversity that synthetic scenes cannot provide.** That makes engine capture a
prerequisite for the headline claim, not just for the robustness story — which
raises the priority of the Windows/GPU decision considerably.

## Baseline-only numbers

24 frames, 320×180 → 640×360 (2×), on `default_scene`:

| method | PSNR (dB) | SSIM | temporal instability |
|---|---|---|---|
| bilinear | 12.82 | 0.538 | 0.173 |
| lanczos (spatial only) | 10.36 | 0.372 | 0.253 |
| **fsr (temporal)** | **13.18** | 0.527 | **0.049** |

The temporal instability column is the one that matters. FSR is ~3.5× more
stable than bilinear and ~5× more stable than the same spatial filter without
accumulation — that gap *is* temporal accumulation working. Any learned
accumulator has to beat 0.049 without giving back PSNR.

## Two honest caveats about these numbers

**The scene is deliberately brutal.** The zone-plate backdrop contains
frequencies far above Nyquist across most of the frame, where the correct answer
is flat grey. That depresses absolute PSNR for everyone and flatters blurry
methods on SSIM — which is why bilinear's SSIM looks competitive despite being
visibly worse in motion. A milder, more game-like scene is a TODO before these
numbers get quoted anywhere.

**This baseline is weaker than real FSR 3.1.4.** The lock detection pass, luma
instability, and the reactive/shading-change masks are not ported (see the
`baseline.py` docstring for the full list). The visible ghosting on the moving
glyph card traces directly to the missing lock pass. Treat these figures as a
floor, not as "FSR's score", and do not publish a comparison against them as
though they represented AMD's shipping quality.

## Design note: why synthetic data

Capturing real paired data needs Windows + a discrete GPU, which is a hard
blocker on Apple Silicon. Analytic scenes sidestep that: content is a closed-form
function of position and time, so ground truth is just supersampling and motion
vectors are exact.

The tradeoff is real and worth restating. These scenes have no shading, no
specular, no transparency — and perfectly *truthful* motion vectors. Real engine
data has motion vectors that lie (shadows, reflections, VFX), and learning when
to distrust them is the actual thesis of this project. So synthetic data proves
the mechanism works; it cannot prove robustness. Engine capture is still required
before any quality claim means anything.

## Next

1. **Expert routing** — the phase-two contribution, with routing state carried in
   the hidden state for temporal stability.
2. **Port the lock pass** so the baseline is a fair opponent. Until this is done
   the headline result overstates the win.
3. **Measure cost at 4K** — parameter count is not milliseconds.
4. **Engine capture** on Windows, for motion vectors that lie.
