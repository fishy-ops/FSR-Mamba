# FSR-Mamba — Orientation

Prepared 2026-07-18. Read this first when you come back.

---

## 1. Version situation (correction to what I told you earlier)

I said FSR 3.1.5 was the newest open-source version. **That was wrong — the newest with
actual source is FSR Upscaler 3.1.4.** 3.1.5 exists in AMD's docs, but the source was
never published for it.

The repo's version numbering is genuinely confusing, so here's the map:

| Git tag        | Upscaler version | Source available?              |
|----------------|------------------|--------------------------------|
| `v1.1.4`       | **FSR 3.1.4**    | **Yes — full C++ + HLSL/GLSL** |
| `fsr3-v3.0.4`  | FSR 3.0.4        | Yes                            |
| `v2.0.0`–`v2.3.0` (incl. `main`) | FSR 4.x | **No — signed DLLs only** |

SDK v2.x is where AMD switched to the closed DLL model. `main` today ships FSR 4.1.1 as a
binary; the upscaler source is gone from it. So `v1.1.4` is the end of the open line and
that's what we're on.

Licence is MIT (`LICENSE.txt`, Copyright 2024 AMD). Cleanly forkable and commercialisable.

## 2. What's on disk

```
FSR_Mamba/
├── ORIENTATION.md      ← this file
└── fsr-upstream/       ← FidelityFX SDK @ tag v1.1.4 (FSR 3.1.4), 847 MB
```

`fsr-upstream/` is **pristine upstream — don't edit it.** It's our reference and our
baseline to diff against. When we start changing things, work happens in a sibling
directory so `git diff` against upstream always stays meaningful.

The clone has full history, so you can `git log`/`git diff` between 3.0.4 and 3.1.4 to see
what AMD changed and why — genuinely useful, since the 3.0→3.1 delta is mostly temporal
stability work, which is our exact problem area.

## 3. The graft point — this is the whole project

**`fsr-upstream/sdk/include/FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_accumulate.h`**

172 lines. That's it. That file is the hand-tuned temporal accumulator you're proposing to
replace with a learned one. The entry point is `Accumulate(FfxInt32x2 iPxHrPos)` at
**line 141**, and it reads as a clean five-step recipe:

```
141  Accumulate(iPxHrPos)
147    ReprojectHistoryColor()      ← move the notes along the motion vectors
151    UpdateLockStatus()           ← hand-tuned "is this a thin detail worth protecting?"
153    ComputeBaseAccumulationWeight()  ← how much do I trust memory this frame
155    ComputeUpsampledColorAndWeight()
157    RectifyHistory()             ← clamp history to a colour box (anti-ghosting)
159    Accumulate()                 ← the actual blend
165    StoreInternalColorAndWeight()    ← write the notes for next frame
```

Everything I described conceptually maps to a named function here. `RectifyHistory` at
line 44 is the neighbourhood colour-clamp; `UpdateLockStatus` at line 72 is the thin-detail
protection. Both are pure heuristic — look at the magic numbers (`3.0f`, `1.0f`, `0.15f`,
`20.0f`, `1.7f`). Those constants are years of AMD engineers tuning against artifacts.
Replacing them with something learned is the thesis.

## 4. The recurrent state is 4 channels — confirmed

This is the single most important fact for planning, and it's better news than expected.

`accumulate.h:165`:
```c
StoreInternalColorAndWeight(iPxHrPos, FfxFloat32x4(data.fHistoryColor, data.fLock));
```

The entire frame-to-frame memory is **one RGBA texture**: RGB history colour + one scalar
lock value. That's the "notes." It ping-pongs between two buffers
(`INTERNAL_UPSCALED_COLOR_1`/`_2`, selected by frame parity — `ffx_fsr3upscaler.cpp:872`).

So your Mamba change is concretely: **widen this from 4 channels to N**, and replace the
hand-written update rules with a learned one. The plumbing for a per-pixel recurrent state
already exists — you're extending it, not inventing it. That's a much smaller surface than
it sounded like in the abstract.

Budget note: at 4K, each extra channel is ~8.3M × 2 bytes (fp16) = ~17 MB of read+write
traffic per frame. 8 channels ≈ 133 MB/frame. That's the number to keep an eye on.

## 5. The warp — "moving the notes"

**`ffx_fsr3upscaler_reproject.h:63`**, `ReprojectHistoryColor()`:

```c
fReprojectedHistory = HistorySample(params.fReprojectedHrUv, PreviousFrameUpscaleSize());
```

and `ComputeReprojectedUVs` (line 56) is just:
```c
params.fReprojectedHrUv  = params.fHrUv + params.fMotionVector;
params.bIsExistingSample = IsUvInside(params.fReprojectedHrUv);
```

Note two things we discussed, now visible in code:

- The resample is a **bicubic/Lanczos** filter (line 32–33), not bilinear. Higher quality
  than I assumed, but the point stands — it's still lossy interpolation applied to the
  state every single frame. This is your "smudging the notes" problem, and it's where the
  question of whether learned state survives interpolation gets real.
- `bIsExistingSample` is the **disocclusion test**, and it's binary — off-screen means no
  history, full stop. Line 147 skips reprojection entirely in that case, so the state falls
  back to whatever `InitPassData` zeroed it to. A learned init here is an easy, cheap early
  win worth trying.

## 6. Full pipeline order

From `ffx_fsr3upscaler.cpp:1142`, the passes dispatch in this order each frame:

```
PrepareInputs → LumaPyramid → ShadingChangePyramid → ShadingChange
  → PrepareReactivity → LumaInstability → Accumulate[+Sharpen] → RCAS → [DebugView]
```

Only `Accumulate` is in scope for phase one. Everything upstream of it produces the masks
it consumes (reactive / disocclusion / shading-change / accumulation — unpacked at
`accumulate.h:125-129`). Those four masks are worth understanding, because they're
hand-computed signals your network might learn to derive better itself — or might want as
free input features. Probably start by feeding them in as inputs rather than replacing
them.

## 7. ⚠️ Blocker to plan around: you're on an Apple Silicon Mac

The build scripts are `BuildSamplesSolutionDX12.bat` / `...VK.bat` — Windows batch files.
FSR's sample app needs Windows + Visual Studio + DirectX 12 or Vulkan. **You cannot build
or run this on macOS**, and there's no reasonable port.

This doesn't block the project, but it does shape the plan:

- ✅ **On the Mac:** read the source, build the PyTorch prototype, train, evaluate. This is
  the majority of the work and all of phase one.
- ❌ **Needs a Windows box with a discrete GPU:** running the sample, capturing training
  data from it, and eventually the HLSL port.

Which means the data question from our earlier conversation gets sharper: you need a
Windows machine (or cloud GPU instance) at *some* point regardless. Worth deciding early
whether that's a dual-boot, a desktop you have access to, or a cloud VM — it gates the
capture step.

## 7b. Progress — the Mac blocker is mostly sidestepped for phase one

Since writing section 7 I built `prototype/` (see its README). Two things landed:

**Synthetic scenes** (`prototype/fsrmamba/synth.py`) generate paired training data
analytically — procedural textures on layers moving along known paths, so ground
truth is just supersampling and motion vectors are exact. This means steps 1, 4
and 5 below no longer wait on Windows. Engine capture is still needed eventually,
because synthetic motion vectors never lie and learning when they lie is the
whole point — but it is no longer the *first* blocker.

**The FSR baseline is ported and verified** (`prototype/fsrmamba/baseline.py`).
It beats spatial-only upscaling by 2.8 dB and is ~5× more temporally stable,
which is the signature of accumulation actually working.

Worth knowing what that took, because it is the kind of bug that would have
quietly poisoned every downstream result: the first version had no temporal
convergence at all. FSR's resolve is a *sliced* Lanczos filter — each frame
contributes only `fAverageLanczosWeightPerFrame` ≈ 0.046 of weight against a
history weight of up to 1.0 (`common.h:47`, applied at `upsample.h:629`). Miss
that one constant and every frame overwrites its own history, while the output
still looks superficially reasonable. Three related bugs came with it: the
Lanczos kernel is radial rather than separable, the rectification box uses a
different kernel entirely (`exp(-2.3·d²)`), and the lock constants are 1.0/2.0.

## 7c. Phase one works — the learned accumulator beats the baseline

On held-out scenes (trained seeds 0–5, evaluated seeds 1000–1002): **+1.4 dB
PSNR, +0.044 SSIM, 4.5× better temporal stability**, from an 11k-parameter model.
Details and caveats in `prototype/README.md`.

Two things were learned the hard way and are worth carrying forward:

**Architecture: keep FSR's structure.** The first design used a fully abstract
state and a residual on the resolve. It started at spatial-only quality and
barely improved — it had to rediscover temporal accumulation from nothing. Making
the model a *strict generalisation* of FSR (history colour in the first 3 state
channels, output = `lerp(warped_history, resolved, alpha) + residual`, alpha bias
initialised to FSR's converged blend rate) changed it from "barely learns" to
"beats baseline in one epoch". Generalise the incumbent; don't reinvent it.

**Methodology: hold out scenes, not frames.** The first evaluation used the same
scene for train and val, which is training on the test set. Consecutive frames of
one sequence are near-duplicates and a recurrent model carries state across them,
so a frame-wise split leaks nearly everything. `random_scene(seed)` exists to make
the split honest.

**The headline caveat:** the ported baseline is weaker than real FSR 3.1.4 (no
lock pass, no luma instability, no reactive masks). Beating it is not beating
AMD's shipping quality, and the comparison should not be presented as though it
were. Porting the lock pass is the highest-value correctness work outstanding.

## 7d. Routing is built, and it does nothing yet

Expert routing is implemented with the decision carried in the hidden state and
reprojected along the motion vectors — the novel part. On held-out scenes it
makes **no measurable difference** (20.33 vs 20.32 dB, inside noise).

The router was probed rather than assumed (`prototype/diagnose_router.py`). It
fails in two opposite ways depending on the load-balancing weight: without one it
starves two of four experts; with one it goes near-uniform and averages them.
Router flip rate is 7–16% of pixels per frame, which does confirm that router
churn is a real phenomenon here — the thing single-frame prior work never faced.

**This is not evidence against the idea.** The testbed cannot detect the effect:
every scene comes from one 4-layer template, so there are only ~4 content types,
present in similar proportion everywhere. Routing needs heterogeneous content to
have anything to divide up.

**Consequence for planning:** engine capture is now a prerequisite for testing
the *headline claim*, not merely for the robustness story. That moves the
Windows/GPU question from "eventually" to "blocking the interesting result".

## 8. Suggested next steps

1. **Read `accumulate.h` end to end.** It's 172 lines and it is the entire project. Don't
   move on until the five steps make sense.
2. **Decide the Windows/GPU access question.** It gates training data, which gates
   everything else.
3. **Settle the data plan.** Either capture from the FSR sample (it ships scenes and
   already has all the buffers wired up, which is a real head start) or from UE5.
4. **Build the PyTorch harness** — reimplement FSR's accumulator in Python as a baseline
   first. Sounds like busywork; it isn't. It gives you a reference to beat, validates that
   your data pipeline is correct, and forces you to actually understand the heuristics.
5. *Then* swap in the recurrent accumulator. Routing comes after that works.

---

**Open questions for you:**
- Windows/GPU access — what do you have?
- Data source: FSR sample (easier, already wired) vs UE5 (more realistic content)?
