# CONTINUE HERE — session 2 handoff (2026-07-22)

Read this first. It captures what session 2 built on the Windows machine, the
exact commands/locations so you don't re-derive them, the gotchas that cost real
time, results, and the next move. The original project context is in `HANDOFF.md`
+ `ORIENTATION.md` (macOS session 1); this supersedes them for anything
operational.

---

## TL;DR — where the project is

- Full **engine capture → ground-truth dataset → training** pipeline works end to
  end on this machine.
- The learned accumulator **beats FSR 3.1.4 on PSNR** (40.73 vs 40.17 dB on
  held-out real scenes) but **loses on SSIM and temporal stability**. The SSIM
  gap looks **architectural** (no sharpening head like FSR's RCAS), not
  loss-tunable — a mild SSIM loss did nothing, a strong one thrashed.
- **Strategic reframe (important):** beating FSR 3 is a *low bar* — FSR 4 and
  DLSS 4 are ML and already crush FSR 3, mostly on **temporal stability /
  artifacts**, not PSNR. The novel, defensible contribution was never "beat FSR
  3 on PSNR"; it is the **routing thesis** (temporally-stable recurrent expert
  routing on real content), which is **still unrun**. Next move should target
  that and/or temporal stability, not more PSNR grinding.

---

## 1. Filesystem — everything is on D:, never C:

C: has ~16 GB free and is OneDrive-synced; keep heavy stuff off it.

```
D:\FSR-Mamba\
├── repo\                         git repo (private: github.com/fishy-ops/FSR-Mamba)
│   ├── .venv\                    torch 2.13.0+cu126 (Python 3.13). NOT in git.
│   ├── fsr-upstream\             FidelityFX SDK v1.1.4 (FSR 3.1.4). GITIGNORED.
│   │   ├── bin\FFX_API_FSR_DX12.exe   the built sample (+ configs\, media next to it)
│   │   ├── bin\configs\cauldronconfig.json   framework config (resolution, pools)
│   │   ├── bin\configs\fsrapiconfig.json     scene/camera config
│   │   ├── media\                5.6 GB, 12 glTF scenes
│   │   └── samples\fsrapi\fsrapirendermodule.{cpp,h}   the instrumented sample
│   ├── prototype\                the PyTorch project
│   │   ├── fsrmamba\baseline.py  FSR accumulator ported to torch
│   │   ├── fsrmamba\mamba.py      the learned accumulator (+ routing)
│   │   ├── fsrmamba\capture.py    loads raw engine dumps -> tensors
│   │   ├── fsrmamba\engine_data.py  pairs FSR+GT, crops, tonemaps, caches
│   │   ├── fsrmamba\metrics.py    psnr/ssim/temporal + differentiable losses
│   │   ├── train.py               synthetic + --engine-data training
│   │   └── out\*.pt               trained checkpoints. GITIGNORED.
│   └── engine_capture\           the capture harness (committed)
│       ├── fsrapi_capture.patch   the C++ instrumentation (fsr-upstream is gitignored)
│       ├── capture_scenes.py      multi-scene FSR+GT orchestrator
│       └── README.md              full capture docs
├── captures\<scene>\{fsr,gt}\    the dataset (10 scenes; each has _cache_tm1.pt)
├── build_sample_only.cmd         rebuild the sample (Release, ReleaseDX12)
└── build_fsr_sample.cmd          full SDK+sample build (only needed once)
```

The venv python is `D:\FSR-Mamba\repo\.venv\Scripts\python.exe`. From
`prototype/` you can call it as `..\.venv\Scripts\python.exe`.

---

## 2. Environment / tooling facts

- **GPU:** desktop RTX 2070 SUPER, 8 GB. **215 W** TDP — but MSI Afterburner had
  the power limit slid down to ~125 W (looked like a mobile part in nvidia-smi).
  If perf looks capped, check Afterburner's power-limit slider is at 100%.
- **8 GB VRAM is the hard limit.** VRAM spillover into shared system RAM is
  catastrophic (PCIe paging → training crawls). Keep peak under ~7 GB.
  `nvidia-smi` shows *reserved* (PyTorch caching allocator), which reads higher
  than Task Manager's active number — trust nvidia-smi.
- **Toolchain:** VS 2022 Build Tools (MSVC 14.44), Win11 SDK 10.0.26100, CMake —
  use the **VS-bundled 3.31.6**, not standalone 4.x (4.x rejects the SDK's old
  `cmake_minimum_required`).
- **gh CLI:** `"C:\Program Files\GitHub CLI\gh.exe"` (logged in as fishy-ops).
- **Shell:** the Bash tool (Git Bash) is easiest for git/python/launching. Env
  vars set inline in bash DO propagate to child exes.

---

## 3. How to build the FSR sample

The C++ instrumentation lives in `engine_capture/fsrapi_capture.patch` because
`fsr-upstream/` is gitignored. After a fresh clone of the SDK:

```bash
cd fsr-upstream && git apply ../engine_capture/fsrapi_capture.patch
```

To rebuild after editing `fsr-upstream/samples/fsrapi/fsrapirendermodule.{cpp,h}`
(or the framework camera):

```bash
# MUST close the running exe first — it locks the file:
taskkill //F //IM FFX_API_FSR_DX12.exe
# rebuild (background, ~2-5 min). Launch, then watch the log for the marker:
cmd //c "D:\\FSR-Mamba\\build_sample_only.cmd" > D:/FSR-Mamba/build.log 2>&1 &
# wait for: "=== SAMPLE BUILD OK ==="  (grep the log)
```

**Build gotchas (cost hours in session 2):**
1. AMD's menu maps "FSR" to `-DFFX_FSR1=ON`, which does NOT build the FSR3
   upscaler. The scripts already pass `-DFFX_FSR3=ON`. Keep it.
2. The samples solution renames configs per-backend: build `--config ReleaseDX12`,
   not `Release` (plain Release → MSB8013). The scripts handle this.
3. The sample links the **prebuilt signed** `amd_fidelityfx_dx12.dll` (fine — we
   want working FSR buffers). The from-source SDK build only validates the
   toolchain.

---

## 4. How to run capture

**Launch the exe via Git Bash `./FFX_API_FSR_DX12.exe`, NOT `cmd`** — `cmd` on
this machine does not search the working dir for executables (it silently fails
with "not recognized"). subprocess-by-absolute-path also works.

Env-var interface (no config edits):
```
FSRMAMBA_CAPTURE_DIR         output dir (also enables capture)
FSRMAMBA_CAPTURE_WARMUP_SEC  wall-clock seconds before capturing (use 60)
FSRMAMBA_CAPTURE_COUNT       frames to grab (default 32)
FSRMAMBA_CAMERA_ORBIT        auto-pan yaw rad/frame (use 0.006) -> motion vectors
FSRMAMBA_UPSCALER            "fsr" (upscale) or "native" (ground truth)
FSRMAMBA_SCALE               FSR ratio (2 = 2x)
```

Capture the whole dataset (already done, but this is how):
```bash
cd engine_capture
../.venv/Scripts/python.exe capture_scenes.py --out D:/FSR-Mamba/captures
# or a subset: --scenes sponza bistro
```

**Capture gotchas (each cost a debug loop in session 2 — do NOT re-discover):**
- **Warmup must be wall-clock TIME, not a frame count.** The HDD is the
  bottleneck; the app renders empty frames fast while assets stream slowly, so a
  frame count elapses long before the scene is drawn. `WARMUP_SEC=60`. Symptom of
  getting this wrong: depth/motion come back all-zero (empty room) — it is NOT a
  GPU/capture bug.
- **Motion vectors need camera movement.** Static camera → zero MVs. The orbit
  env var handles it; it also gates on warmup and pans back-and-forth so it stays
  in the scene interior (a continuous orbit drifts out to the exterior).
- **Heavy scenes (Bistro/Brutalism) overflow the DynamicBufferPool** — a
  fixed-size CPU-side pool (default 75 MB in `cauldronconfig.json`), NOT VRAM.
  Overflow → critical assert → modal-dialog hang that looks like OOM while VRAM
  stays low. Fix: `Allocations.DynamicBufferPoolSize = 512 MB` (the orchestrator
  sets this).
- **Frame generation is ON by default** and inserts interpolated (fake) frames
  with no ground truth — forced OFF during capture.
- **The sample remembers a manual UI Method change** — it silently left the
  "FSR" pass at 1.0x (no upscaling). The env vars force method + scale, so don't
  rely on defaults.
- Renders at **1920x1080** (960x540 render at 2x) — the project's 1080p scope.
- 2 of 12 scenes (MetalRoughSpheres, animated Toyshop) have no embedded camera
  and are skipped (the framework default camera is hardcoded to a Sponza pose).

---

## 5. How to train

```bash
cd prototype
../.venv/Scripts/python.exe -u train.py --engine-data D:/FSR-Mamba/captures \
  --crop 256 --epochs 120 --bptt 6 --device cuda \
  --val-scenes 3 --eval-crops 8 \
  --state-channels 16 --feature-channels 24 --lr 1.5e-3 \
  --save out/model.pt
```

Key args: `--crop` (render-res crop; out = crop*scale), `--bptt`,
`--state-channels` + `--feature-channels` (capacity), `--lr`, `--epochs`,
`--eval-crops` (crops per val scene — use >=8 for a trustworthy number),
`--ssim-weight`, `--grad-weight`, `--temporal-weight`, `--num-experts`.

**Training gotchas / tips:**
- **First load decodes rg11b10 HDR color (~17 s/scene, ~3 min for 10) — this is
  CPU-only.** It is cached to `_cache_tm1.pt` next to each capture, so later runs
  load in seconds. Delete the caches to force re-decode.
- **The GPU looks under-used during training** because the ported `_upsample`
  uses a **3x3 Python loop** launching tiny kernels — it is launch/latency-bound,
  not compute-bound. Vectorizing it is the biggest available speedup (and would
  actually saturate the card). Not done yet.
- **Motion-vector convention:** engine MVs already match the synthetic UV
  convention exactly (calibrated by warp-alignment: scale 1.0, +x/+y). No
  conversion needed.
- **Bigger model needs a lower LR + the warmup**, or it diverges to a constant-
  output collapse (frozen eval at ~13 dB PSNR / 0.08 SSIM / 0.0 temporal). The
  warmup is built in now.
- **VRAM sizing that fits 8 GB:** crop 256/512 + bptt 6/8 + feature 24-64 sits
  at 4-7 GB. `feature 64 / crop 512 / bptt 8` spilled — drop bptt to 6.
- Eval is **multi-crop** (`--eval-crops` seeded crops per val scene, averaged) —
  a single crop is noisy (an early single-crop baseline read 47 dB; the true
  multi-crop baseline is 40.17). Always trust the multi-crop number.
- Held-out **scenes** for val (not frames). Best checkpoint saved by PSNR+SSIM
  composite, not last epoch (the recurrent curve is noisy).

---

## 6. Results (held-out bistro/brutalism/chess, multi-crop, tonemapped)

Baseline **FSR 3.1.4 (captured, real output):** PSNR **40.17**, SSIM **0.9744**,
temporal **0.01273**.

| Run | PSNR | SSIM | temporal | verdict |
|---|---|---|---|---|
| simple L1 (feat 24, state 16, lr 1.5e-3) | **40.73** | 0.948 | 0.0140 | wins PSNR only. **Best so far.** `out/engine_serious.pt` |
| big + SSIM loss (feat 64, state 32, lr 5e-4) | 39.83 | 0.948 | 0.0142 | loses all 3 (undertrained + traded PSNR) |
| strong SSIM/grad + high lr | thrashed | — | — | unstable, killed |

**The finding:** SSIM plateaus at ~0.948 regardless of loss weight → it is
architectural. FSR wins SSIM via its dedicated **RCAS sharpening** pass, which the
learned model has no equivalent of. PSNR (fidelity) it wins; structure/temporal
FSR still holds.

---

## 7. Next moves (in recommended order)

1. **Run the routing experiment (the actual research question, still unrun).**
   `--num-experts 4 --balance-weight 0.01` on the engine data. On the homogeneous
   synthetic data routing showed no benefit; the whole point of capturing
   heterogeneous real content was to test whether it helps now. Also check
   `prototype/diagnose_router.py`. This is a real result either way and needs no
   new infra.
2. **Target temporal stability, not PSNR** — it is the metric upscalers are
   judged on and where FSR 4 / DLSS 4 made their gains. Measure/optimize it
   directly (stronger temporal loss, longer BPTT).
3. **If you want a clean FSR-3 win:** add a sharpening head (learned high-freq
   residual, or port FSR's RCAS onto the learned output) — the SSIM gap is
   architectural, not loss-tunable. But note this is polishing a low bar (see
   TL;DR).
4. **Speed:** vectorize the ported `_upsample` 3x3 loop (baseline.py) — biggest
   iteration-time win.
5. **Housekeeping:** merge PR #1 (CUDA fix + lock pass) and PR #2 (this session's
   capture harness + training). Neither is merged. Both branch off `main`.

---

## 8. Git / PRs

- Working branch: `feature/engine-capture-harness` → **PR #2**
  (https://github.com/fishy-ops/FSR-Mamba/pull/2). All of session 2's work.
- `fix/cuda-device-plumbing` → **PR #1** (CUDA device fix + FSR lock-pass port
  from session 1's follow-up).
- Both off `main`, independent, **neither merged** — user's call.
- Commit messages end with the Co-Authored-By trailer. Don't push to `main`.

---

## 9. Tips for the next chat (save time & money)

- **Don't re-derive the filesystem or build process** — it's all above.
- **Launch long jobs in the background** and watch the log with an `until grep`
  loop; the `&` detaches so the launcher returns immediately (watch the log
  file, not the task exit).
- **Kill stray processes:** `taskkill //F //IM python.exe` /
  `taskkill //F //IM FFX_API_FSR_DX12.exe`.
- **Read `MEMORY.md`** in the memory dir — it indexes the persistent facts.
- The heavy operational knowledge is also in `engine_capture/README.md`.
- Don't chase "beat FSR 3 on all metrics" without re-reading the TL;DR strategic
  note first — it may be the wrong goal.
