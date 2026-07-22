# Engine capture harness

Instruments AMD's FidelityFX FSR sample (`fsrapi`) to dump per-frame training
data — the FSR upscaler's inputs and output — for training the PyTorch
accumulator on real rendered content instead of synthetic scenes.

`fsr-upstream/` is gitignored (kept pristine and re-cloneable), so the C++
changes live here as a patch rather than in-tree.

## What it captures

Per frame, into `$FSRMAMBA_CAPTURE_DIR`:

| file | buffer | notes |
|---|---|---|
| `frame_NNNN_color_lr.bin` | low-res input color | render-res sub-rect of a display-sized target |
| `frame_NNNN_color_hr.bin` | FSR upscaled output | the baseline to beat (not ground truth) |
| `frame_NNNN_depth.bin` | depth | render-res |
| `frame_NNNN_motion.bin` | motion vectors | render-res; only nonzero while the camera moves |
| `frame_NNNN.json` | metadata | render/upscale sizes, jitter, per-buffer dtype/dims |

Load them with `prototype/fsrmamba/capture.py` (`load_sequence(dir)`), which
crops each buffer to its valid region and returns tensors shaped like the
synthetic `render_sequence` frames.

## Applying the patch

```bash
cd fsr-upstream
git apply ../engine_capture/fsrapi_capture.patch
# then rebuild the FSR sample (see ../HANDOFF or the build_*.cmd scripts on D:)
```

## Running a capture

Driven entirely by environment variables — no config-file edits:

```
FSRMAMBA_CAPTURE_DIR         output directory (also enables capture when set)
FSRMAMBA_CAPTURE_WARMUP_SEC  wall-clock seconds to wait before capturing (default 0)
FSRMAMBA_CAPTURE_START       frames to skip before capturing (default 8)
FSRMAMBA_CAPTURE_COUNT       frames to capture (default 32)
FSRMAMBA_CAMERA_ORBIT        yaw radians/frame for the auto-orbit (default 0 = off)
FSRMAMBA_UPSCALER            "fsr" (upscale) or "native" (ground truth). Forces the
                             upscaler Method + scale preset, overriding any UI/
                             persisted state. Frame generation is forced off.
FSRMAMBA_SCALE               FSR upscale ratio: 1.5 | 1.7 | 2 | 3 (default 2)
```

Fully unattended capture — no human at the keyboard — looks like:

```bash
mkdir -p /d/FSR-Mamba/captures/sponza
cd fsr-upstream/bin
FSRMAMBA_CAPTURE_DIR=D:/FSR-Mamba/captures/sponza \
FSRMAMBA_CAPTURE_WARMUP_SEC=60 \
FSRMAMBA_CAPTURE_COUNT=24 \
FSRMAMBA_CAMERA_ORBIT=0.006 \
  ./FFX_API_FSR_DX12.exe
```

Create the output dir first (the app does not). On this machine the exe must be
launched via a shell that resolves the current directory (Git Bash
`./FFX_API_FSR_DX12.exe`), not a bare name under `cmd` — `cmd` here does not
search the working directory for exes.

`FSRMAMBA_CAMERA_ORBIT` implements the framework's own commented-out
"Support camera animations" TODO in `CameraComponent::Update()`: it pans the
arc-ball camera automatically, giving the whole frame consistent motion vectors
without anyone flying it. `0.006` rad/frame is a gentle, realistic pan; larger
values move faster (keep it small enough that FSR's history reprojection stays
valid). This is why the patch also touches `framework/.../cameracomponent.cpp`.

Two guards keep the *interesting* content in frame:

- The pan **does not start until `FSRMAMBA_CAPTURE_WARMUP_SEC` has elapsed.**
  Otherwise it would spin for the whole (long) asset-streaming warmup and drift
  out of the scene — e.g. from Sponza's detailed interior out to the plain brick
  exterior — before capture even begins.
- It **reverses direction periodically** (every ~25 frames), panning back and
  forth within a bounded arc (~9° at `0.006`) around the start view instead of
  orbiting all the way around and out.

Note: the **first captured frame has zero motion** — it is the sequence's seed
frame, whose previous frame was the static warmup. This is correct; a recurrent
accumulator initialises on frame 0 with no history regardless.

## Ground truth (paired capture)

Engine capture gives FSR's *output*, which is the baseline to beat, not a
training target. The sample can also render at full display resolution with no
upscaling (`FSRMAMBA_UPSCALER=native`) — that native render is the reference the
upscaler is trying to reconstruct, i.e. the ground truth.

Capture each scene **twice** with the *same deterministic camera* (fixed
`WARMUP_SEC` + frame-counted pan), so frame *i* is the same viewpoint in both:

```bash
# FSR baseline pass (low-res input + motion + depth + FSR output)
FSRMAMBA_UPSCALER=fsr    FSRMAMBA_SCALE=2 ... ./FFX_API_FSR_DX12.exe   # -> sponza_fsr/
# Ground-truth pass (native full-res render)
FSRMAMBA_UPSCALER=native               ... ./FFX_API_FSR_DX12.exe      # -> sponza_gt/
```

Then `frame_i` of `sponza_fsr` (input + FSR's answer) pairs with `frame_i` of
`sponza_gt` (the correct answer). Training target = GT; scoring = FSR-baseline
vs learned-model, both against GT.

Sanity check on Sponza (2x, tonemapped): FSR output vs native GT is **27.7 dB /
0.913 SSIM**, per-frame PSNR range 26.9–28.3. The tight range confirms the two
passes are frame-aligned; 27.7 dB is the baseline the learned accumulator must
beat.

Both `FSRMAMBA_UPSCALER` and `FSRMAMBA_SCALE` **force** the Method and scale
preset at startup, overriding any UI or persisted state — the sample otherwise
remembers a manual Method change and can silently leave the "FSR" pass at 1.0x
(no upscaling). Frame generation is forced off during capture: it inserts
*interpolated* frames that are not true renders and have no ground truth.

## Capturing the whole dataset

`capture_scenes.py` runs both passes across every scene that ships with a camera,
into `<out>/<scene>/{fsr,gt}/`:

```bash
..\.venv\Scripts\python capture_scenes.py --out D:/FSR-Mamba/captures
..\.venv\Scripts\python capture_scenes.py --scenes sponza bistro   # subset
```

Per scene it rewrites `fsrapiconfig.json` to load that scene's glTF and its
artist-placed camera (extracted from the glTF), strips the Sponza-positioned
particle spawners, runs the FSR and GT passes, and restores the original config
at the end. It also raises `DynamicBufferPoolSize` in `cauldronconfig.json` (see
below) and renders at **1920x1080** (`-resolution`, matching the project scope:
1080p output / 540p render at 2x).

Result across 10 scenes (Sponza, Bistro, Brutalism, Chess, Hangar,
HybridReflections, Locomotive, MiniatureTable, SpaceShip, Toyshop): all captured
completely. FSR-2x-vs-GT baseline ranges 20 dB (chess, reflective) to 42 dB
(toyshop), **mean 32.5 dB** — the number a learned accumulator must beat.

MetalRoughSpheres and the animated Toyshop are skipped: they embed no camera, and
the framework's default camera is hardcoded to a Sponza-ish position.

## Heavy scenes: raise the dynamic buffer pool, it is NOT VRAM

Complex scenes (Bistro, Brutalism) overflow the framework's **dynamic buffer
pool** -- a *fixed-size, CPU-side* ring buffer for per-draw constant buffers,
default 75 MB in `cauldronconfig.json`. Overflow fires a critical assert
("DynamicBufferPool has run out of memory. Please increase the allocation size")
and hangs the app on a modal dialog. This looks like an out-of-memory crash but
**VRAM stays low** -- it is the pool, not the GPU. Fix: raise
`Allocations.DynamicBufferPoolSize` (the orchestrator sets 512 MB). It is
system-RAM-backed upload memory, so a large value is cheap.

## Two things learned the hard way (both now have built-in fixes)

1. **Wait for the scene to load — in wall-clock time, not frames.** Capturing
   early gives a nearly-empty room: valid color but empty depth and motion. This
   is not a capture bug; the scene genuinely was not drawn yet. A *frame count*
   (`FSRMAMBA_CAPTURE_START`) is an unreliable proxy because when the disk is the
   bottleneck the app renders empty frames quickly while assets stream slowly —
   hundreds of frames elapse before the scene appears. Use
   `FSRMAMBA_CAPTURE_WARMUP_SEC` instead (≈60s on a spinning disk here). It times
   from app start and does not begin capturing until that many real seconds pass.

2. **Motion vectors need camera movement.** With a static camera they are
   legitimately zero. `FSRMAMBA_CAMERA_ORBIT` drives an automatic orbit so every
   captured frame has motion, unattended. (Flying the camera manually during the
   capture window also works, but is not reproducible.)

## How it works (for the next person editing it)

Capture runs in `FSRRenderModule::OnPreFrame()` — the one frame boundary (before
`BeginFrame`) where all the FSR resources still hold the previous frame's
coherent state and no frame command list is pending, so a synchronous readback
is race-free.

Depth and motion are *transient* GBuffer targets that read back empty at that
point, so `CaptureCopyInputs()` snapshots them into persistent textures during
`Execute()` (recorded on the frame command list, while they are valid FSR
inputs) — the same trick the sample already uses to copy color into
`m_pTempTexture`. Color LR/HR come straight from their persistent targets.

`DumpTextureRaw()` is a synchronous GPU→CPU readback adapted from
`SwapChainInternal::DumpSwapChainToFile`. Byte layout comes from
`GetCopyableFootprints` (handles every format incl. depth) rather than
`GetResourceFormatStride` (which aborts on the depth format).
