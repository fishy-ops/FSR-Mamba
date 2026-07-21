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
