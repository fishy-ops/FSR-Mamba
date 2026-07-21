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
FSRMAMBA_CAPTURE_DIR    output directory (also enables capture when set)
FSRMAMBA_CAPTURE_START  frames to skip before capturing (default 8)
FSRMAMBA_CAPTURE_COUNT  frames to capture (default 32)
```

Create the output dir first (the app does not), then launch the built exe with
those vars set. On this machine the exe must be launched via a shell that
resolves the current directory (Git Bash `./FFX_API_FSR_DX12.exe`), not a bare
name under `cmd` — `cmd` here does not search the working directory for exes.

## Two things learned the hard way

1. **Wait for the scene to load.** `FSRMAMBA_CAPTURE_START` must be large enough
   (a few hundred frames) that the scene has finished streaming in. Capturing
   too early gives a nearly-empty room: valid color but near-empty depth and
   motion. This is not a bug in the capture — the scene genuinely was not drawn
   yet.

2. **Motion vectors need camera movement.** With a static camera they are
   legitimately zero. Fly the camera during the capture window, or add a
   programmatic camera path for reproducible unattended capture.

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
