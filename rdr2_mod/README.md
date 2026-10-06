# Game runtime: Red Dead Redemption 2 (DX12)

A drop-in replacement for the game's `ffx_fsr2_api_x64.dll` (AMD FSR 2.2.1 API library). It
forwards all eleven exported FSR 2 functions to the original library, renamed
`ffx_fsr2_api_x64.orig.dll`. The exception is DX12 at exactly 2x upscaling, where it runs the
learned upscaler instead:

    HLSL pack compute shader -> U-Net on DirectML -> HLSL resolve compute shader

Any runtime error is logged once and the DLL forwards to the original FSR 2 for the rest of the
session. Every other quality mode, every other upscaler and Vulkan pass straight through.

Story mode only. Do not use a modified DLL online.

## Requirements

- Windows 10/11, a D3D12 GPU with fp16 support, DX12 selected in the game's graphics options.
- FSR 2 enabled at quality mode **Performance** (exactly 2x per axis). If the game's upscaler
  setting is DLSS or anything else, the mod does not run.
- The stock `ffx_fsr2_api_x64.dll` must be the known 2.2.1 build (SHA-256
  `887d11aaef717aa2d4713f4b6fb56b3d4c75737af51042de22f70f0a3fb89f83`). With another build the mod only
  forwards, unless `require_known_fsr2=0` is set.

## Build

Not in the repository: the unpacked `Microsoft.AI.DirectML` NuGet package (`DML_ROOT`), `dxc.exe`
(`DXC`), and Python 3 (embeds the compiled shaders).

MSVC (x64 Native Tools prompt):

    cmake -S rdr2_mod -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDML_ROOT=C:\path\to\Microsoft.AI.DirectML -DDXC=C:\path\to\dxc.exe
    cmake --build build

Cross-compile from Linux or macOS with llvm-mingw:

    cmake -S rdr2_mod -B rdr2_mod/build-cross -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_MINGW_ROOT=/path/to/llvm-mingw \
      -DCMAKE_TOOLCHAIN_FILE="$PWD/rdr2_mod/cmake/mingw-w64-x86_64.cmake" \
      -DDML_ROOT=/path/to/Microsoft.AI.DirectML -DDXC=/path/to/dxc -DPYTHON=python3
    cmake --build rdr2_mod/build-cross -j8

Outputs: `ffx_fsr2_api_x64.dll`, `offline_test.exe` (with `DirectML.dll` beside it),
`cpu_test.exe`, `abi_layout.exe`.

## Install and test

1. Export weights and a reference sequence (CPU PyTorch is enough):

        python tools/export_sequence.py ckpt/MODEL.pt seq.bin --frames 8 --weights fsrmamba_weights.bin

   `tools/export_weights.py CKPT OUT` exports weights alone. Each checkpoint needs its `.pt.json` sidecar.
2. Verify the GPU build against the PyTorch reference:

        offline_test.exe fsrmamba_weights.bin seq.bin --tol 0.003

   It prints per-frame error and GPU milliseconds for pack, network and resolve, and exits 0 on pass,
   2 above tolerance, 1 on a setup error. fp16 drift of 1e-3 to 1e-2 with correct structure is expected.
   For a garbled 2x2 block pattern try `--d2s-alt`; `--debug-view 1|2` show the base or reprojected
   history only.
3. In the game folder, rename the original DLL to `ffx_fsr2_api_x64.orig.dll`, then copy in
   `ffx_fsr2_api_x64.dll`, `fsrmamba.ini`, `fsrmamba_weights.bin` and `DirectML.dll`.
4. Start the game in DX12 with FSR 2 on Performance. `fsrmamba.log` records the original DLL hash,
   DirectML load, context creation and `Learned 2x dispatch active`.
5. `]` toggles between the learned upscaler and real FSR 2 (`toggle_key`); `[` cycles presets
   (`preset_key`). History restarts at every switch.
6. To uninstall, delete the added files and rename the original DLL back.

## Configuration (`fsrmamba.ini`)

| Key | Meaning |
|---|---|
| `enable`, `weights`, `toggle_key`, `log` | Master switch, weight file beside the DLL, A/B key, logging |
| `exposure_mode` | `fsr` scales by exposure/preExposure like FSR 2; `one` applies none |
| `ring_frames` | Upload ring size (default 8) |
| `jitter_flip_x/y`, `mv_flip_x/y`, `depth_to_space_alt` | Convention switches; try one at a time if the picture shimmers or ghosts |
| `debug_view` | 0 normal, 1 base only, 2 reprojected history only |
| `stabilize`, `stabilize_tau`, `stabilize_eps` | Live anti-flicker filter (below) |
| `foliage_*`, `fallback_*` | Experimental instability-gated filtering, default off |
| `preset_key`, `preset1`..`preset6` | Cycle named sets of the live controls |

Live controls reload every 120 dispatches without rebuilding the pipeline, and each effective change
is logged once. Values are clamped to their ranges.

**Stabiliser.** An fp32 post-filter in model space that blends small changes toward clamped reprojected
history relative to the nearest sample's 3x3 contrast; larger changes pass through. `stabilize` is
0 to 0.95 (0 bypasses all filter arithmetic and preserves offline parity), `stabilize_tau` is
0.05 to 4, `stabilize_eps` is 0.0001 to 0.1. Reset, first-frame and disoccluded pixels bypass it.
Higher strength softens moving detail. It works with every model configuration without retraining.

**Presets.** Comma-separated `key=value` lists over the live controls, applied on top of the ini values.
Pressing `preset_key` advances and wraps, logs the applied values and beeps N times for preset N.
Unknown keys or bad values reject the whole preset. Example:

    preset_key=0xDB
    preset1=stabilize=0,foliage_strength=0
    preset2=foliage_strength=1
    preset3=stabilize=0.85,stabilize_tau=1.0

## Model formats

`tools/export_weights.py` picks the architecture from the checkpoint and its sidecar and writes an
`FSMWGT1` container with strictly validated tensor names, shapes and dtypes; convolution weights are fp16.
The KPN export carries `trunk_stride` (1 or 2), `taps` (5 or 3), `history_filter` (`bicubic` or `catmull`),
`sigma_min`, `proximity` and `proximity_gain`. Stride 2 packs the render image into 64 half-resolution
channels and removes the depth-to-space pass. Changing stride, taps or the history filter needs
retraining. Architectures the runtime cannot express are rejected by both the exporter and the DLL.

## Tests

From `prototype/`, with a native fixture directory:

    clang++ -std=c++17 -I../rdr2_mod/src ../rdr2_mod/tests/cpu_test.cpp ../rdr2_mod/src/cpu_math.cpp ../rdr2_mod/src/weights.cpp -o "$CPU_TEST_BINARY"
    python ../rdr2_mod/tests/test_cpu.py "$CPU_TEST_BINARY" --temp-dir "$CPU_TEST_TMP"
    python ../rdr2_mod/tests/test_kpn.py

These cover the frame maths, weight round trips for all supported configurations, KPN pack and resolve
references against PyTorch, preset parsing and cycling, and the stabiliser's bypass and clamping rules.
GPU parity, latency and visual quality need Windows and the offline commands above.

## Frame dumps and offline analysis

Set `dump_frames=N`, `dump_dir` and `dump_every=K` under `[fsrmamba]` to capture N dispatches to disk while
forwarding to the original FSR 2. `tools/analyze_dump.py` converts a dump to a training cache,
`tools/accumulate_gt.py --frozen` builds a high-quality target from a frozen photo-mode shot (64-frame
accumulation), `tools/hires_to_cache.py` pairs high-resolution captures, and `tools/eval_cache.py` scores a
model on a cache including a flicker metric. A dump takes about 2.6 GB per shot, so the mod refuses to
start one below 25 GB of free space.

## Known limitations

- 2x only; any other render scale is FSR 2.
- DirectML trunk cost has a floor of about 1.8 ms; the full pipeline is about 4 ms at 720p to 1440p on an RTX 2070 SUPER.
- Auto exposure is not modelled beyond the game-supplied exposure value.
- Foliage can shimmer; the stabiliser reduces it at the cost of some softness in motion.
- The output texture must be UAV-capable (for example `R16G16B16A16_FLOAT`).
