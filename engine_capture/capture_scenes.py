"""Capture paired FSR + ground-truth training data across every scene.

For each scene this rewrites the sample's runtime config to load that scene's
glTF and its artist-placed camera, then runs the instrumented FSR sample twice
with the deterministic capture camera:

    <out>/<scene>/fsr/   FSR 2x upscale -> low-res input + motion + depth + FSR output
    <out>/<scene>/gt/    native full-res render -> ground truth

Frame i aligns across the two passes (same warmup + frame-counted pan), so
<scene>/fsr/frame_i pairs with <scene>/gt/frame_i for supervised training.

Run from anywhere with the venv python, e.g.:
    ..\\.venv\\Scripts\\python capture_scenes.py --out D:/FSR-Mamba/captures
    ..\\.venv\\Scripts\\python capture_scenes.py --scenes sponza bistro   # subset

Scenes with no embedded camera (MetalRoughSpheres, the animated Toyshop) are
omitted -- the framework's default camera is hardcoded to a Sponza-ish position
and would not frame them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

# repo/fsr-upstream regardless of where this is run from
FSR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fsr-upstream"))
BIN = os.path.join(FSR_ROOT, "bin")
EXE = os.path.join(BIN, "FFX_API_FSR_DX12.exe")
CONFIG = os.path.join(BIN, "configs", "fsrapiconfig.json")
CAULDRON_CONFIG = os.path.join(BIN, "configs", "cauldronconfig.json")
CAULDRON_LOG = os.path.join(BIN, "Cauldron.log")

# The default 75 MB dynamic buffer pool overflows on heavy scenes (Bistro,
# Brutalism) -- a fixed-size CPU-side pool, unrelated to VRAM, whose overflow
# fires a critical assert and hangs the app. 512 MB holds every scene here.
DYNAMIC_BUFFER_POOL = 512 * 1024 * 1024

# (short name, glTF path relative to fsr-upstream, camera node name)
SCENES = [
    ("sponza",     "media/SponzaNew/MainSponza.gltf",                              "PhysCamera003"),
    ("bistro",     "media/BistroInterior/BistroInrterior.gltf",                    "Camera_1"),
    ("brutalism",  "media/Brutalism/BrutalistHall.gltf",                           "persp1_Orientation"),
    ("chess",      "media/Chess/scene.gltf",                                       "Camera_1"),
    ("hangar",     "media/Hangar/Hangar_1105.gltf",                                "Hangar_Camera_000"),
    ("hybridrefl", "media/HybridReflections/scene.gltf",                           "Camera"),
    ("locomotive", "media/Locomotive/Locomotive.gltf",                             "Camera_1"),
    ("table",      "media/MiniatureTable/Table.gltf",                              "Camera_2"),
    ("spaceship",  "media/SpaceShipPlatform/Ship.gltf",                            "Camera_001"),
    ("toyshop",    "media/Toyshop_TeddyGI/Toyshop_Teddy_static/Toyshop_Teddy_static.gltf", "Camera2"),
]

WARMUP_SEC = 60
COUNT = 24
ORBIT = 0.006
SCALE = 2
PASS_TIMEOUT = 300  # seconds to wait for one pass before giving up
# 1080p, not the window's native 2560x1440: it matches the project's 1080p scope
# and, crucially, the native/GT pass renders the full frame with no upscaling --
# at 1440p a large scene exhausts the 2070 Super's 8 GB VRAM and crashes. 1080p
# is ~44% fewer pixels and fits.
RES_W, RES_H = 1920, 1080
GT_RES = None
_RESTORE_FULLSCREEN = []


def write_config(base_cfg: dict, gltf: str, camera: str) -> None:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    content = cfg["FidelityFX FSR FFXAPI"]["Content"]
    content["Scenes"] = ["../" + gltf]
    content["Camera"] = camera
    content.pop("ParticleSpawners", None)  # Sponza-positioned; drop elsewhere
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=4)


def _kill_stale() -> None:
    """Make sure no previous instance is still holding the GPU/swapchain."""
    subprocess.run(["taskkill", "/F", "/IM", "FFX_API_FSR_DX12.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)  # let the GPU actually release the killed process's memory


def run_pass(mode: str, out_dir: str, res=None) -> int:
    """Launch one capture pass, wait for completion, return frame count."""
    _kill_stale()
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    if os.path.exists(CAULDRON_LOG):
        os.remove(CAULDRON_LOG)

    env = dict(os.environ)
    env.update({
        "FSRMAMBA_CAPTURE_DIR": out_dir.replace("\\", "/"),
        "FSRMAMBA_CAPTURE_WARMUP_SEC": str(WARMUP_SEC),
        "FSRMAMBA_CAPTURE_START": "0",
        "FSRMAMBA_CAPTURE_COUNT": str(COUNT),
        "FSRMAMBA_CAMERA_ORBIT": str(ORBIT),
        "FSRMAMBA_UPSCALER": mode,
        "FSRMAMBA_SCALE": str(SCALE),
    })
    # Launch by absolute path with cwd=bin (so relative media/config paths
    # resolve). subprocess runs the exe directly, avoiding cmd's cwd-search quirk.
    rw, rh = res if res else (RES_W, RES_H)
    proc = subprocess.Popen([EXE, "-resolution", str(rw), str(rh)],
                            cwd=BIN, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + WARMUP_SEC + PASS_TIMEOUT
    done = False
    while time.time() < deadline:
        if proc.poll() is not None:  # crashed / exited on its own
            break
        n = len([f for f in os.listdir(out_dir) if f.endswith(".json")])
        if n >= COUNT or _log_says_complete():
            done = True
            break
        time.sleep(3)
    # Give the last file writes a moment, then stop the app.
    time.sleep(1)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    n = len([f for f in os.listdir(out_dir) if f.endswith(".json")])
    return n


def _log_says_complete() -> bool:
    try:
        with open(CAULDRON_LOG, "r", errors="ignore") as f:
            return "capture complete" in f.read().lower()
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    global COUNT, ORBIT, RES_W, RES_H, GT_RES
    ap.add_argument("--out", default="D:/FSR-Mamba/captures")
    ap.add_argument("--scenes", nargs="*", help="subset of scene names (default: all)")
    ap.add_argument("--orbit", type=float, default=ORBIT,
                    help="camera yaw radians/frame. Varying this is how we get multiple "
                         "camera paths per scene -- different motion magnitude exercises "
                         "the velocity-dependent behaviour and changes disocclusion rate, "
                         "which is the diversity axis that matters for a temporal upscaler.")
    ap.add_argument("--suffix", type=str, default="",
                    help="appended to the output scene name, so several paths through the "
                         "same scene land in separate directories.")
    ap.add_argument("--res", type=str, default="",
                    help="render resolution as WxH, e.g. 2560x1440. Default 1920x1080. "
                         "Higher is worth trying because the GT pass is a 1-SAMPLE-PER-PIXEL "
                         "native render, i.e. an aliased supervision target -- production "
                         "upscalers train against supersampled references. Capturing GT at "
                         "1440p and downsampling to 1080p yields ~1.78 samples/pixel. The "
                         "risk is VRAM: the native pass renders the full frame with no "
                         "upscaling and 1440p was previously observed to exhaust 8 GB on "
                         "large scenes.")
    ap.add_argument("--fullscreen", action="store_true",
                    help="run the sample fullscreen so the requested render height is not clamped by the window client area. Required for an exact 2560x1440 GT pass.")
    ap.add_argument("--gt-res", type=str, default="",
                    help="render the NATIVE/GT pass at this WxH instead of --res. Set it to an exact integer multiple of --res (same aspect ratio) and the GT can be downsampled to the FSR pass size to give a genuinely supersampled, anti-aliased target. The default 1-sample-per-pixel GT is itself aliased, which caps what any model trained on it can learn. Aspect MUST match or the two passes frame the scene differently and the pairs are silently misaligned.")
    ap.add_argument("--count", type=int, default=COUNT,
                    help="frames to capture per pass (default 24). Longer sequences give "
                         "deeper temporal accumulation and longer comparison clips, but cost "
                         "~92 MB/frame across both passes -- watch disk.")
    args = ap.parse_args()
    COUNT = args.count
    ORBIT = args.orbit
    if args.res:
        RES_W, RES_H = (int(v) for v in args.res.lower().split("x"))
        print(f"render resolution overridden -> {RES_W}x{RES_H}")
    if args.gt_res:
        GT_RES = tuple(int(v) for v in args.gt_res.lower().split("x"))
        ar_f, ar_g = RES_W / RES_H, GT_RES[0] / GT_RES[1]
        if abs(ar_f - ar_g) > 1e-3:
            raise SystemExit(f"aspect mismatch: fsr pass {ar_f:.4f} vs gt pass "
                             f"{ar_g:.4f}. The passes would frame the scene "
                             f"differently and every pair would be misaligned.")
        print(f"GT pass renders at {GT_RES[0]}x{GT_RES[1]} "
              f"({GT_RES[0]/RES_W:.2f}x linear = {(GT_RES[0]/RES_W)**2:.1f}x samples/pixel)")
    print(f"capturing {COUNT} frames per pass")

    names = {s[0] for s in SCENES}
    todo = SCENES if not args.scenes else [s for s in SCENES if s[0] in set(args.scenes)]
    if args.scenes:
        missing = set(args.scenes) - names
        if missing:
            raise SystemExit(f"unknown scenes: {missing}. Known: {sorted(names)}")

    # Ensure the dynamic buffer pool is large enough for the heaviest scenes.
    with open(CAULDRON_CONFIG) as f:
        ccfg = json.load(f)
    root = next(iter(ccfg.values()))
    if args.fullscreen:
        # Windowed mode silently clamps the render height to the client area -- a
        # requested 2560x1440 comes back as 2560x1421, which changes the ASPECT RATIO
        # (1.8015 vs 1.7778) and therefore the framing. Two passes at different aspects
        # frame the scene differently, so their frames do not correspond and every
        # training pair built from them is misaligned. Measured: a 1440p GT downsampled
        # to 1080p scores only 24.6 dB against the 1080p GT of the same frame.
        # Fullscreen removes the clamp, so 1920x1080 and 2560x1440 share an exact 1.7778
        # aspect and the GT pass can be a true supersample of the FSR pass.
        pres = root.setdefault("Presentation", {})
        pres["Fullscreen"] = True
        # cauldronconfig.json is PERSISTENT and the restore path below only ever
        # covered the scene config, so a single --fullscreen run silently left the
        # sample in fullscreen for every later capture -- which forces both passes to
        # the display resolution and defeats --gt-res without any visible error. Ten
        # minutes of captures were produced at ratio 1.0 that way. Registered for
        # restore in the finally block.
        _RESTORE_FULLSCREEN.append(True)
        with open(CAULDRON_CONFIG, "w") as f:
            json.dump(ccfg, f, indent=4)
        print("fullscreen enabled (removes the windowed height clamp)")
    alloc = root.setdefault("Allocations", {})
    if alloc.get("DynamicBufferPoolSize", 0) < DYNAMIC_BUFFER_POOL:
        alloc["DynamicBufferPoolSize"] = DYNAMIC_BUFFER_POOL
        with open(CAULDRON_CONFIG, "w") as f:
            json.dump(ccfg, f, indent=4)
        print(f"raised DynamicBufferPoolSize -> {DYNAMIC_BUFFER_POOL // (1024*1024)} MB\n")

    with open(CONFIG) as f:
        base_cfg = json.load(f)
    backup = CONFIG + ".orig"
    if not os.path.exists(backup):
        shutil.copy(CONFIG, backup)

    print(f"capturing {len(todo)} scenes -> {args.out}\n")
    results = []
    try:
        for i, (name, gltf, cam) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {name}  ({cam})")
            write_config(base_cfg, gltf, cam)
            # --suffix lands several camera paths through one scene in separate
            # directories (e.g. hangar, hangar_slow, hangar_fast).
            out_name = name + args.suffix
            # A decoded cache sits BESIDE fsr/ and gt/, and load_engine_scene prefers
            # it over the raw frames. run_pass only clears the pass directories, so a
            # cache left from an earlier capture of this scene silently wins -- a
            # 6-frame test cache made a freshly captured 48-frame scene load as 6.
            # Drop it here so the next load re-decodes what was actually captured.
            for _stale in glob.glob(os.path.join(args.out, out_name, "_cache_*.pt")):
                os.remove(_stale)
                print(f"      removed stale cache {os.path.basename(_stale)}")
            nf = run_pass("fsr", os.path.join(args.out, out_name, "fsr"))
            ng = run_pass("native", os.path.join(args.out, out_name, "gt"),
                          res=GT_RES)
            ok = nf >= COUNT and ng >= COUNT
            print(f"      fsr={nf} gt={ng}  {'OK' if ok else 'INCOMPLETE'}")
            results.append((name, nf, ng, ok))
    finally:
        shutil.copy(backup, CONFIG)  # restore Sponza default config
        print("\nrestored original config")

    print("\n=== summary ===")
    for name, nf, ng, ok in results:
        print(f"  {name:12s} fsr={nf:2d} gt={ng:2d}  {'OK' if ok else 'INCOMPLETE'}")
    good = sum(1 for _, _, _, ok in results if ok)
    print(f"\n{good}/{len(results)} scenes captured completely")


if __name__ == "__main__":
    main()
