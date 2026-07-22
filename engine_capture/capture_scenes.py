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


def run_pass(mode: str, out_dir: str) -> int:
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
    proc = subprocess.Popen([EXE, "-resolution", str(RES_W), str(RES_H)],
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
    ap.add_argument("--out", default="D:/FSR-Mamba/captures")
    ap.add_argument("--scenes", nargs="*", help="subset of scene names (default: all)")
    args = ap.parse_args()

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
            nf = run_pass("fsr", os.path.join(args.out, name, "fsr"))
            ng = run_pass("native", os.path.join(args.out, name, "gt"))
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
