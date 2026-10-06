"""A RAM-resident bank of pre-cropped training windows.

Why this exists
---------------
The capture set lives on a 7200 RPM HDD that reads at **19 MB/s** for this access
pattern (measured, sequentially, on a 2.78 GB scene cache). The multi-path dataset
is ~70 GB, so a single pass over it costs ~an hour, and `train.py`'s original loop
re-reads a fresh random crop from *every frame of every scene* on *every epoch*.
That is ~2,000 frames of strided reads per epoch and it does not amortise, because
each epoch picks new crop locations and therefore touches new pages. Memory-mapping
the caches keeps RAM flat (the dataset no longer has to fit in 64 GB) but does
nothing for throughput -- the disk is still the wall.

The fix is to pay the read **once**. Each scene is read sequentially, a small number
of large windows are extracted, and those windows are kept in RAM for the whole run.
Training then draws its actual crops from the windows, which costs nothing.

Why *large* windows rather than more small crops
------------------------------------------------
Storing finished 256-crops would fix the throughput problem but throw away
augmentation: a bank of K crops per scene means only K distinct locations, forever.
Storing a 384-wide window instead and sub-cropping 256 out of it at use time gives
(384-256)^2 = 16k distinct locations per window at 2.25x the bytes, so crop diversity
is effectively restored while the disk is still read only once.

What is stored, and what is not
-------------------------------
- **fp16.** The consumers all cast to float32 at the crop boundary anyway.
- **mv/depth at render resolution.** `crop_sequence` upsamples them to output res,
  which quadruples them; doing that at use time on the GPU is free by comparison.
- **no `fsr_out`.** It is the FSR baseline, only ever scored on validation. Carrying
  it through training windows would add ~40% for nothing.

At the default settings (2 windows x 96 frames x 384 render crop) one scene costs
~1.0 GB, so 21 training scenes is ~21 GB -- comfortable against the ~53 GB free.
"""

from __future__ import annotations

import os
import time

import torch

from .engine_data import load_engine_scene
from .augment import augment_sequence, random_op


def build_bank(captures_dir: str, scenes: list[str], out_path: str,
               windows: int = 2, win_render: int = 384, scale: int = 2,
               frames: int | None = None, seed: int = 0) -> None:
    """Read each scene once, sequentially, and save its windows as fp16.

    ``mmap=False`` on purpose: this is the one place a full sequential read is what
    we want, and mmap would turn it into scattered page faults.
    """
    rng = torch.Generator().manual_seed(seed)
    bank: list[dict] = []
    t_start = time.time()
    for si, name in enumerate(scenes):
        t0 = time.time()
        scene = load_engine_scene(f"{captures_dir}/{name}", device="cpu",
                                  half=True, mmap=False)
        n = len(scene) if frames is None else min(frames, len(scene))
        rh, rw, _ = scene[0]["lr"].shape
        uh, uw, _ = scene[0]["gt"].shape
        # A scene captured at a different quality mode has a different render->output
        # ratio, and cropping it with this scale silently misregisters LR against GT:
        # the two crops cover different physical areas of the frame. That trains on
        # wrong pairs rather than crashing, so it is checked here and excluded.
        # (Found this way: hangar_fast was captured at 1280x684 -> 1920x1080, a
        # non-uniform 1.500x1.579, while every other scene is 960x540 -> 1920x1080.)
        if uh != rh * scale or uw != rw * scale:
            print(f"  [{si+1}/{len(scenes)}] {name}: SKIPPED -- render {rw}x{rh} -> gt "
                  f"{uw}x{uh} is {uw/rw:.3f}x{uh/rh:.3f}, not {scale}x. Re-capture it "
                  f"at the same quality mode as the rest of the set.", flush=True)
            del scene
            continue
        ch = cw = min(win_render, rh, rw)
        for w in range(windows):
            y = int(torch.randint(0, max(1, rh - ch + 1), (1,), generator=rng))
            x = int(torch.randint(0, max(1, rw - cw + 1), (1,), generator=rng))
            Y, X, CH, CW = y * scale, x * scale, ch * scale, cw * scale
            bank.append({
                "scene": name,
                "y": y, "x": x, "ch": ch, "cw": cw,
                "full_rh": rh, "full_rw": rw,
                "lr": torch.stack([f["lr"][y:y + ch, x:x + cw] for f in scene[:n]]),
                "mv": torch.stack([f["mv"][y:y + ch, x:x + cw] for f in scene[:n]]),
                "depth": torch.stack([f["depth"][y:y + ch, x:x + cw] for f in scene[:n]]),
                "gt": torch.stack([f["gt"][Y:Y + CH, X:X + CW] for f in scene[:n]]),
                # jitter is a plain (x, y) tuple in the capture, not a tensor -- keep
                # it that way so the model sees exactly what load_engine_scene hands it.
                "jitter": [f["jitter"] for f in scene[:n]],
            })
        del scene
        gb = sum(v.numel() * v.element_size() for v in bank[-1].values()
                 if torch.is_tensor(v)) / 2 ** 30
        print(f"  [{si+1}/{len(scenes)}] {name}: {n} frames, {windows} windows, "
              f"{gb*windows:.2f} GB, {time.time()-t0:.0f}s "
              f"(elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)

    total = sum(v.numel() * v.element_size() for e in bank for v in e.values()
                if torch.is_tensor(v)) / 2 ** 30
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(bank, out_path)
    print(f"bank: {len(bank)} windows, {total:.1f} GB -> {out_path} "
          f"({(time.time()-t_start)/60:.1f}m total)", flush=True)


def load_bank(path: str) -> list[dict]:
    """Load a bank. Mapped, so a second run in the same session hits page cache.

    A bank must be homogeneous in render resolution: the model's FSR front-end
    precomputes sampling grids for one render->output size, so entries captured at a
    different quality mode are not merely unusual, they are misregistered. Banks
    built before that check existed can still contain them, so they are dropped here
    too -- loudly, because losing a scene changes the training set.
    """
    bank = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sizes = {}
    for e in bank:
        sizes.setdefault((e["full_rh"], e["full_rw"]), []).append(e["scene"])
    if len(sizes) > 1:
        modal = max(sizes, key=lambda k: len(sizes[k]))
        for size, names in sizes.items():
            if size != modal:
                print(f"  crop bank: DROPPING {sorted(set(names))} -- render "
                      f"{size[1]}x{size[0]} does not match the set's {modal[1]}x{modal[0]}. "
                      f"Rebuild the bank to remove this permanently.")
        bank = [e for e in bank if (e["full_rh"], e["full_rw"]) == modal]
    return bank


def sample_from_window(entry: dict, crop_render: int, scale: int, rng,
                       edge_bias: float = 0.0, frames: int | None = None,
                       augment_rng=None):
    """Draw one training sequence out of a stored window.

    Mirrors `crop_sequence`'s contract exactly -- same keys, same float32 cast, same
    motion-vector rescale, same edge-bias semantics -- so the training loop cannot
    tell the difference.

    The motion-vector rescale is the subtle part and it is the same trap as in
    `crop_sequence`: vectors are stored as *full-frame* UV offsets, but the model
    builds its sampling grid over the crop, so the offset must be multiplied by
    (full_size / crop_size) to describe the same physical displacement. Getting this
    from the *full frame* size and not the window size matters -- an earlier version
    of this bug (a hardcoded 3840x2160) miscalibrated motion for 27 runs.

    ``edge_bias`` snaps the crop to a window edge. Only the windows that actually
    touch a true frame border can offer a genuine disocclusion edge, so the snap is
    restricted to those sides; snapping to an interior window edge would teach the
    model that ordinary content is a disocclusion.
    """
    ch = cw = crop_render
    wh, ww = entry["ch"], entry["cw"]
    if wh < ch or ww < cw:
        ch = cw = min(wh, ww)

    # A contiguous slice, not a stride: the state is recurrent, so skipping frames
    # would present motion the model will never see at inference. Random start =
    # temporal augmentation, and it lets one epoch cost less than a whole scene.
    total = entry["lr"].shape[0]
    n = total if frames is None else min(frames, total)
    t0 = rng.randint(0, total - n)

    y = rng.randint(0, wh - ch)
    x = rng.randint(0, ww - cw)
    if edge_bias > 0.0 and rng.random() < edge_bias:
        # Sides where this window is flush against the real frame border.
        sides = []
        if entry["y"] == 0:
            sides.append("top")
        if entry["y"] + wh == entry["full_rh"]:
            sides.append("bottom")
        if entry["x"] == 0:
            sides.append("left")
        if entry["x"] + ww == entry["full_rw"]:
            sides.append("right")
        if sides:
            s = rng.choice(sides)
            if s == "top":
                y = 0
            elif s == "bottom":
                y = wh - ch
            elif s == "left":
                x = 0
            else:
                x = ww - cw

    Y, X, CH, CW = y * scale, x * scale, ch * scale, cw * scale
    # Rescale against the FULL frame, not the window.
    mv_rescale = torch.tensor([entry["full_rw"] / cw, entry["full_rh"] / ch])

    from .engine_data import _upsample_to
    out = []
    for i in range(t0, t0 + n):
        mv_c = entry["mv"][i, y:y + ch, x:x + cw].float() * mv_rescale
        depth_c = entry["depth"][i, y:y + ch, x:x + cw].float()
        out.append({
            "lr": entry["lr"][i, y:y + ch, x:x + cw].float(),
            "mv": _upsample_to(mv_c, CH, CW, "bilinear"),
            "depth": _upsample_to(depth_c, CH, CW, "nearest"),
            "gt": entry["gt"][i, Y:Y + CH, X:X + CW].float(),
            "jitter": entry["jitter"][i],
        })
    return augment_sequence(out, random_op(augment_rng)) if augment_rng is not None else out
