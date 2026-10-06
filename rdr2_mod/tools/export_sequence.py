"""Write a binary test sequence (inputs plus PyTorch reference outputs) for tests/offline_test.cpp.

Modes:
  default / --synthetic    procedural scene (prototype/fsrmamba/synth.py), no private data needed
  --engine-data DIR --scene NAME --frames N
                           real captures: DIR/NAME/frame_*.json (or DIR/NAME/fsr/), or the
                           _cache_tm1.pt written by prototype/tools/make_fake_engine.py

The reference is FusedFast.step_reference or KPNAccumulator.forward in fp32 on CPU. Inputs are stored in model space
(Reinhard x/(1+x), rounded to fp16 like the CUDA path). Layout:
  "FSMSEQ1\\0", u32 width, height, frames, metadata_length, metadata JSON, then per frame
  f64 jitter_x, f64 jitter_y, u32 first_frame, f32 lr[h][w][3], f32 mv[h][w][2], f32 depth[h][w],
  f32 reference_rgb[2h][2w][3]   (all little-endian)
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from export_weights import export, load, model_config
import torch
from fsrmamba.fused import FusedFast


def engine_frames(root, scene, frames):
    from fsrmamba.capture import load_frame
    directory = root / scene
    cache = directory / "_cache_tm1.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu")
        if len(data) < frames:
            raise ValueError(f"requested {frames} frames, cache has {len(data)}")
        for f in data[:frames]:
            yield f["lr"].float(), f["mv"].float(), f["depth"].float(), tuple(float(j) for j in f["jitter"])
        return
    paths = sorted((directory / "fsr" if (directory / "fsr").is_dir() else directory).glob("frame_*.json"))
    if len(paths) < frames:
        raise ValueError(f"requested {frames} frames, found {len(paths)} under {directory}")
    for path in paths[:frames]:
        f = load_frame(str(path))
        yield f["lr"], f["mv"].float(), f["depth"], tuple(f["jitter"])


def synthetic_frames(render, frames, seed):
    from fsrmamba.synth import halton_jitter, random_scene
    output = tuple(2 * s for s in render)
    scene = random_scene(seed, output)
    for i in range(frames):
        jitter = halton_jitter(i)
        lr = scene.render(i, render, jitter=jitter)
        aux = scene.render(i, render)
        yield lr.color.float(), aux.mv.float(), aux.depth.float(), tuple(float(j) for j in jitter)


def first_shape(source):
    items = iter(source)
    first = next(items)
    def chain():
        yield first
        yield from items
    return first[0].shape[:2], chain()


def export_sequence(checkpoint, output, frames, weights, source, render=None, reset_frames=()):
    if render is None:
        (h, w), source = first_shape(source)
    else:
        h, w = render
    model = load(checkpoint, (h, w))
    tensors = export(model, weights)
    del tensors
    kpn = model_config(model)["arch"] == "kpn"
    fused = model if kpn else FusedFast(model)
    state = fused.init_state()
    metadata = dict(config=model_config(model), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    reference=("KPNAccumulator.forward fp32 on CPU, checkpoint trunk_stride/taps/history_filter/sigma_min/proximity/proximity_gain, "
                               "fp16-rounded colour, fp32 depth" if kpn else
                               "FusedFast.step_reference fp32 on CPU, fp16-rounded colour/depth input"),
                    depth_gate=("KPN fp32: old_depth < .9*dmin or > 1.1*dmax; soft mode resets age only" if kpn else
                                "Hard osc_n > clamp(soft_osc_threshold, .05, 4); reversed-Z larger is nearer"
                                if model.depth_soft_osc else "depth_soft_osc disabled"),
                    color_space="Reinhard", motion_space="UV")
    metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    with output.open("wb") as f, torch.inference_mode():
        f.write(b"FSMSEQ1\0" + struct.pack("<IIII", w, h, frames, len(metadata)) + metadata)
        for index, (lr, mv, depth, jitter) in enumerate(source):
            lr = lr.clamp(min=0)
            lr = (lr / (1 + lr)).half().float()
            depth = depth.float() if kpn else depth.half().float()
            if tuple(lr.shape) != (h, w, 3) or tuple(mv.shape) != (h, w, 2) or tuple(depth.shape) != (h, w):
                raise ValueError("sequence changes resolution")
            if not all(torch.isfinite(t).all() for t in (lr, mv, depth)):
                raise ValueError("nonfinite input")
            first = index == 0 or index in reset_frames
            if first:
                state = fused.init_state()
            rgb, state = (fused(state, lr, mv, depth, jitter) if kpn else
                          fused.step_reference(state, lr, mv, depth, jitter))
            f.write(struct.pack("<ddI", float(jitter[0]), float(jitter[1]), int(first)))
            for t in (lr, mv, depth, rgb):
                f.write(t.contiguous().numpy().astype("<f4", copy=False).tobytes())
            print(f"Exported frame {index + 1}/{frames}", flush=True)
            if index + 1 == frames:
                break
    print(f"Wrote {output} ({h}x{w} -> {2 * h}x{2 * w}) and weights {weights}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--weights", type=Path, help="weights file to write (default: OUTPUT with .weights.bin)")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--reset-frame", type=int, action="append", default=[],
                   help="Zero-based frame at which to reset temporal state (repeatable)")
    p.add_argument("--synthetic", action="store_true", help="procedural scene (default)")
    p.add_argument("--render", default="64x96", help="synthetic render size HxW (odd sizes exercise padding)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine-data", type=Path)
    p.add_argument("--scene")
    a = p.parse_args()
    if a.frames < 1:
        p.error("--frames must be positive")
    if any(i < 0 or i >= a.frames for i in a.reset_frame):
        p.error("--reset-frame must be within the exported sequence")
    weights = a.weights or a.output.with_suffix(".weights.bin")
    if a.engine_data:
        if not a.scene or a.synthetic:
            p.error("--engine-data needs --scene and excludes --synthetic")
        export_sequence(a.checkpoint, a.output, a.frames, weights, engine_frames(a.engine_data, a.scene, a.frames), reset_frames=a.reset_frame)
    else:
        try:
            render = tuple(int(x) for x in a.render.lower().split("x"))
            assert len(render) == 2 and min(render) >= 8
        except (ValueError, AssertionError):
            p.error("--render must be HxW with both >= 8")
        export_sequence(a.checkpoint, a.output, a.frames, weights, synthetic_frames(render, a.frames, a.seed), render, reset_frames=a.reset_frame)


if __name__ == "__main__":
    main()
