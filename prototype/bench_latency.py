"""Synchronised recurrent-step latency, module timings, and optional ONNX export."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import time

import torch
from torch import nn

from fsrmamba.config import add_arch_args, build_model, load_checkpoint, new_state
from fsrmamba.fast import FastAccumulator
from fsrmamba.mamba import MambaState
from fsrmamba.synth import halton_jitter
from tools.onnx_fast import export_fast, time_ort


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clock_read(device):
    synchronise(device)
    return time.perf_counter()


def amp_context(device, fp16):
    return torch.autocast(device.type, dtype=torch.float16) if fp16 else nullcontext()


def initial_state(model, device, channels_last):
    state = new_state(model, device)
    if channels_last and hasattr(state, "hidden"):
        state.hidden = state.hidden[None].contiguous(memory_format=torch.channels_last)[0]
    return state


def smooth_motion(size, render, device):
    """Subpixel translation plus a smooth field, in normalised frame UV."""
    h, w = size
    ys = torch.linspace(0, 2 * torch.pi, h, device=device)
    xs = torch.linspace(0, 2 * torch.pi, w, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack(((0.6 + 0.2 * torch.sin(y) * torch.cos(x)) / render[1],
                        (-0.3 + 0.15 * torch.cos(y) * torch.sin(x)) / render[0]), dim=-1)


def make_inputs(model, device):
    render, output = model.render_size, model.output_size
    aux = render if isinstance(model, FastAccumulator) else output
    return (torch.rand((*render, 3), device=device), smooth_motion(aux, render, device),
            1 + torch.rand(aux, device=device))


@torch.no_grad()
def run_pass(model, inputs, device, frames, warmup, fp16=False, channels_last=False,
             install=None, mode="full"):
    state = initial_state(model, device, channels_last)
    lr, mv, depth = inputs
    prev_depth = None
    fast = isinstance(model, FastAccumulator)
    if mode != "full" and not fast:
        raise ValueError("Separated timings require the fast architecture")
    mv_hr = None
    if fast and mode != "full":
        mv_hr = torch.nn.functional.interpolate(mv.permute(2, 0, 1)[None],
            size=model.output_size, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    times = []
    cleanup = None
    with amp_context(device, fp16):
        # Populate the recurrent state before measuring reprojection in isolation.
        if mode == "reproject":
            _, state = model(state, lr, mv, depth, halton_jitter(0))
        def step(i):
            nonlocal state, prev_depth
            hist = model.reproject(state, mv_hr) if mode == "network" else None
            jitter = halton_jitter(i)
            start = clock_read(device)
            if mode == "reproject":
                model.reproject(state, mv_hr)
            else:
                kwargs = {"hist": hist} if mode == "network" else {}
                _, state = model(state, lr, mv, depth, jitter,
                                 prev_depth_hr=prev_depth, **kwargs)
                prev_depth = depth
            return (clock_read(device) - start) * 1000

        for i in range(warmup):
            step(i)
        if install is not None:
            cleanup = install()
        try:
            for i in range(frames):
                times.append(step(i + warmup))
        finally:
            if cleanup is not None:
                cleanup()
    return times


def report_times(label, times):
    values = torch.tensor(times, dtype=torch.float64)
    mean = values.mean().item()
    median = values.quantile(0.5).item()
    p95 = values.quantile(0.95).item()
    print(f"{label}: mean {mean:.3f} ms, median {median:.3f} ms, "
          f"p95 {p95:.3f} ms; {1000 / mean:.2f} frames/s")
    return median


def timing_hooks(model, device, totals):
    handles, methods, starts = [], [], defaultdict(list)

    def pre(name):
        def hook(module, args):
            starts[name].append(clock_read(device))
        return hook

    def post(name):
        def hook(module, args, output):
            totals[name] += (clock_read(device) - starts[name].pop()) * 1000
        return hook

    if isinstance(model, FastAccumulator):
        modules = [(name, getattr(model, name)) for name in ("stem", "fuse", "out", "film")]
        for name in ("enc", "down", "up"):
            modules.extend((f"{name}[{i}]", module) for i, module in enumerate(getattr(model, name)))
    else:
        modules = [(name, getattr(model, name, None)) for name in
                   ("encoder", "encoder_unet", "lr_up", "kpn_head", "swin_refine",
                    "resolve_refine", "sharpen_net")]
        for name in ("alpha_heads", "decoders"):
            modules.extend((f"{name}[{i}]", module) for i, module in
                           enumerate(getattr(model, name, ())))
    for name, module in modules:
        if module is not None:
            handles.extend((module.register_forward_pre_hook(pre(name)),
                            module.register_forward_hook(post(name))))
            totals[name] = 0.0

    def wrap(owner, attr, name):
        original = getattr(owner, attr)
        methods.append((owner, attr, original))
        totals[name] = 0.0

        def timed(*args, **kwargs):
            start = clock_read(device)
            result = original(*args, **kwargs)
            totals[name] += (clock_read(device) - start) * 1000
            return result
        setattr(owner, attr, timed)

    if isinstance(model, FastAccumulator):
        wrap(model, "reproject", "reproject")
    else:
        wrap(model._resolve, "_upsample", "_resolve._upsample")
        wrap(model, "_warp", "_warp")

    def cleanup():
        for handle in handles:
            handle.remove()
        for owner, attr, original in methods:
            setattr(owner, attr, original)
    return cleanup


class RecurrentStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden, lr, mv, depth, jitter, prev_depth):
        rgb, state = self.model(MambaState(hidden, frame_index=1), lr, mv, depth, jitter,
                                prev_depth_hr=prev_depth)
        return rgb, state.hidden


def export_onnx(model, inputs, path, device):
    lr, mv, depth = inputs
    hidden = new_state(model, device).hidden
    jitter = torch.tensor(halton_jitter(1), device=device)
    try:
        with torch.no_grad():
            torch.onnx.export(RecurrentStep(model), (hidden, lr, mv, depth, jitter, depth),
                              path, input_names=["hidden", "lr", "mv", "depth", "jitter", "prev_depth"],
                              output_names=["rgb", "new_hidden"], opset_version=18, dynamo=False)
        print(f"ONNX exported to {path}")
    except Exception as exc:
        print(f"ONNX export failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt")
    add_arch_args(ap)
    arch_defaults = vars(ap.parse_args([]))
    arch_defaults.pop("ckpt")
    ap.add_argument("--render", default="540x960", help="Render height x width.")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--onnx")
    ap.add_argument("--ort", action="store_true", help="Time the exported fast graph on GPU.")
    args = ap.parse_args()
    try:
        render = tuple(int(x) for x in args.render.lower().split("x"))
        if len(render) != 2 or min(*render, args.scale, args.frames) < 1 or args.warmup < 0:
            raise ValueError
    except ValueError:
        ap.error("render must be HxW, sizes/frames positive, and warmup nonnegative")
    if args.ckpt and any(getattr(args, key) != default for key, default in arch_defaults.items()):
        ap.error("Use either --ckpt or architecture flags")
    output = tuple(size * args.scale for size in render)
    device = torch.device(args.device)
    model = (load_checkpoint(args.ckpt, render, output, device)[0] if args.ckpt
             else build_model(args, render, output, device))
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    fp16 = args.fp16 and device.type != "cpu"
    if args.fp16 and device.type == "cpu":
        print("FP16 requested on CPU; running fp32.")
    if args.ort and not args.onnx:
        ap.error("--ort requires --onnx OUT.onnx")
    fast = isinstance(model, FastAccumulator)
    if args.ort and not fast:
        ap.error("--ort requires the fast architecture")
    inputs = make_inputs(model, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    label = f"Latency ({device.type}, {'fp16' if fp16 else 'fp32'})"
    times = run_pass(model, inputs, device, args.frames, args.warmup, fp16, args.channels_last)
    report_times(label + (" full step" if fast else ""), times)
    if fast:
        for mode, name in (("network", "network + blend"), ("reproject", "reprojection alone")):
            times = run_pass(model, inputs, device, args.frames, args.warmup, fp16,
                             args.channels_last, mode=mode)
            report_times(label + " " + name, times)
    totals = defaultdict(float)
    hooked = run_pass(model, inputs, device, args.frames, args.warmup, fp16, args.channels_last,
                      lambda: timing_hooks(model, device, totals))
    whole = sum(hooked) / args.frames
    modules = {name: total / args.frames for name, total in totals.items()}
    modules["other"] = whole - sum(modules.values())
    summed = sum(modules.values())
    print("\nPer-module breakdown (separate pass; includes synchronisation overhead)")
    print("module                           ms/frame   % of total")
    for name, ms in sorted(modules.items(), key=lambda pair: pair[1], reverse=True):
        print(f"{name:30s} {ms:10.3f} {100 * ms / summed:11.2f}%")
    print(f"Whole forward in breakdown pass: {whole:.3f} ms")
    if args.onnx:
        if fast:
            exported = export_fast(model, inputs, args.onnx, device, fp16)
            if args.ort:
                time_ort(args.onnx, exported, model, device, args.frames, args.warmup)
        else:
            export_onnx(model, inputs, args.onnx, device)


if __name__ == "__main__":
    main()
