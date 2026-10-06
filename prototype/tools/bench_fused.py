"""CUDA-only fused fast accumulator and autocast forward latency."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_latency import make_inputs
from fsrmamba.config import add_arch_args, build_model, load_checkpoint
from fsrmamba.fused import FusedFast, _fold_head, _run_enc, state_rgb


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path)
    add_arch_args(ap)
    ap.set_defaults(arch="fast", fast_state=0)
    ap.add_argument("--render", default="540x960", help="Render height x width")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=1, help="Report medians over N benchmark repeats")
    ap.add_argument("--compile-trunk", action="store_true", help="Also time an optional compiled trunk")
    ap.add_argument("--write-rgb", action="store_true",
                    help="Include the eager display write in resolve/total (always enabled for carry_raw)")
    ap.add_argument("--fold", choices=("host", "gpu"), default="host", help="Per-frame head folding")
    ap.add_argument("--layout", choices=("nhwc", "nchw"), default="nhwc")
    ap.add_argument("--trunk", choices=("cudnn", "tensorrt"), default="cudnn")
    ap.add_argument("--all-backends", action="store_true",
                    help="Run cudnn/nhwc, cudnn/nchw and tensorrt/nchw")
    ap.add_argument("--profile-kernels", action="store_true",
                    help="Time only pack and resolve with each optional feature toggled (both layouts, "
                         "cuDNN trunk); the display is always written")
    return ap


def measure(fn, device, n, warmup, prepare=None):
    values = []
    for i in range(n + warmup):
        if prepare is not None:
            prepare()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        if i >= warmup:
            values.append((time.perf_counter() - start) * 1000)
    t = torch.tensor(values, dtype=torch.float64)
    return dict(median_ms=t.quantile(0.5).item(), p95_ms=t.quantile(0.95).item())


def median_repeats(runs):
    return {name: {key: torch.tensor([run[name][key] for run in runs], dtype=torch.float64)
                  .quantile(0.5).item() for key in runs[0][name]}
            for name in runs[0]}


@torch.no_grad()
def layer_breakdown(fused, n, warmup):
    """Separate eager passes over resident layer inputs; outputs are never reused in place."""
    model, jitter, device = fused._net, fused._jitter, fused._device
    results = {}
    def layer(name, fn, prepare=None):
        results[name] = measure(fn, device, n, warmup, prepare)
        if prepare is not None:
            prepare()
        return fn()
    x = layer("stem", lambda: F.relu_(model.stem(fused._X)))
    skips = []
    for i, block in enumerate(model.enc):
        x = layer(f"enc {i}", lambda block=block, x=x: _run_enc(block, x))
        if i < len(model.down):
            skips.append(x)
            x = layer(f"down {i}", lambda i=i, x=x: F.relu_(model.down[i](x)))
    for i in range(len(model.down) - 1, -1, -1):
        skip = torch.empty_like(skips[i])
        def up(i=i, x=x, skip=skip):
            fused._shuffle_add(model.up[i](x), skip)
            return skip
        x = layer(f"up {i}", up, lambda i=i, skip=skip: skip.copy_(skips[i]))
    x = layer("fuse", lambda: F.relu_(model.fuse(x)))
    if model.film is None:
        layer("out", lambda: model.out(x))
    else:
        def fold_head():
            if fused.fold == "gpu":
                weight, bias = _fold_head(model, jitter, x.dtype, x.device)
                fused._fold_weight.copy_(weight)
                fused._fold_bias.copy_(bias)
            else:
                weight, bias, _, _, _, _ = fused._host_frames.get(fused._last_inputs[-1])
                fused._fold_weight.copy_(weight, non_blocking=True)
                fused._fold_bias.copy_(bias, non_blocking=True)
            return fused._fold_weight, fused._fold_bias
        weight, bias = layer(f"FiLM/head fold ({fused.fold})", fold_head)
        layer("out", lambda: F.conv2d(x, weight, bias))
    if model.accum:
        layer("phase weights (host)", lambda: fused._wc.copy_(
            fused._host_frames.get(fused._last_inputs[-1])[2], non_blocking=True))
    if model.nearest_sample:
        layer("sample offsets (host)", lambda: fused._offsets.copy_(
            fused._host_frames.get(fused._last_inputs[-1])[3], non_blocking=True))
    if model.base_gate:
        def base_constants():
            _, _, _, _, kernels, windows = fused._host_frames.get(fused._last_inputs[-1])
            fused._base_kernels.copy_(kernels, non_blocking=True)
            fused._base_windows.copy_(windows, non_blocking=True)
        layer("base kernels/windows (host)", base_constants)
    return results


def print_results(results):
    print(f"{'stage':24s} {'median ms':>12s} {'p95 ms':>12s}")
    for name, stats in results.items():
        print(f"{name:24s} {stats['median_ms']:12.4f} {stats['p95_ms']:12.4f}")


PROFILE_TOGGLES = (("base_gate", "fast_base_gate"), ("carry_raw", "fast_carry_raw"),
                   ("nearest_sample", "fast_nearest_sample"), ("accum", "fast_accum"),
                   ("conf_consistent", "fast_conf_consistent"), ("depth_test", "fast_depth_test"),
                   ("depth_soft", "fast_depth_soft"), ("coverage", "fast_coverage"),
                   ("bicubic", "fast_hist_filter"))


def _setting(args, attr):
    return getattr(args, attr) == "bicubic" if attr == "fast_hist_filter" else bool(getattr(args, attr))


def _flipped(args, attr, value):
    out = copy.copy(args)
    setattr(out, attr, ("bicubic" if value else "bilinear") if attr == "fast_hist_filter" else value)
    if attr == "fast_depth_soft" and value:
        out.fast_depth_test = True
    if not out.fast_depth_test:
        out.fast_depth_soft = False
    if not out.fast_accum:
        out.fast_carry_raw = False
    return out


def profile_variants(args):
    """The configured model, each option flipped alone, and the plain/gated/gated+carry corners."""
    rows = [("as configured", args)]
    for label, attr in PROFILE_TOGGLES:
        value = _setting(args, attr)
        rows.append((("- " if value else "+ ") + label, _flipped(args, attr, not value)))
    plain = copy.copy(args)
    plain.fast_base_gate = plain.fast_carry_raw = False
    rows.append(("plain (no base_gate, no carry_raw)", plain))
    gated = copy.copy(plain)
    gated.fast_base_gate = True
    rows.append(("base_gate only", gated))
    both = copy.copy(gated)
    both.fast_carry_raw = both.fast_accum = True
    rows.append(("base_gate + carry_raw", both))
    seen, unique = set(), []
    for label, a in rows:
        key = tuple(_setting(a, attr) for _, attr in PROFILE_TOGGLES)
        if key not in seen:
            seen.add(key)
            unique.append((label, a))
    return unique


def kernel_time(fn, device, n, warmup, repeats=5):
    """Back-to-back launches between two events: launch overhead is not part of the figure."""
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        for _ in range(n):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        values.append(start.elapsed_time(end) / n)
    return torch.tensor(values, dtype=torch.float64).quantile(0.5).item()


@torch.no_grad()
def profile_kernels(args, device, render, output, ap):
    n = max(10, min(args.frames, 100))
    print(f"{torch.cuda.get_device_name(device)}: {render} -> {output}; pack/resolve only, "
          f"{n} back-to-back launches x5, median ms; resolve includes the display write.")
    print(f"{'layout':5s} {'variant':36s} {'pack':>8s} {'resolve':>8s}")
    for layout in ("nhwc", "nchw"):
        for label, a in profile_variants(args):
            try:
                model = build_model(a, render, output, device)
            except ValueError:
                continue
            model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
            fused = FusedFast(model, layout=layout, trunk="cudnn")
            inputs = (*make_inputs(model, device), (0.125, -0.375))
            fused.step(fused.init_state(), *inputs, write_rgb=True)
            fused.step(fused._state(1, 1), *inputs, write_rgb=True)
            with fused._cp.cuda.Device(device.index), fused._stream_context():
                state = fused.init_state()
                fused._stage(state, *inputs)
                pack = kernel_time(lambda: fused._pack(0, False), device, n, 10)
                resolve = kernel_time(lambda: fused._resolve(1, True), device, n, 10)
            print(f"{layout:5s} {label:36s} {pack:8.4f} {resolve:8.4f}", flush=True)
            del fused, model
            torch.cuda.empty_cache()
    return 0


@torch.no_grad()
def main():
    ap = parser()
    args = ap.parse_args()
    try:
        render = tuple(int(x) for x in args.render.lower().split("x"))
        if len(render) != 2 or min(*render, args.scale, args.frames, args.repeat) < 1 or args.warmup < 0:
            raise ValueError
    except ValueError:
        ap.error("render must be HxW, sizes/frames/repeat positive, warmup nonnegative")
    if args.ckpt and any(arg.split("=")[0].startswith("--fast-") or arg.split("=")[0] == "--arch"
                         for arg in sys.argv[1:]):
        ap.error("Use either --ckpt or architecture flags")
    if not args.all_backends and args.trunk == "tensorrt" and args.layout != "nchw":
        ap.error("--trunk tensorrt requires --layout nchw")
    if not torch.cuda.is_available():
        print("CUDA is required for bench_fused; no GPU timings were run.")
        return 1
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cudnn.benchmark = True
    output = tuple(v * args.scale for v in render)
    try:
        model = (load_checkpoint(args.ckpt, render, output, device)[0] if args.ckpt
                 else build_model(args, render, output, device))
    except ValueError as exc:
        ap.error(str(exc))
    if args.profile_kernels:
        return profile_kernels(args, device, render, output, ap)
    backends = (("nhwc", "cudnn"), ("nchw", "cudnn"), ("nchw", "tensorrt")) if args.all_backends else (
        (args.layout, args.trunk),)
    status = 0
    for layout, trunk in backends:
        try:
            status = max(status, benchmark(model, device, render, output, args, layout, trunk))
        except ValueError as exc:
            ap.error(str(exc))
    return status


@torch.no_grad()
def benchmark(model, device, render, output, args, layout, trunk):
    model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
    fused = FusedFast(model, fold=args.fold, layout=layout, trunk=trunk)
    inputs = (*make_inputs(model, device), (0.125, -0.375))
    forward_inputs = (*inputs[:3], torch.tensor(inputs[-1], device=device))
    try:
        fused.step(fused.init_state(), *inputs, write_rgb=args.write_rgb)
    except RuntimeError as exc:
        print(f"Fused CUDA setup failed (layout={layout}, trunk={trunk}): {exc}")
        return 1
    print(f"{torch.cuda.get_device_name(device)}: {render} -> {output}, "
          f"{args.frames} measured frames, {args.warmup} warmup, {args.repeat} repeats", flush=True)
    print("Fixed resident frame, smooth UV motion, recurrent history, fixed jitter.")
    print(f"Fold: {args.fold}; layout: {layout}; trunk: {trunk}; CPU jitter supplied to fused steps.")
    print(f"Nearest sample: {model.nearest_sample}; consistent confidence: {model.conf_consistent}; "
          f"carry raw: {model.carry_raw}; base gate: {model.base_gate}; coverage: {model.coverage}.")
    print("Separate synchronised passes; total includes input staging, trunk includes frame constants and backend execution.")
    display_written = args.write_rgb or model.carry_raw
    print(f"Display write in resolve/total: {display_written}; state_rgb is timed separately.")
    compiled = None
    if args.compile_trunk:
        print("Preparing optional compiled trunk (first compilation excluded).", flush=True)
        try:
            compiled = torch.compile(fused.run_trunk_module, mode="max-autotune-no-cudagraphs")
            actual = compiled(fused._X, fused._jitter)
            expected = fused.run_trunk_module(fused._X, fused._jitter)
            torch.testing.assert_close(actual, expected, rtol=0, atol=3e-3)
            torch.cuda.synchronize(device)
            del actual, expected
        except Exception as exc:
            print(f"Compiled trunk unavailable or failed parity: {type(exc).__name__}: {exc}", flush=True)
            compiled = None
    runs, layers = [], []
    for repeat in range(args.repeat):
        print(f"Benchmark repeat {repeat + 1}/{args.repeat}", flush=True)
        results = fused.timings(args.frames, args.warmup, write_rgb=args.write_rgb)
        state = model.init_state(device)
        def forward():
            nonlocal state
            _, state = model(state, *forward_inputs)
        with torch.autocast("cuda", dtype=torch.float16):
            results["forward autocast"] = measure(forward, device, args.frames, args.warmup)
        display_state = fused._state(1, 1)
        rgb_stage = "state_rgb" if model.carry_raw else "lazy state_rgb"
        results[rgb_stage] = measure(lambda: state_rgb(display_state), device,
                                     args.frames, args.warmup)
        if compiled is not None:
            def compiled_middle():
                fused._update_frame(inputs[-1])
                compiled(fused._X, fused._jitter)
            try:
                results["compiled trunk"] = measure(compiled_middle, device, args.frames, args.warmup)
            except Exception as exc:
                print(f"Compiled trunk timing failed: {type(exc).__name__}: {exc}", flush=True)
                compiled = None
                for run in runs:
                    run.pop("compiled trunk", None)
        runs.append(results)
        layers.append(layer_breakdown(fused, args.frames, args.warmup))
    print("Stage timings: median of repeat medians and median of repeat p95s.")
    results = median_repeats(runs)
    results["total (with display)"] = {
        key: results["total"][key] + (0 if model.carry_raw else results["lazy state_rgb"][key])
        for key in results["total"]}
    print_results(results)
    if model.carry_raw:
        print("carry_raw always writes the display in resolve; total (with display) equals total.")
    else:
        print("total (with display) is total + lazy state_rgb; p95 is the sum of separate p95s.")
    widths = ",".join(str(v) for v in (model.stem.out_channels, *(m.out_channels for m in model.down)))
    depths = ",".join(str(len(block)) for block in model.enc)
    print(f"SUMMARY widths={widths} depths={depths} layout={layout} trunkbackend={trunk} "
          f"pack={results['pack']['median_ms']:.4f} trunk={results['trunk']['median_ms']:.4f} "
          f"resolve={results['resolve']['median_ms']:.4f} total={results['total']['median_ms']:.4f} "
          f"total_with_display={results['total (with display)']['median_ms']:.4f}")
    print("Trunk layer breakdown: separate eager passes, each synchronised; includes host overhead.")
    print("Layer times do not sum to backend execution time. FiLM/head folding is timed separately.")
    print_results(median_repeats(layers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
