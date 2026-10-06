"""Sweep fast pyramid sizes and state widths using synchronised step timings."""

from __future__ import annotations

import argparse
import csv
from itertools import islice, product
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_latency import make_inputs, run_pass
from fsrmamba.config import build_model


WIDTHS = ((16,), (24,), (32,), (16, 32), (24, 48), (32, 64), (48, 96), (32, 64, 128))


def configs():
    for widths in WIDTHS:
        grid = [(0, 1, 2)] + [(1, 2, 4)] * (len(widths) - 1)
        for depths in product(*grid):
            for n_state in (0, 8):
                yield dict(arch="fast", widths=list(widths), depths=list(depths), n_state=n_state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", default="540x960", help="Render height x width.")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--budget-ms", type=float)
    ap.add_argument("--csv")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    try:
        render = tuple(int(x) for x in args.render.lower().split("x"))
        if len(render) != 2 or min(*render, args.scale, args.frames) < 1 or args.warmup < 0:
            raise ValueError
    except ValueError:
        ap.error("render must be HxW, sizes/frames positive, and warmup nonnegative")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.budget_ms is not None and args.budget_ms <= 0:
        ap.error("--budget-ms must be positive")
    device = torch.device(args.device)
    fp16 = args.fp16 and device.type != "cpu"
    if args.fp16 and device.type == "cpu":
        print("FP16 requested on CPU; running fp32.")
    output = tuple(v * args.scale for v in render)
    rows = []
    selected = configs() if args.limit is None else islice(configs(), args.limit)
    for i, cfg in enumerate(selected):
        torch.manual_seed(0)
        model = build_model(cfg, render, output, device)
        inputs = make_inputs(model, device)
        medians = {}
        for mode in ("network", "full"):
            times = run_pass(model, inputs, device, args.frames, args.warmup, fp16, mode=mode)
            medians[mode] = torch.tensor(times, dtype=torch.float64).quantile(0.5).item()
        within = args.budget_ms is not None and medians["network"] <= args.budget_ms
        rows.append(dict(widths=",".join(map(str, cfg["widths"])),
                         depths=",".join(map(str, cfg["depths"])), n_state=cfg["n_state"],
                         parameters=sum(p.numel() for p in model.parameters()),
                         network_ms=medians["network"], full_ms=medians["full"],
                         within_budget=within))
        print(f"Measured config {i + 1}: {cfg['widths']} / {cfg['depths']} / state={cfg['n_state']}",
              flush=True)
        del model, inputs
    rows.sort(key=lambda row: row["network_ms"])
    print(f"\nMedian latency ({device.type}, {'fp16' if fp16 else 'fp32'}); "
          f"{args.frames} frames after {args.warmup} warmup")
    print("widths       depths    state  parameters  network + blend ms  full step ms  budget")
    for r in rows:
        print(f"{r['widths']:12s} {r['depths']:9s} {r['n_state']:5d} {r['parameters']:11d} "
              f"{r['network_ms']:19.3f} {r['full_ms']:13.3f}  "
              + ("*" if r["within_budget"] else ""))
    if args.budget_ms is not None:
        print(f"* network + blend within {args.budget_ms:g} ms budget")
    if args.csv:
        with Path(args.csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
