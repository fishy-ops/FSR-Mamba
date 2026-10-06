"""Export the HQ tensor-only trunk at static 960x540 / 1280x720 render sizes.

Packing, UV reprojection, exposure normalization, reset masking, phase resolve
and age updates remain in HQAccumulator's reference PyTorch path. The caller
passes padded packed features, signed jitter and already-warped/reset hidden
state; it crops/shuffles parameter planes and persists the returned hidden state.
No TensorRT performance or end-to-end deployment claim is made by this export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import add_arch_args, build_model, load_checkpoint


def export_inputs(model, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, model.in_channels, *model.padded_size, generator=generator),
            torch.rand(1, 2, generator=generator) - .5,
            torch.rand(1, model.n_state, *(v // 8 for v in model.padded_size),
                       generator=generator) * 2 - 1)


def export_hq(model, path):
    try:
        import onnx
    except ImportError:
        print("SKIP: ONNX is not installed; no graph exported.")
        return False
    if model.__class__.__name__ != "HQAccumulator":
        raise ValueError("HQ export requires an HQ checkpoint")
    model = model.cpu().float().eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = export_inputs(model)
    names = ["packed", "jitter", "hidden"]
    with torch.no_grad():
        torch.onnx.export(model.trunk, inputs, str(path), dynamo=False, opset_version=18,
                          input_names=names, output_names=["parameters", "hidden_next"])
    onnx.checker.check_model(onnx.load(str(path)))
    print(f"Exported {path}: render={model.render_size}, padded={model.padded_size}")
    print(json.dumps(model.cost(), sort_keys=True))
    try:
        import onnxruntime as ort
    except ImportError:
        print("SKIP: onnxruntime is not installed; output parity unverified.")
        return True
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for seed in (0, 1):
        inputs = export_inputs(model, seed)
        with torch.no_grad():
            expected = model.trunk(*inputs)
        actual = session.run(None, {name: value.numpy() for name, value in zip(names, inputs)})
        for name, ref, result in zip(("parameters", "hidden_next"), expected, actual):
            result = torch.from_numpy(result)
            if result.shape != ref.shape or not torch.isfinite(result).all():
                raise AssertionError(f"Invalid ONNX {name}: {result.shape}")
            error = (ref - result).abs().max().item()
            if error > 1e-3:
                raise AssertionError(f"ONNX {name} max absolute error {error:g} > 1e-3")
            print(f"Parity seed={seed} {name}: max absolute error {error:.6g}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", help="HQ checkpoint with sidecar; omit for initialized weights")
    ap.add_argument("--render", nargs="+", choices=("960x540", "1280x720"),
                    default=["960x540", "1280x720"], help="width x height; exports both by default")
    add_arch_args(ap)
    ap.set_defaults(arch="hq")
    args = ap.parse_args()
    if args.arch != "hq":
        ap.error("this exporter supports --arch hq only")
    if not args.ckpt:
        print("Using initialized weights; pass --ckpt for trained weights.")
    for size in args.render:
        w, h = (int(v) for v in size.split("x"))
        model = (load_checkpoint(args.ckpt, (h, w), (2 * h, 2 * w))[0] if args.ckpt
                 else build_model(args, (h, w), (2 * h, 2 * w)))
        export_hq(model, Path(args.out_dir) / f"hq_{size}.onnx")


if __name__ == "__main__":
    main()
