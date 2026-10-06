"""Export a strict v1 model; no pickle is read by the Windows runtime."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
import torch
from fsrmamba.config import load_checkpoint, load_sidecar
from fsrmamba.fused import _validate
from fsrmamba.kpn_unet import KPNAccumulator, DEFAULT_WIDTHS, LITE_WIDTHS


def model_config(model):
    if isinstance(model, KPNAccumulator):
        sigma_min = float(getattr(model, "sigma_min", .3))
        proximity = float(getattr(model, "proximity", 0.))
        proximity_gain = float(getattr(model, "proximity_gain", 2.))
        if not math.isfinite(sigma_min) or not 1e-3 <= sigma_min <= 2.5:
            raise ValueError("KPN sigma_min must be finite and in [1e-3, 2.5]")
        if not math.isfinite(proximity) or not (proximity == 0 or 1e-3 <= proximity <= 16):
            raise ValueError("KPN proximity must be 0 (off) or finite and in [1e-3, 16]")
        if not math.isfinite(proximity_gain) or not 1e-3 <= proximity_gain <= 64:
            raise ValueError("KPN proximity_gain must be finite and in [1e-3, 64]")
        preset = "lite" if model.lite else "default"
        if model.widths != (LITE_WIDTHS if model.lite else DEFAULT_WIDTHS):
            preset = "custom"
        return dict(arch="kpn", scale=2, widths=list(model.widths), lite=model.lite,
                    trunk_stride=model.trunk_stride, taps=model.taps, history_filter=model.history_filter,
                    residual=model.residual, mv_dilate=model.mv_dilate, depth_soft=model.depth_soft,
                    jitter_sign=model.jitter_sign, sigma_min=sigma_min, proximity=proximity,
                    proximity_gain=proximity_gain, preset=preset, render_size=list(model.render_size),
                    output_size=list(model.output_size), padded_size=list(model.padded_size))
    _validate(model)
    if model.scale != 2 or not model.accum or model.hist_residual:
        raise ValueError("v1 requires scale=2, accum=True, hist_residual=False")
    if model.jitter_sign not in (-1, 1):
        raise ValueError("v1 requires jitter_sign=+1 or -1")
    cfg = {key: getattr(model, key) for key in (
        "n_state", "learned_clamp", "detail_ch", "resolve", "hist_residual", "accum",
        "depth_test", "depth_soft", "depth_soft_osc", "nearest_sample", "conf_consistent", "carry_raw", "base_gate",
        "hist_filter", "conf_motion", "jitter_sign", "conf_max",
        "mv_dilate", "depth_dilate", "thin_lock", "coverage", "history_age")}
    cfg.update(coverage_bias=float(model.coverage_bias) if model.coverage else -6.0,
               arch="fast", scale=2, stem_kernel=1, film=model.film is not None,
               widths=[model.stem.out_channels] + [layer.out_channels for layer in model.down],
               depths=[len(block) for block in model.enc])
    return cfg


def export(model, output):
    cfg = model_config(model)
    tensors = {}
    for name, tensor in model.named_parameters():
        cpu = name.startswith(("out.", "film.")) or tensor.ndim == 0
        tensor = tensor.detach().cpu().to(torch.float32 if cpu else torch.float16).contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite or overflowing weights: {name}")
        tensors[name] = tensor
    config = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    with open(output, "wb") as f:
        f.write(b"FSMWGT1\0" + struct.pack("<I", len(config)) + config)
        f.write(struct.pack("<I", len(tensors)))
        for name, tensor in sorted(tensors.items()):
            name = name.encode("ascii")
            dtype = 1 if tensor.dtype == torch.float16 else 2
            raw = tensor.numpy().astype("<f2" if dtype == 1 else "<f4", copy=False).tobytes()
            f.write(struct.pack("<I", len(name)) + name)
            f.write(struct.pack("<II", dtype, tensor.ndim))
            f.write(struct.pack("<" + "I" * tensor.ndim, *tensor.shape))
            f.write(struct.pack("<Q", len(raw)) + raw)
    return tensors


def load(path, render=(8, 8)):
    if load_sidecar(path) is None:
        raise ValueError("checkpoint architecture sidecar (.json) is required")
    model, _ = load_checkpoint(path, render, tuple(2 * x for x in render), "cpu")
    model_config(model)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    tensors = export(load(a.checkpoint), a.output)
    print(f"Exported {len(tensors)} tensors to {a.output}")


if __name__ == "__main__":
    main()
