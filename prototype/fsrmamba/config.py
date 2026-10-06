"""Checkpoint architecture metadata and strict legacy reconstruction."""

from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path

import torch

from .fast import FastAccumulator
from .hq import HQAccumulator
from .kpn_unet import KPNAccumulator, DEFAULT_WIDTHS, LITE_WIDTHS
from .phase import PhaseAccumulator
from .mamba import MambaAccumulator, MambaState


ARCH_DEFAULTS = {
    "state_channels": 8, "feature_channels": 24, "num_experts": 1,
    "sharpen": False, "rectify": False, "box_max": 3.0,
    "learned_resolve": False, "swin_resolve": False, "swin_dim": 48,
    "lr_upsampler": False, "lr_dim": 64, "lr_film": False,
    "separable": False, "unet": 0, "kernel_predict": False,
    "robust_disocc": False, "lock_feature": False, "alpha_scale": 1.0,
    "encoder_depth": 2, "history_input": False, "unet_unshuffle": False,
    "ablate_ssm": False, "jitter_sign": 1.0,
}


FAST_DEFAULTS = {
    "widths": [32, 64], "depths": [1, 2], "n_state": 8, "film": True,
    "depth_test": False, "stem_kernel": 1, "learned_clamp": False, "resolve": "nearest", "detail_ch": 0, "hist_filter": "bilinear", "accum": False, "conf_motion": False, "hist_residual": False, "nearest_sample": False,
    "conf_consistent": False, "carry_raw": False, "base_gate": False,
    "reset_lanczos": False, "coverage": False, "coverage_bias": -6.0, "depth_soft": False, "mv_dilate": False, "depth_dilate": False, "thin_lock": False,
    "jitter_sign": 1.0, "depth_soft_osc": False, "history_age": False,
}


HQ_DEFAULTS = {
    "c0": 24, "widths": [48, 96, 128], "depths": [2, 2, 2],
    "n_state": 16, "jitter_sign": 1.0,
}


PHASE_DEFAULTS = {
    "widths": [32, 64], "depths": [1, 2], "n_state": 0, "film": True, "residual": False,
    "box_max": 3.0, "jitter_sign": 1.0,
}


KPN_DEFAULTS = {
    "widths": list(DEFAULT_WIDTHS), "lite": False, "residual": True,
    "mv_dilate": True, "depth_soft": False, "jitter_sign": 1.0, "sigma_min": .3,
    "proximity": 0., "proximity_gain": 2., "trunk_stride": 1, "taps": 5, "history_filter": "bicubic",
}


def add_arch_args(ap):
    """Add architecture options without duplicating existing training flags."""
    def add(option, **opts):
        if option not in ap._option_string_actions:
            ap.add_argument(option, **opts)

    add("--arch", choices=("mamba", "fast", "phase", "hq", "kpn"), default="mamba")
    add("--kpn-trunk-stride", type=int, choices=(1, 2), default=1)
    add("--kpn-taps", type=int, choices=(3, 5), default=5)
    add("--kpn-history-filter", choices=("bicubic", "catmull"), default="bicubic")
    add("--kpn-widths", default=None, help="Six encoder/bottleneck widths; overrides the selected preset.")
    add("--kpn-lite", action="store_true", help="Widths 16,24,48,64,96,128; one conv per decoder level.")
    add("--kpn-no-residual", action="store_true")
    add("--kpn-no-mv-dilate", action="store_true")
    add("--kpn-depth-soft", action="store_true", help="Keep color history on depth mismatch; still reset age.")
    add("--kpn-sigma-min", type=float, default=.3, help="Lower bound of the Gaussian sigmas in render pixels.")
    add("--kpn-proximity", type=float, default=0., help="Proximity-weighted accumulation sigma in output pixels (0 = off).")
    add("--kpn-proximity-gain", type=float, default=2., help="Gain on the proximity-weighted current-frame weight.")
    add("--phase-residual", action="store_true")
    add("--hq-c0", type=int, default=24)
    add("--hq-widths", default="48,96,128")
    add("--hq-depths", default="2,2,2")
    add("--hq-state", type=int, default=16)
    for key, default in ARCH_DEFAULTS.items():
        opts = {"default": default, "help": f"{key.replace('_', ' ')} (default: {default})."}
        if isinstance(default, bool):
            opts["action"] = "store_true"
        else:
            opts["type"] = type(default)
        add("--" + key.replace("_", "-"), **opts)
    add("--fast-widths", default="32,64")
    add("--fast-depths", default="1,2")
    add("--fast-state", type=int, default=8)
    add("--fast-no-film", action="store_true")
    add("--fast-coverage", action="store_true")
    add("--fast-history-age", action="store_true")
    add("--coverage-bias", type=float, default=-3.0)
    add("--fast-depth-test", action="store_true")
    add("--fast-depth-soft", action="store_true",
        help="Feed depth mismatch to the trunk without resetting history; requires --fast-depth-test.")
    add("--fast-depth-soft-osc", action="store_true",
        help="Gate soft depth handling with coverage oscillation; requires depth-test, depth-soft and coverage.")
    add("--fast-stem-kernel", type=int, default=1)
    add("--fast-learned-clamp", action="store_true")
    add("--fast-resolve", choices=("nearest", "lanczos"), default="nearest")
    add("--fast-detail", type=int, default=0)
    add("--fast-hist-filter", choices=("bilinear", "bicubic"), default="bilinear")
    add("--fast-accum", action="store_true")
    add("--fast-conf-motion", action="store_true")
    add("--fast-hist-residual", action="store_true")
    add("--fast-nearest-sample", action="store_true")
    add("--fast-conf-consistent", action="store_true")
    add("--fast-carry-raw", action="store_true")
    add("--fast-base-gate", action="store_true")
    add("--fast-reset-lanczos", action="store_true")
    for key in ("mv-dilate", "depth-dilate", "thin-lock"):
        add("--fast-" + key, action="store_true")


def _arch(src):
    values = src if isinstance(src, dict) else vars(src)
    arch = values.get("arch", "mamba")
    if arch == "mamba":
        return dict(arch=arch, **{key: values.get(key, default)
                                 for key, default in ARCH_DEFAULTS.items()})
    if arch == "kpn":
        lite = values.get("lite", values.get("kpn_lite", False))
        widths = values.get("widths", values.get("kpn_widths"))
        if widths is None:
            widths = LITE_WIDTHS if lite else DEFAULT_WIDTHS
        widths = [int(v) for v in widths.split(",")] if isinstance(widths, str) else list(widths)
        cfg = dict(arch=arch, widths=widths, lite=lite,
                   trunk_stride=values.get("trunk_stride", values.get("kpn_trunk_stride", 1)),
                   taps=values.get("taps", values.get("kpn_taps", 5)),
                   history_filter=values.get("history_filter", values.get("kpn_history_filter", "bicubic")),
                   residual=values.get("residual", not values.get("kpn_no_residual", False)),
                   mv_dilate=values.get("mv_dilate", not values.get("kpn_no_mv_dilate", False)),
                   depth_soft=values.get("depth_soft", values.get("kpn_depth_soft", False)),
                   jitter_sign=values.get("jitter_sign", 1.0),
                   sigma_min=float(values.get("sigma_min", values.get("kpn_sigma_min", .3))),
                   proximity=float(values.get("proximity", values.get("kpn_proximity", 0.))),
                   proximity_gain=float(values.get("proximity_gain", values.get("kpn_proximity_gain", 2.))))
        if (len(widths) != 6 or any(not isinstance(c, int) or c < 1 for c in widths)
                or cfg["jitter_sign"] not in (-1., 1.)
                or cfg["trunk_stride"] not in (1, 2) or cfg["taps"] not in (3, 5)
                or cfg["history_filter"] not in ("bicubic", "catmull")):
            raise ValueError(f"Invalid KPN architecture: {cfg}")
        return cfg
    if arch == "hq":
        cfg = dict(arch=arch)
        for key, default in HQ_DEFAULTS.items():
            alias = "hq_state" if key == "n_state" else "hq_" + key
            value = values.get(key, values.get(alias, default))
            if key in ("widths", "depths"):
                value = [int(v) for v in value.split(",")] if isinstance(value, str) else list(value)
            cfg[key] = value
        if (len(cfg["widths"]) != 3 or len(cfg["depths"]) != 3
                or min(cfg["c0"], cfg["n_state"], *cfg["widths"], *cfg["depths"]) < 1
                or cfg["jitter_sign"] not in (-1., 1.)):
            raise ValueError(f"Invalid HQ architecture: {cfg}")
        return cfg
    if arch == "phase":
        cfg = dict(arch=arch)
        alias = {"widths": "fast_widths", "depths": "fast_depths", "n_state": "fast_state",
                 "residual": "phase_residual"}
        for key, default in PHASE_DEFAULTS.items():
            value = values.get(key, values.get(alias.get(key), default))
            if key in ("widths", "depths"):
                value = [int(v) for v in value.split(",")] if isinstance(value, str) else list(value)
            if key == "film" and "fast_no_film" in values and "film" not in values:
                value = not values["fast_no_film"]
            cfg[key] = value
        return cfg
    if arch != "fast":
        raise ValueError(f"Unknown architecture: {arch!r}")
    cfg = dict(arch=arch)
    aliases = {"widths": "fast_widths", "depths": "fast_depths", "n_state": "fast_state",
               "depth_test": "fast_depth_test", "depth_soft": "fast_depth_soft",
               "depth_soft_osc": "fast_depth_soft_osc",
               "stem_kernel": "fast_stem_kernel",
               "learned_clamp": "fast_learned_clamp", "resolve": "fast_resolve",
               "detail_ch": "fast_detail", "hist_filter": "fast_hist_filter",
               "accum": "fast_accum", "conf_motion": "fast_conf_motion",
               "hist_residual": "fast_hist_residual",
               "nearest_sample": "fast_nearest_sample",
               "conf_consistent": "fast_conf_consistent", "carry_raw": "fast_carry_raw",
               "base_gate": "fast_base_gate", "reset_lanczos": "fast_reset_lanczos",
               "mv_dilate": "fast_mv_dilate", "depth_dilate": "fast_depth_dilate",
               "thin_lock": "fast_thin_lock", "coverage": "fast_coverage",
               "history_age": "fast_history_age"}
    for key, default in FAST_DEFAULTS.items():
        value = values.get(key, values.get(aliases.get(key), default))
        if key in ("widths", "depths"):
            value = [int(v) for v in value.split(",")] if isinstance(value, str) else list(value)
        if key == "film" and "fast_no_film" in values and "film" not in values:
            value = not values["fast_no_film"]
        cfg[key] = value
    if (not cfg["widths"] or len(cfg["widths"]) != len(cfg["depths"])
            or any(v < 1 for v in cfg["widths"]) or any(v < 0 for v in cfg["depths"])
            or cfg["n_state"] < 0 or cfg["stem_kernel"] < 1 or cfg["stem_kernel"] % 2 == 0):
        raise ValueError(f"Invalid fast architecture: {cfg}")
    if cfg["depth_soft"] and not cfg["depth_test"]:
        raise ValueError("depth_soft requires depth_test=True")
    if cfg["depth_soft_osc"] and not (cfg["depth_test"] and cfg["depth_soft"] and cfg["coverage"]):
        raise ValueError("depth_soft_osc requires depth_test, depth_soft and coverage")
    return cfg


def model_kwargs(src):
    """Return the selected architecture and supported constructor options."""
    cfg = _arch(src)
    if cfg["arch"] == "kpn":
        return dict(cfg, widths=tuple(cfg["widths"]))
    if cfg["arch"] in ("fast", "phase", "hq"):
        return dict(cfg, widths=tuple(cfg["widths"]), depths=tuple(cfg["depths"]))
    accepted = inspect.signature(MambaAccumulator.__init__).parameters
    for key, value in cfg.items():
        if key != "arch" and key not in accepted and value != ARCH_DEFAULTS[key]:
            raise ValueError(f"MambaAccumulator does not support {key}={value!r}")
    return {key: value for key, value in cfg.items() if key == "arch" or key in accepted}


def build_model(cfg, render_size, output_size, device="cpu"):
    kwargs = model_kwargs(cfg)
    cls = {"fast": FastAccumulator, "phase": PhaseAccumulator, "hq": HQAccumulator, "kpn": KPNAccumulator,
           "mamba": MambaAccumulator}[kwargs.pop("arch")]
    return cls(tuple(render_size), tuple(output_size), **kwargs, device=device).eval()


def new_state(model, device="cpu"):
    if hasattr(model, "init_state"):
        return model.init_state(device)
    return MambaState.zeros(model.output_size, model.state_channels, device=device)


def save_sidecar(ckpt_path, src):
    model_kwargs(src)
    Path(str(ckpt_path) + ".json").write_text(
        json.dumps(_arch(src), sort_keys=True, indent=2) + "\n")


def load_sidecar(ckpt_path):
    path = Path(str(ckpt_path) + ".json")
    if not path.exists():
        return None
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid architecture sidecar: {path}")
    allowed = {"fast": FAST_DEFAULTS, "phase": PHASE_DEFAULTS, "hq": HQ_DEFAULTS, "kpn": KPN_DEFAULTS}.get(cfg.get("arch"), ARCH_DEFAULTS)
    if (cfg.get("arch", "mamba") not in ("mamba", "fast", "phase", "hq", "kpn")
            or set(cfg) - (set(allowed) | {"arch"})):
        raise ValueError(f"Invalid architecture sidecar: {path}")
    return cfg


def _overrides(cfg, overrides):
    if overrides:
        allowed = {"fast": FAST_DEFAULTS, "phase": PHASE_DEFAULTS, "hq": HQ_DEFAULTS, "kpn": KPN_DEFAULTS}.get(cfg["arch"], ARCH_DEFAULTS)
        unknown = set(overrides) - allowed.keys()
        if unknown:
            raise ValueError(f"Unknown architecture overrides: {sorted(unknown)}")
        cfg.update(overrides)
    return cfg


def _infer_phase(sd, overrides):
    c, channels = _conv_shape(sd, "stem")
    out, _ = _conv_shape(sd, "gates")
    # stem in = 4*(6 + 5p) + n, gates out = 4*p*(3 | 6) + 2n.
    residual = "res_gain" in sd
    div = 16 if residual else 28
    numerator = 2 * channels - out - 48
    p = numerator // div
    if numerator % div or p < 1 or math.isqrt(p) ** 2 != p:
        raise ValueError(f"Invalid phase head/stem shapes: stem in {channels}, gates {out}")
    widths = [c]
    for i in sorted({int(m[1]) for key in sd if (m := re.fullmatch(r"down\.(\d+)\.weight", key))}):
        widths.append(_conv_shape(sd, f"down.{i}")[0])
    depths = [len({int(m[1]) for key in sd
                   if (m := re.match(rf"enc\.{i}\.(\d+)\.", key))}) for i in range(len(widths))]
    cfg = dict(arch="phase", widths=widths, depths=depths, n_state=channels - 4 * (6 + 5 * p),
               film="film.0.weight" in sd, residual=residual, box_max=3.0,
               jitter_sign=float(sd["_jitter_sign"]) if "_jitter_sign" in sd else 1.0)
    return _overrides(cfg, overrides)


def _infer_fast(sd, overrides):
    """Solve head/stem shapes; acc_sharp and _carry_marker identify accumulation modes.

    in_ch = 9 + 3p + 3*learned_clamp + p*(accum + carry_raw) + thin_lock + 2*coverage + history_age.
    Per-pixel outputs = (4 + learned_clamp + base_gate + accum + carry_raw + coverage + 3*hist_residual)*p.
    Without the detail branch: stem in = 4*in_ch + n, out = 4*n_px + 2n.
    With it: detail in = detail_ch + in_ch, detail_out = n_px, stem in = 4*in_ch + n.
    """
    c, channels = _conv_shape(sd, "stem")
    out, _ = _conv_shape(sd, "out")
    accum = "acc_sharp" in sd
    hres = "hres_gain" in sd
    carry_raw = "_carry_marker" in sd
    base_gate = "_base_gate_marker" in sd
    coverage = "_coverage_marker" in sd
    dch = sd["detail.weight"].shape[0] if "detail.weight" in sd else 0
    found = []
    for learned_clamp in (False, True):
        if carry_raw and (not accum or learned_clamp or hres):
            continue
        for p in (1, 4, 9, 16):
            in_ch = 9 + 3 * p + 3 * learned_clamp + p * (accum + carry_raw) + ("thin_slack" in sd) + 2 * coverage + ("alpha_min" in sd)
            n_px = (4 + learned_clamp + base_gate + accum + carry_raw + 3 * hres + coverage) * p
            n_state = channels - 4 * in_ch
            if n_state < 0:
                continue
            if dch:
                ok = (sd["detail.weight"].shape[1] == dch + in_ch
                      and sd["detail_out.weight"].shape[0] == n_px and out == 4 * dch + 2 * n_state)
            else:
                ok = out == 4 * n_px + 2 * n_state
            if ok:
                found.append((n_state, learned_clamp))
    if not found:
        raise ValueError(f"Invalid fast head/stem shapes: stem in {channels}, out {out}")
    # Several (scale, state) pairs can fit the same shapes; the one with the fewest state
    # channels is the real one in practice (state is 0 or small).
    n_state, learned_clamp = min(found)
    return _fast_cfg(sd, c, n_state, learned_clamp, dch, accum, overrides)


def _fast_cfg(sd, c, n_state, learned_clamp, detail_ch, accum, overrides):
    if n_state < 0:
        raise ValueError(f"Invalid fast state channels: {n_state}")
    widths = [c]
    for i in sorted({int(m[1]) for key in sd if (m := re.fullmatch(r"down\.(\d+)\.weight", key))}):
        widths.append(_conv_shape(sd, f"down.{i}")[0])
    depths = [len({int(m[1]) for key in sd
                   if (m := re.match(rf"enc\.{i}\.(\d+)\.", key))}) for i in range(len(widths))]
    # Options with no tensor footprint (nearest_sample, conf_consistent, ...) keep their
    # defaults here; a sidecar or `overrides` supplies them.
    cfg = dict(FAST_DEFAULTS, arch="fast")
    cfg.update(widths=widths, depths=depths, n_state=n_state,
               film="film.0.weight" in sd, depth_test="soft_osc_threshold" in sd,
               depth_soft="soft_osc_threshold" in sd, depth_soft_osc="soft_osc_threshold" in sd,
               stem_kernel=sd["stem.weight"].shape[2], learned_clamp=learned_clamp,
               resolve="lanczos" if "jit_gain" in sd else "nearest", detail_ch=detail_ch,
               hist_filter="bilinear", accum=accum, conf_motion="conf_m" in sd,
               hist_residual="hres_gain" in sd, carry_raw="_carry_marker" in sd,
               base_gate="_base_gate_marker" in sd,
               mv_dilate="_mv_dilate_marker" in sd, depth_dilate="_depth_dilate_marker" in sd,
               thin_lock="thin_slack" in sd, coverage="_coverage_marker" in sd,
               coverage_bias=float(sd.get("coverage_bias", -6.0)), history_age="alpha_min" in sd,
               jitter_sign=float(sd["_jitter_sign"]) if "_jitter_sign" in sd else 1.0)
    return _overrides(cfg, overrides)


def _conv_shape(sd, prefix):
    key = prefix + ".weight"
    if prefix + ".pw.weight" in sd:
        key = prefix + ".pw.weight"
        dw = sd.get(prefix + ".dw.weight")
        if dw is None or dw.ndim != 4 or dw.shape[1] != 1:
            raise ValueError(f"Invalid shape for {prefix}.dw.weight: {getattr(dw, 'shape', None)}")
        in_ch = dw.shape[0]
    else:
        weight = sd.get(key)
        in_ch = weight.shape[1] if weight is not None and weight.ndim == 4 else None
    weight = sd.get(key)
    if weight is None or weight.ndim != 4:
        raise ValueError(f"Invalid shape for {key}: {getattr(weight, 'shape', None)}")
    return weight.shape[0], in_ch


def infer_config(state_dict, overrides=None):
    """Recover shape-encoded architecture; legacy disocclusion defaults to robust."""
    sd = state_dict
    if "_kpn_spec" in sd:
        spec = [int(v) for v in sd["_kpn_spec"].tolist()]
        return _overrides(dict(arch="kpn", widths=spec[:6], lite=bool(spec[6]),
                               trunk_stride=int(sd.get("_kpn_stride", 1)), taps=int(sd.get("_kpn_taps", 5)),
                               history_filter="catmull" if "_kpn_catmull" in sd else "bicubic",
                               residual=bool(spec[7]), mv_dilate=bool(spec[8]),
                               depth_soft=bool(spec[9]), jitter_sign=float(sd["_jitter_sign"]),
                               sigma_min=round(float(sd.get("_sigma_min", .3)), 6),
                               proximity=round(float(sd["_proximity"][0]), 6) if "_proximity" in sd else 0.,
                               proximity_gain=round(float(sd["_proximity"][1]), 6) if "_proximity" in sd else 2.), overrides)
    if "_hq_spec" in sd:
        c0, a, b, c, d, e, f, n = (int(v) for v in sd["_hq_spec"].tolist())
        return _overrides(dict(arch="hq", c0=c0, widths=[a, b, c], depths=[d, e, f],
                               n_state=n, jitter_sign=float(sd["_jitter_sign"])), overrides)
    if "stem.weight" in sd and "gates.weight" in sd:
        return _infer_phase(sd, overrides)
    if "stem.weight" in sd and "out.weight" in sd and "to_delta.weight" not in sd:
        return _infer_fast(sd, overrides)
    cfg = dict(ARCH_DEFAULTS, arch="mamba")
    prefixes = {
        "lr_upsampler": "lr_up.", "swin_resolve": "swin_refine.",
        "learned_resolve": "resolve_refine.", "kernel_predict": "kpn_head.",
        "sharpen": "sharpen_net.", "lr_film": "lr_up.film.",
    }
    for flag, prefix in prefixes.items():
        cfg[flag] = any(key.startswith(prefix) for key in sd)
    cfg["jitter_sign"] = float(sd["_jitter_sign"]) if "_jitter_sign" in sd else 1.0
    cfg["rectify"] = "to_boxscale.weight" in sd
    cfg["separable"] = any(key.endswith(".dw.weight") for key in sd)
    blocks = {int(m[1]) for key in sd if (m := re.match(r"encoder_unet\.enc\.(\d+)\.", key))}
    cfg["unet"] = len(blocks) - 1 if blocks else 0
    prefix = "encoder_unet.enc.0.body.0" if blocks else "encoder.0"
    cfg["feature_channels"], in_ch = _conv_shape(sd, prefix)
    if blocks and in_ch in (48, 52, 76, 80):
        cfg["unet_unshuffle"] = True
        in_ch //= 4
    if in_ch not in (12, 13, 19, 20):
        raise ValueError(f"Unsupported input shape for {prefix}: input channels "
                         f"{_conv_shape(sd, prefix)[1]}")
    cfg["lock_feature"] = in_ch in (13, 20)
    cfg["history_input"] = in_ch in (19, 20)
    if not blocks:
        convs = {int(m[1]) for key in sd
                 if (m := re.fullmatch(r"encoder\.(\d+)\.(?:pw\.|dw\.)?weight", key))}
        cfg["encoder_depth"] = len(convs)
    n_learned, _ = _conv_shape(sd, "to_delta")
    cfg["num_experts"] = _conv_shape(sd, "router")[0] if "router.weight" in sd else 1
    k = cfg["num_experts"]
    cfg["state_channels"] = n_learned + 3 + (k if k > 1 else 0)
    if cfg["lr_upsampler"]:
        cfg["lr_dim"] = _conv_shape(sd, "lr_up.head")[0]
    if cfg["swin_resolve"]:
        cfg["swin_dim"] = _conv_shape(sd, "swin_refine.embed")[0]
    cfg["robust_disocc"] = True
    return _overrides(cfg, overrides)


def load_model_state(model, sd, strict=True):
    """Expand optional evidence inputs/heads and default new scalars for legacy files."""
    if strict and isinstance(model, FastAccumulator) and model.depth_soft_osc and "soft_osc_threshold" not in sd:
        sd = dict(sd, soft_osc_threshold=model.soft_osc_threshold.detach().clone())
    if isinstance(model, FastAccumulator) and model.history_age and "alpha_min" not in sd:
        sd = dict(sd)
        own = model.state_dict()
        for name, extra, start in (("stem.weight", 4, model.stem.in_channels - model.n_state - 4),
                                   ("detail.weight", 1, model.detail.in_channels - 1 if model.detail_ch else 0)):
            if name not in sd:
                continue
            src = sd[name]
            # Coverage may also be new; its expansion below preserves this final channel.
            insertion = start - (8 if name == "stem.weight" else 2) if model.coverage and "_coverage_marker" not in sd else start
            if src.shape[1] < insertion:
                continue
            sd[name] = torch.cat((src[:, :insertion], src.new_zeros(src.shape[0], extra, *src.shape[2:]),
                                  src[:, insertion:]), 1)
        sd["alpha_min"] = own["alpha_min"]
    if isinstance(model, FastAccumulator) and model.coverage and "_coverage_marker" not in sd:
        sd = dict(sd)
        own = model.state_dict()
        input0 = model.stem.in_channels - model.n_state - 8 - 4 * model.history_age
        head0 = model._cov_offset * (1 if model.detail_ch else 4)
        head_extra = model.p * (1 if model.detail_ch else 4)
        def expand(name, axis, start, count):
            if name not in sd:
                return
            src, dst = sd[name], own[name].clone()
            shape = list(dst.shape)
            shape[axis] -= count
            if tuple(shape) != tuple(src.shape):
                return  # Other architecture mismatches retain load_state_dict's error.
            dst.narrow(axis, 0, start).copy_(src.narrow(axis, 0, start))
            dst.narrow(axis, start + count, src.shape[axis] - start).copy_(
                src.narrow(axis, start, src.shape[axis] - start))
            if axis == 1 or name.endswith("weight"):
                dst.narrow(axis, start, count).zero_()
            sd[name] = dst
        expand("stem.weight", 1, input0, 8)
        if model.detail_ch:
            expand("detail.weight", 1, model.detail.in_channels - 2 - model.history_age, 2)
        head = "detail_out" if model.detail_ch else "out"
        expand(head + ".weight", 0, head0, head_extra)
        expand(head + ".bias", 0, head0, head_extra)
        sd["coverage_bias"] = own["coverage_bias"]
        if strict:
            sd["_coverage_marker"] = own["_coverage_marker"]
        return model.load_state_dict(sd, strict=strict)
    return model.load_state_dict(sd, strict=strict)


def load_checkpoint(path, render_size, output_size, device="cpu", overrides=None):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    cfg = load_sidecar(path)
    if cfg is None:
        cfg = infer_config(sd, overrides)
    else:
        cfg = _arch(cfg)
        cfg = _overrides(cfg, overrides)
    model = build_model(cfg, render_size, output_size, device)
    try:
        load_model_state(model, sd, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint {path} does not match architecture {cfg}:\n{exc}") from exc
    if isinstance(model, FastAccumulator) and model.coverage:
        cfg["coverage_bias"] = float(model.coverage_bias)
    return model, cfg
