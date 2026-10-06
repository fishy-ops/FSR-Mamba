"""Train the learned accumulator and compare it against the FSR baseline.

    ../.venv/bin/python train.py --epochs 20 --seq-len 12

Data is pre-rendered and cached, because the analytic scenes are not free to
evaluate and we want many passes over the same sequences.

Training unrolls frame by frame with the warp inside the loop and
backpropagates through time -- there is no parallel scan here, for the reason
laid out in `mamba.py`'s docstring. `--bptt` truncates the window so memory
stays bounded; state carries across windows detached.

The loss has two terms:

  L1 to ground truth                     -- per-frame fidelity
  motion-compensated frame-to-frame diff -- temporal stability

The second term is the one that matters. Optimising L1 alone produces a model
that scores well per frame and shimmers in motion, which is precisely the
failure the whole project is meant to avoid.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import time

import torch
import torch.nn.functional as F

from fsrmamba.augment import augment_dither, augment_exposure, exposure_range, luma_range
from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.mamba import MambaAccumulator, MambaState
from fsrmamba.fast import FastAccumulator
from fsrmamba.phase import PhaseAccumulator
from fsrmamba import config
from fsrmamba.metrics import (
    edge_gradient_l1, freq_l1, gradient_l1, perceptual_l1, psnr, ssim, ssim_map_mean,
    temporal_instability,
    temporal_deviation,
    fdl,
)
from fsrmamba.synth import halton_jitter, random_scene
from fsrmamba.engine_data import (
    crop_sequence, list_scenes, load_engine_scene, load_extra_engine_scenes,
)
from fsrmamba.cropbank import load_bank, sample_from_window
from fsrmamba.gan import TemporalPatchDiscriminator, d_hinge_loss, g_adv_loss


def render_sequence(scene, n_frames, render_size, out_size, gt_ss):
    """Pre-render one sequence into a list of per-frame dicts."""
    frames = []
    for i in range(n_frames):
        j = halton_jitter(i)
        lr = scene.render(i, render_size, jitter=j, supersample=1)
        aux = scene.render(i, out_size, jitter=(0.0, 0.0), supersample=1)
        gt = scene.render(i, out_size, jitter=(0.0, 0.0), supersample=gt_ss)
        frames.append(
            {
                "lr": lr.color,
                "mv": aux.mv,
                "depth": aux.depth,
                "gt": gt.color,
                "jitter": j,
            }
        )
    return frames


def warp_prev(img, mv):
    uv_grid = None
    h, w, _ = img.shape
    ys = (torch.arange(h, device=img.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=img.device, dtype=torch.float32) + 0.5) / w
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    uv = torch.stack((gx, gy), dim=-1) + mv
    return F.grid_sample(
        img.permute(2, 0, 1).unsqueeze(0),
        (uv * 2 - 1).unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)


def flicker_loss(out, previous, gt, previous_gt, mv, reset, mask=None, previous_mask=None, halo=0):
    """Penalise excess motion-compensated change over the reference's change."""
    h, w = out.shape[:2]
    if mv.shape[:2] != (h, w):
        mv = F.interpolate(mv.permute(2, 0, 1)[None], (h, w), mode="bilinear",
                           align_corners=False)[0].permute(1, 2, 0)
    excess = ((out - warp_prev(previous, mv)).abs()
              - (gt - warp_prev(previous_gt, mv)).abs()).clamp(min=0)
    valid = torch.ones(h, w, device=out.device, dtype=out.dtype)
    if reset is not None:
        r = torch.as_tensor(reset, device=out.device).float()
        if r.numel() == 1:
            valid = valid * (r == 0).to(out.dtype)
        else:
            r = F.interpolate(r.reshape(1, 1, *r.shape[-2:]), (h, w), mode="nearest")[0, 0]
            valid = valid * (r == 0).to(out.dtype)
    if mask is not None:
        valid = valid * (mask > .5)
    if previous_mask is not None:
        valid = valid * (warp_prev(previous_mask[..., None].float(), mv)[..., 0] >= 1 - 1e-6)
    y, x = torch.meshgrid((torch.arange(h, device=out.device) + .5) / h,
                          (torch.arange(w, device=out.device) + .5) / w, indexing="ij")
    uv = torch.stack((x, y), -1) + mv
    valid = valid * ((uv >= 0) & (uv < 1)).all(-1)
    excess, valid = loss_core(excess, halo), loss_core(valid, halo)
    return (excess * valid[..., None]).sum() / (valid.sum() * out.shape[-1]).clamp(min=1)


def loss_core(x, halo):
    """Exclude an output-pixel halo from an (H,W[,C]) loss input."""
    if halo < 0 or 2 * halo >= min(x.shape[:2]):
        raise ValueError("loss halo must be nonnegative and leave a nonempty core")
    return x[halo:-halo, halo:-halo] if halo else x


def edge_l1(out, gt, halo=0, mask=None):
    """Mean RGB L1 over valid GT edges, with neighbourhoods evaluated before cropping."""
    edges = loss_core(luma_range(gt) > .08, halo).to(out.dtype)
    if mask is not None:
        edges = edges * loss_core(mask, halo)
    error = (loss_core(out, halo) - loss_core(gt, halo)).abs().mean(-1)
    return (error * edges).sum() / edges.sum().clamp(min=1)


class CoverageAdam(torch.optim.Adam):
    """Scale selected Adam updates, preserving moments and the scheduled base LR."""

    def __init__(self, model, lr, multiplier):
        super().__init__(model.parameters(), lr=lr)
        self.multiplier = multiplier
        stop = model.stem.in_channels - model.n_state - 4 * model.history_age
        self.slices = [(model.stem.weight, (slice(None), slice(stop-8, stop)))]
        head = model.detail_out if model.detail_ch else model.out
        rep = 1 if model.detail_ch else 4
        rows = slice(model._cov_offset * rep, (model._cov_offset + model.p) * rep)
        self.slices.extend(((head.weight, rows), (head.bias, rows)))
        if model.detail_ch:
            end = model.detail.in_channels - int(model.history_age)
            self.slices.append((model.detail.weight, (slice(None), slice(end-2, end))))

    @torch.no_grad()
    def step(self, closure=None):
        previous = [p[index].clone() for p, index in self.slices]
        loss = super().step(closure)
        for (p, index), before in zip(self.slices, previous):
            p[index].add_(p[index] - before, alpha=self.multiplier - 1)
        return loss


def training_optimizer(model, lr, coverage_lr_mult=1):
    if not math.isfinite(coverage_lr_mult) or coverage_lr_mult <= 0:
        raise ValueError("coverage LR multiplier must be positive and finite")
    if coverage_lr_mult == 1:
        return torch.optim.Adam(model.parameters(), lr=lr)
    if not isinstance(model, FastAccumulator) or not model.coverage:
        raise ValueError("coverage LR multiplier requires --arch fast --fast-coverage")
    return CoverageAdam(model, lr, coverage_lr_mult)


def balance_core(model, halo):
    """Restrict per-pixel expert usage to the same core as reconstruction losses."""
    weights = getattr(model, "_last_weights", None)
    if not halo or weights is None:
        return model.load_balance_loss()
    weights = loss_core(weights[0].permute(1, 2, 0), halo)
    k = weights.shape[-1]
    winners = weights.argmax(dim=-1).flatten()
    frac = torch.bincount(winners, minlength=k).to(weights.dtype) / winners.numel()
    return k * (frac * weights.mean(dim=(0, 1))).sum()


def evaluate(model, frames, out_size, state_channels):
    """Run a model over a sequence and return (psnr, ssim, temporal instability)."""
    state = (new_state(model, frames[0]["lr"].device)
             if isinstance(model, MambaAccumulator) or hasattr(model, "init_state")
             else FSRState.zeros(out_size))
    prev_out, prev_depth, prev_gt = None, None, None
    tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0}
    n_ti = 0
    with torch.no_grad():
        for f in frames:
            out, state = model(
                state, f["lr"], f["mv"], f["depth"], f["jitter"], prev_depth_hr=prev_depth
            )
            prev_depth = f["depth"]
            tot["psnr"] += psnr(out, f["gt"])
            tot["ssim"] += ssim(out, f["gt"])
            if prev_out is not None:
                tot["ti"] += temporal_instability(out, prev_out, f["mv"])
                dev, _ = temporal_deviation(out, prev_out, f["gt"], prev_gt, f["mv"])
                tot["dev"] += abs(dev)
                n_ti += 1
            prev_out, prev_gt = out, f["gt"]
    n = len(frames)
    return (tot["psnr"] / n, tot["ssim"] / n,
            tot["ti"] / max(1, n_ti), tot["dev"] / max(1, n_ti))


def evaluate_captured(frames) -> tuple[float, float, float, float]:
    """Baseline metrics from the *captured* FSR output vs ground truth.

    In engine mode the baseline is the real FSR 3.1.4 output we captured, not the
    ported accumulator -- so this just scores frames["fsr_out"] against the GT.
    """
    tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0}
    n_ti = 0
    prev = prev_gt = None
    for f in frames:
        tot["psnr"] += psnr(f["fsr_out"], f["gt"])
        tot["ssim"] += ssim(f["fsr_out"], f["gt"])
        if prev is not None and prev_gt is not None:
            tot["ti"] += temporal_instability(f["fsr_out"], prev, f["mv"])
            dev, _ = temporal_deviation(f["fsr_out"], prev, f["gt"], prev_gt, f["mv"])
            tot["dev"] += abs(dev)
            n_ti += 1
        prev, prev_gt = f["fsr_out"], f["gt"]
    n = len(frames)
    return (tot["psnr"] / n, tot["ssim"] / n,
            tot["ti"] / max(1, n_ti), tot["dev"] / max(1, n_ti))


def make_model(args, render_size, out_size, dev):
    """The accumulator selected by --arch, built for one (render, output) size."""
    if args.arch in ("hq", "kpn"):
        return config.build_model(args, render_size, out_size, dev).train()
    if args.arch == "phase":
        return PhaseAccumulator(render_size, out_size,
                                widths=tuple(int(v) for v in args.fast_widths.split(",")),
                                depths=tuple(int(v) for v in args.fast_depths.split(",")),
                                n_state=args.fast_state, film=not args.fast_no_film,
                                residual=args.phase_residual, box_max=args.box_max,
                                jitter_sign=args.jitter_sign, device=dev)
    if args.arch == "fast":
        return FastAccumulator(render_size, out_size,
                               widths=tuple(int(v) for v in args.fast_widths.split(",")),
                               depths=tuple(int(v) for v in args.fast_depths.split(",")),
                               n_state=args.fast_state, film=not args.fast_no_film,
                               depth_test=args.fast_depth_test, depth_soft=args.fast_depth_soft,
                               depth_soft_osc=args.fast_depth_soft_osc,
                               stem_kernel=args.fast_stem_kernel,
                               learned_clamp=args.fast_learned_clamp,
                               resolve=args.fast_resolve,
                               detail_ch=args.fast_detail,
                               hist_filter=args.fast_hist_filter,
                               accum=args.fast_accum,
                               conf_motion=args.fast_conf_motion,
                               hist_residual=args.fast_hist_residual,
                               nearest_sample=args.fast_nearest_sample,
                               conf_consistent=args.fast_conf_consistent,
                               carry_raw=args.fast_carry_raw,
                               base_gate=args.fast_base_gate,
                               mv_dilate=args.fast_mv_dilate, depth_dilate=args.fast_depth_dilate,
                               thin_lock=args.fast_thin_lock, coverage=args.fast_coverage,
                               coverage_bias=getattr(args, "coverage_bias", -3.0),
                               history_age=getattr(args, "fast_history_age", False),
                               reset_lanczos=args.fast_reset_lanczos,
                               jitter_sign=args.jitter_sign, device=dev)
    return MambaAccumulator(render_size, out_size, state_channels=args.state_channels,
                            feature_channels=args.feature_channels,
                            num_experts=args.num_experts, sharpen=args.sharpen,
                            rectify=args.rectify, box_max=args.box_max,
                            learned_resolve=args.learned_resolve,
                            swin_resolve=args.swin_resolve, swin_dim=args.swin_dim,
                            lr_upsampler=args.lr_upsampler, lr_dim=args.lr_dim,
                            lr_film=args.lr_film, separable=args.separable, unet=args.unet,
                            kernel_predict=args.kernel_predict,
                            robust_disocc=args.robust_disocc,
                            lock_feature=args.lock_feature,
                            alpha_scale=args.alpha_scale,
                            encoder_depth=args.encoder_depth,
                            jitter_sign=args.jitter_sign, device=dev)


def new_state(m, dev):
    """Fresh recurrent state for either accumulator."""
    if hasattr(m, "init_state"):
        return m.init_state(dev)
    return MambaState.zeros(m.output_size, m.state_channels, device=dev)


def warmup_length(length, count, cold_prob, rng):
    """Choose a warm prefix without drawing randomness when warm-up is disabled."""
    if count < 0 or count >= length:
        raise ValueError("warmup frames must leave at least one training frame per sequence")
    return 0 if count and rng.random() < cold_prob else count


def warm_sequence(model, teacher, frames, count, dev):
    """Start both recurrences on the same prefix, retaining full-frame history."""
    state = new_state(model, dev)
    t_state = new_state(teacher, dev) if teacher is not None else None
    prev_out = prev_depth = prev_gt = None
    with torch.no_grad():
        for f in frames[:count]:
            out, state = model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                               prev_depth_hr=prev_depth)
            if teacher is not None:
                _, t_state = teacher(t_state, f["lr"], f["mv"], f["depth"], f["jitter"],
                                     prev_depth_hr=prev_depth)
            prev_out, prev_depth, prev_gt = out, f["depth"], f["gt"]
    return state, t_state, prev_out, prev_depth, prev_gt


def plan_epoch(main_count, extra_count, per_epoch, rng):
    """Shuffle all main sequences with a capped sample of distinct extras."""
    if min(main_count, extra_count, per_epoch) < 0:
        raise ValueError("epoch sequence counts must be nonnegative")
    sources = [("main", i) for i in range(main_count)]
    count = min(extra_count, per_epoch)
    if count:
        sources.extend(("extra", i) for i in rng.sample(range(extra_count), count))
    rng.shuffle(sources)
    return sources


def sequence_window(frames, count, rng):
    """Draw a contiguous temporal window, retaining shorter sequences whole."""
    if count < 1:
        raise ValueError("extra frames must be positive")
    if len(frames) <= count:
        return frames
    start = rng.randrange(len(frames) - count + 1)
    return frames[start:start + count]


def train_engine(args) -> None:
    """Train on captured engine data: (LR, motion, depth) -> ground truth.

    Holds out whole scenes for validation (same discipline as the synthetic
    path). Trains on fixed random crops -- full 1080p BPTT will not fit in 8 GB,
    and crop_sequence rescales the motion vectors to crop-local UV. The baseline
    is the captured FSR output, scored on the same held-out crops.
    """
    dev = torch.device(args.device)
    crop_r = args.crop
    crop_o = crop_r * int(args.scale)
    render_size = (crop_r, crop_r)
    out_size = (crop_o, crop_o)
    if args.warmup_frames < 0:
        raise ValueError("warmup frames must be nonnegative")
    if args.loss_halo < 0 or crop_o - 2 * args.loss_halo < 2:
        raise ValueError("loss halo must leave at least two output pixels per axis")
    if not 0 <= args.cold_start_prob <= 1 or not 0 <= args.edge_crop_prob <= 1:
        raise ValueError("cold-start and edge-crop probabilities must be in [0,1]")

    scenes = list_scenes(args.engine_data)
    if len(scenes) < 2:
        raise SystemExit(f"need >=2 scenes in {args.engine_data}, found {scenes}")
    n_val = max(1, args.val_scenes)
    val_names, train_names = scenes[:n_val], scenes[n_val:]
    # Test scenes are never loaded here: not trained on, not used to pick a checkpoint.
    test_names = train_names[len(train_names) - args.test_scenes:] if args.test_scenes else []
    train_names = train_names[:len(train_names) - len(test_names)]
    print(f"engine data: {len(train_names)} train / {len(val_names)} val scenes"
          + (f" / held-out test {test_names}" if test_names else ""))
    print(f"  train: {train_names}")
    print(f"  val:   {val_names}")
    print(f"crop render {crop_r}x{crop_r} -> out {crop_o}x{crop_o}")
    extra_per_epoch = ((len(train_names) + 2) // 3 if args.extra_per_epoch is None
                       else args.extra_per_epoch)
    if args.extra_engine_data and (args.extra_frames < 1 or extra_per_epoch < 0):
        raise SystemExit("--extra-frames must be positive and --extra-per-epoch nonnegative")

    # Full sequences stay on CPU; crops move to the GPU per use (1080p x all
    # scenes will not fit in VRAM, but a crop is tiny).
    t0 = time.time()
    # half=True: ten 96-frame 1080p scenes is ~59 GB in float32 and will not fit in
    # 64 GB of RAM. fp16 halves it to ~30 GB, and (since half implies mmap) the
    # frames are now file-backed, so the 70 GB multi-path set no longer has to fit in
    # RAM at all -- cold pages are simply evicted.
    #
    # RAM stopped being the binding constraint once mmap landed; the DISK did. The
    # captures sit on a 7200 RPM HDD that reads this pattern at ~19 MB/s, and the
    # per-epoch loop below re-crops every frame of every scene, so a bare mmap run
    # spends ~20 min/epoch waiting on seeks. --crop-bank pays that read once into a
    # RAM-resident bank of large windows; see fsrmamba/cropbank.py.
    bank = None
    if args.crop_bank:
        bank = load_bank(args.crop_bank)
        if args.edge_crop_prob > 0:
            print("  crop bank: --edge-crop-prob ignored (window sampling)")
        got = sorted({e["scene"] for e in bank})
        # Leakage is the fatal direction: a validation scene inside the bank would
        # inflate every number in the run. A bank that is merely *missing* scenes is
        # legitimate -- load_bank drops entries captured at the wrong scale -- so
        # that is reported, not fatal.
        leaked = sorted(set(got) & set(val_names))
        if leaked:
            raise SystemExit(f"crop bank contains validation scenes {leaked}; "
                             f"training on it would leak. Rebuild the bank.")
        missing = sorted(set(train_names) - set(got))
        if missing:
            print(f"  crop bank: NOT training on {missing} (absent from the bank)")
        extra = sorted(set(got) - set(train_names))
        if extra:
            raise SystemExit(f"crop bank has scenes outside this run's training split: "
                             f"{extra}. Rebuild the bank.")
        print(f"  crop bank: {len(bank)} windows over {len(got)} scenes, "
              f"{bank[0]['lr'].shape[0]} frames, window {bank[0]['ch']}px render")
        train_full = []
    else:
        train_full = [load_engine_scene(f"{args.engine_data}/{s}", device="cpu", half=True)
                      for s in train_names]
    val_full = [load_engine_scene(f"{args.engine_data}/{s}", device="cpu", half=True)
                for s in val_names]
    extra_full = []
    if args.extra_engine_data:
        extra_full = [frames for _, frames in load_extra_engine_scenes(
            args.extra_engine_data, train_names, val_names, test_names, args.scale, args.bptt)]
    print(f"  loaded in {time.time()-t0:.1f}s")

    def to_dev(seq):
        return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in f.items()} for f in seq]

    # Multiple seeded crops per val scene -> a robust average instead of one
    # lucky crop. Kept on CPU; each is moved to the GPU only while it is scored.
    val_crops = []
    for i, s in enumerate(val_full):
        for k in range(args.eval_crops):
            val_crops.append(crop_sequence(s, crop_r, int(args.scale),
                                           random.Random(1000 + i * 1000 + k)))
    print(f"eval: {args.eval_crops} crops x {len(val_full)} val scenes = {len(val_crops)} sequences")

    def _score(frames_cpu, run_model):
        """Score one crop sequence. run_model=None -> captured FSR baseline."""
        frames = to_dev(frames_cpu)
        state = new_state(run_model, dev) if run_model else None
        prev_out, prev_depth, prev_gt = None, None, None
        tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0}
        n_ti = 0
        for f in frames:
            if run_model is not None:
                out, state = run_model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                                       prev_depth_hr=prev_depth)
                prev_depth = f["depth"]
            else:
                out = f["fsr_out"]
            tot["psnr"] += psnr(out, f["gt"])
            tot["ssim"] += ssim(out, f["gt"])
            if prev_out is not None:
                tot["ti"] += temporal_instability(out, prev_out, f["mv"])
                # Distance from the ground truth's OWN instability; raw instability is
                # minimised by blur (see metrics.temporal_deviation).
                d, _ = temporal_deviation(out, prev_out, f["gt"], prev_gt, f["mv"])
                tot["dev"] += abs(d)
                n_ti += 1
            prev_out, prev_gt = out, f["gt"]
        n = len(frames)
        return (tot["psnr"] / n, tot["ssim"] / n, tot["ti"] / max(1, n_ti),
                tot["dev"] / max(1, n_ti))

    def eval_model():
        model.eval()
        with torch.no_grad():
            rs = [_score(c, model) for c in val_crops]
        return tuple(sum(v) / len(rs) for v in zip(*rs))

    def _full_frame(f, uh, uw):
        """One whole frame with mv/depth upsampled to output res (what the model
        consumes). Full frame => no motion rescale, unlike crop_sequence.

        Deliberately ONE frame, not the sequence. Materialising a whole clip costs
        ~80 MB/frame in float32 once mv and depth are at 1080p, so a 120-frame scene
        is ~9 GB in RAM and ~3.7 GB more if it is pushed to an 8 GB GPU -- which is
        what the previous version did, and it pinned VRAM at 7.7/8.2 GB with the GPU
        at 100% making no progress. Streaming frame by frame is the documented fix
        (measured elsewhere at 0.75 s/frame versus 33 s/frame when preloaded).
        """
        from fsrmamba.engine_data import _upsample_to
        # .float(): sequences are held in fp16 to fit in RAM (see load above).
        return {
            "lr": f["lr"].float(),
            "mv": _upsample_to(f["mv"].float(), uh, uw, "bilinear"),
            "depth": _upsample_to(f["depth"].float(), uh, uw, "nearest"),
            "gt": f["gt"].float(), "fsr_out": f["fsr_out"].float(), "jitter": f["jitter"],
        }

    def eval_full():
        """The honest bar: full-frame metrics over held-out scenes (crops hide
        the border/disocclusion regions where the model is weakest).

        The FSR front-end (`_resolve`) precomputes size-specific sampling grids,
        so a crop-trained model can't run full frames directly -- build a
        full-res instance and copy the (size-independent) learned weights in.
        """
        fr = tuple(val_full[0][0]["lr"].shape[:2])
        fo = tuple(val_full[0][0]["gt"].shape[:2])
        fm = make_model(args, fr, fo, dev)
        fm.load_state_dict(model.state_dict())
        fm.eval()

        def score_full(scene, use_model):
            uh, uw, _ = scene[0]["gt"].shape
            state = new_state(fm, dev) if use_model else None
            prev_out, prev_depth, prev_gt = None, None, None
            tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0}
            n_ti = 0
            for raw in scene:
                # Built and moved one frame at a time; see _full_frame.
                f = {k: (v.to(dev) if torch.is_tensor(v) else v)
                     for k, v in _full_frame(raw, uh, uw).items()}
                if use_model:
                    out, state = fm(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                                    prev_depth_hr=prev_depth)
                    prev_depth = f["depth"]
                else:
                    out = f["fsr_out"]
                tot["psnr"] += psnr(out, f["gt"])
                tot["ssim"] += ssim(out, f["gt"])
                if prev_out is not None:
                    tot["ti"] += temporal_instability(out, prev_out, f["mv"])
                    d, _ = temporal_deviation(out, prev_out, f["gt"], prev_gt, f["mv"])
                    tot["dev"] += abs(d)
                    n_ti += 1
                prev_out, prev_gt = out, f["gt"]
            n = len(scene)
            return (tot["psnr"] / n, tot["ssim"] / n, tot["ti"] / max(1, n_ti),
                    tot["dev"] / max(1, n_ti))

        with torch.no_grad():
            lm = [score_full(s, True) for s in val_full]
            lb = [score_full(s, False) for s in val_full]
        return (tuple(sum(v) / len(lm) for v in zip(*lm)),
                tuple(sum(v) / len(lb) for v in zip(*lb)))

    with torch.no_grad():
        rs = [_score(c, None) for c in val_crops]
    b_psnr, b_ssim, b_ti, b_dev = (sum(v) / len(rs) for v in zip(*rs))
    print(f"\nFSR (captured) PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}  "
          f"|dev| {b_dev:.5f}")

    model = make_model(args, render_size, out_size, dev)
    if args.init_from:
        # Warm start. NOTE: this loads weights only -- train.py never saved optimizer
        # moments, the LR-scheduler position, or the epoch, so this is not a true
        # resume and Adam restarts its moment estimates. Pair it with a lower --lr
        # than the original run, or the fresh warmup will damage converged weights.
        # strict=False so a warm start can cross an architecture change: the shared
        # trunk (encoder, SSM, alpha heads) transfers and only genuinely new modules
        # start fresh. With strict=True this raised on any added module, which meant
        # every architectural experiment had to retrain from scratch even though the
        # feature extractor it needed had already been trained.
        sd = torch.load(args.init_from, map_location=dev)
        incompat = config.load_model_state(model, sd, strict=False)
        own = set(model.state_dict().keys())
        transferred = len(own) - len(incompat.missing_keys)
        print(f"warm-started from {args.init_from}: {transferred}/{len(own)} tensors "
              f"transferred (weights only; optimizer moments and scheduler position are "
              f"NOT saved, so Adam restarts -- use a lower --lr than the source run)")
        if incompat.missing_keys:
            print(f"  fresh (not in source): {sorted(incompat.missing_keys)[:8]}"
                  f"{' ...' if len(incompat.missing_keys) > 8 else ''}")
        if incompat.unexpected_keys:
            print(f"  ignored (not in target): {sorted(incompat.unexpected_keys)[:8]}"
                  f"{' ...' if len(incompat.unexpected_keys) > 8 else ''}")
    print(f"learned model  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{args.num_experts} expert(s), sharpen={args.sharpen}, rectify={args.rectify}\n")
    if isinstance(model, FastAccumulator) and model.coverage:
        args.coverage_bias = float(model.coverage_bias)
    # --- distillation teacher -------------------------------------------------
    teacher = None
    if args.distill_from:
        # Any architecture, with its own jitter convention: rebuilt from the sidecar or the
        # tensors (the teacher may predate the jitter-sign fix or carry the corrected sign).
        teacher, _tcfg = config.load_checkpoint(args.distill_from, render_size, out_size, dev)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"distilling from {args.distill_from}: teacher has "
              f"{sum(q.numel() for q in teacher.parameters()):,} params vs student "
              f"{sum(q.numel() for q in model.parameters()):,} "
              f"({sum(q.numel() for q in teacher.parameters())/max(1,sum(q.numel() for q in model.parameters())):.0f}x)")

    disc = disc_opt = None
    if args.gan_weight > 0:
        disc = TemporalPatchDiscriminator().to(dev)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
        print(f"adversarial: temporal PatchGAN, {sum(q.numel() for q in disc.parameters()):,} "
              f"params (training only -- discarded at inference, so zero cost in the "
              f"real-time budget), weight {args.gan_weight}, warmup {args.gan_warmup} epochs")

    if args.select_full:
        # Crop means are dominated by a few easy crops (a sky-only crop sits near 60 dB), so a
        # checkpoint picked on crops is not the best full-frame one. Select on whole frames.
        def eval_model():
            return eval_full()[0]
        b_psnr, b_ssim, b_ti, b_dev = eval_full()[1]
        print(f"selecting on FULL frames: FSR PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  "
              f"temporal {b_ti:.5f}  |dev| {b_dev:.5f}")

    opt = training_optimizer(model, args.lr, args.coverage_lr_mult)
    # Linear warmup then cosine. A bigger model at full LR from step 0 can diverge
    # to a degenerate constant output early and never recover -- the warmup avoids
    # that.
    warmup = max(1, args.epochs // 15)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.02, total_iters=warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup),
                                                    eta_min=args.lr * 0.05)],
        milestones=[warmup],
    )
    # Exponential moving average of the weights. Batch size is 1, so the raw weights jitter
    # from step to step; the average is what gets evaluated and saved.
    ema = ({k: v.detach().clone() for k, v in model.state_dict().items()}
           if args.ema > 0 else None)
    best_score = -1e9
    best_metrics = (0.0, 0.0, 0.0, 0.0)

    accum_i = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        d_acc, g_acc, adv_n = 0.0, 0.0, 0
        # A fresh crop every epoch is the augmentation. With a bank the crop is drawn
        # from a stored window instead of from disk, which is the only difference.
        if bank is not None:
            sources = [(lambda e=e: sample_from_window(e, crop_r, int(args.scale),
                                                       random.Random(), args.edge_bias,
                                                       frames=(args.frames_per_seq or None),
                                                       augment_rng=(random if args.augment else None)))
                       for e in bank]
        else:
            sources = [(lambda f=f: crop_sequence(f, crop_r, int(args.scale),
                                                  edge_bias=args.edge_bias,
                                                  augment_rng=(random if args.augment else None),
                                                  edge_prob=args.edge_crop_prob,
                                                  static_pan=args.static_pan))
                       for f in train_full]
        epoch_plan = plan_epoch(len(sources), len(extra_full), extra_per_epoch, random)
        if args.extra_engine_data:
            used = sum(kind == "extra" for kind, _ in epoch_plan)
            print(f"epoch {epoch:3d}: {used} extra sequences used")
        for kind, index in epoch_plan:
            seq_is_extra = kind == "extra"
            if seq_is_extra:
                frames = sequence_window(extra_full[index], args.extra_frames, random)
                seq = to_dev(crop_sequence(frames, crop_r, int(args.scale),
                                          edge_bias=args.edge_bias,
                                          augment_rng=(random if args.augment else None),
                                          edge_prob=args.edge_crop_prob,
                                          static_pan=args.static_pan))
            else:
                seq = to_dev(sources[index]())
            seq = augment_exposure(seq, args.exposure_aug, random)
            seq = augment_dither(seq, args.dither_aug, random)
            detail_losses = not (seq_is_extra and args.extra_lowpass)
            warm = warmup_length(len(seq), args.warmup_frames, args.cold_start_prob, random)
            state, t_state, prev_out, prev_depth, prev_gt = warm_sequence(
                model, teacher, seq, warm, dev)
            prev_flicker = prev_out
            prev_mask = seq[warm - 1].get("mask") if warm else None
            t_prev_depth = prev_depth
            for start in range(warm, len(seq), args.bptt):
                window = seq[start:start + args.bptt]
                if not window:
                    continue
                if accum_i % args.accum_steps == 0:
                    opt.zero_grad()
                loss = torch.zeros((), device=dev)
                d_queue = []
                for fi, f in enumerate(window):
                    loss_in = loss
                    out, state = model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                                       prev_depth_hr=prev_depth)
                    prev_depth = f["depth"]
                    if prev_flicker is not None and args.flicker_weight > 0:
                        po = prev_flicker if args.temporal_through else prev_flicker.detach()
                        loss = loss + args.flicker_weight * flicker_loss(
                            out, po, f["gt"], prev_gt, f.get("gt_mv", f["mv"]),
                            getattr(model, "_last_reset", f.get("reset")),
                            f.get("mask"), prev_mask, args.loss_halo)
                    prev_flicker = out
                    if detail_losses and args.edge_loss_weight > 0:
                        loss = loss + args.edge_loss_weight * edge_l1(
                            out, f["gt"], args.loss_halo, f.get("mask"))
                    if "mask" in f:   # no loss and no gradient where the reference is not valid
                        m = f["mask"].unsqueeze(-1)
                        out = out * m + f["gt"] * (1 - m)
                    core_out, core_gt = loss_core(out, args.loss_halo), loss_core(f["gt"], args.loss_halo)
                    loss = loss + F.l1_loss(core_out, core_gt)
                    if hasattr(model, "_last_carry"):
                        carry_rgb = model._last_carry[0].permute(1, 2, 0)
                        loss = loss + args.carry_aux_weight * F.l1_loss(
                            loss_core(carry_rgb, args.loss_halo), core_gt)
                    if args.logmse_weight > 0:
                        # The reported PSNR is a mean of per-frame dB, so a frame at 45 dB
                        # counts as much as one at 25 dB. L1 weights frames by absolute error
                        # and all but ignores the easy (static, converged) ones; log-MSE is
                        # the reported quantity itself.
                        mse = F.mse_loss(core_out.clamp(0, 1), core_gt.clamp(0, 1))
                        loss = loss + args.logmse_weight * torch.log10(mse + 1e-8)
                    if detail_losses and args.ssim_weight > 0:
                        loss = loss + args.ssim_weight * (1.0 - ssim_map_mean(core_out, core_gt))
                    if detail_losses and args.grad_weight > 0:
                        loss = loss + args.grad_weight * gradient_l1(core_out, core_gt)
                    if detail_losses and args.edge_grad_weight > 0:
                        loss = loss + args.edge_grad_weight * edge_gradient_l1(core_out, core_gt)
                    if args.alpha_penalty > 0:
                        # Penalise leaning on the current frame, EXCEPT where the pixel
                        # is genuinely disoccluded (there the resolve must win). Unlike
                        # scaling alpha down (v23), this is a real cost the network
                        # cannot cancel by raising its pre-activation -- v23's effective
                        # alpha came out at 0.3558 vs v20's 0.2652 despite a 0.4x scale.
                        a, r = model._last_alpha, model._last_reset
                        # HINGE, not a linear penalty. A linear penalty has constant
                        # gradient, which swamps L1's weak gradient on alpha and drives
                        # it to the floor no matter the weight -- measured: penalty 0.15
                        # and 0.05 both landed alpha at ~0.025, so it acted as a switch
                        # rather than a dial. Penalising only the excess above a target
                        # makes the cost vanish once alpha reaches it, so alpha settles
                        # AT the target and --alpha-target becomes the real knob.
                        excess = (a - args.alpha_target).clamp(min=0.0)
                        penalty = (excess * (1.0 - r))[0].permute(1, 2, 0)
                        loss = loss + args.alpha_penalty * loss_core(penalty, args.loss_halo).mean()
                    if teacher is not None:
                        with torch.no_grad():
                            # The teacher sees the same exposure-adjusted LR; its target
                            # is already in that space and must not receive a second gain.
                            t_out, t_state = teacher(t_state, f["lr"], f["mv"], f["depth"],
                                                     f["jitter"], prev_depth_hr=t_prev_depth)
                            t_prev_depth = f["depth"]
                        loss = loss + args.distill_weight * F.l1_loss(core_out, loss_core(t_out, args.loss_halo))
                    if args.freq_weight > 0:
                        loss = loss + args.freq_weight * freq_l1(core_out, core_gt)
                    if args.fdl_weight > 0:
                        loss = loss + args.fdl_weight * fdl(core_out, core_gt)
                    if args.perceptual_weight > 0:
                        loss = loss + args.perceptual_weight * perceptual_l1(core_out, core_gt)
                    if prev_out is not None:
                        po = prev_out if args.temporal_through else prev_out.detach()
                        d_out = core_out - loss_core(warp_prev(po, f["mv"]), args.loss_halo)
                        d_gt = core_gt - loss_core(warp_prev(prev_gt, f["mv"]), args.loss_halo)
                        loss = loss + args.temporal_weight * F.l1_loss(d_out, d_gt)

                    # --- adversarial ---------------------------------------------
                    # Needs a previous frame: the discriminator judges a (current,
                    # motion-warped previous) PAIR, so that texture must be plausible
                    # *and* stable under the true motion. A per-frame discriminator
                    # would reward any locally-real texture, and invented detail fed
                    # into the recurrence compounds into shimmer.
                    if disc is not None and epoch >= args.gan_warmup and prev_out is not None:
                        pw_f = loss_core(warp_prev(prev_out.detach(), f["mv"]), args.loss_halo).permute(2, 0, 1)[None]
                        pw_r = loss_core(warp_prev(prev_gt, f["mv"]), args.loss_halo).permute(2, 0, 1)[None]
                        cur_f = core_out.permute(2, 0, 1)[None]
                        cur_r = core_gt.permute(2, 0, 1)[None]
                        # G term uses the CURRENT discriminator. The D update must NOT
                        # happen here: this loss is accumulated across the whole BPTT
                        # window and only backpropagated once the window ends, so stepping
                        # D now would mutate parameters the pending graph still needs
                        # ("variable needed for gradient computation has been modified").
                        # The (detached) pair is queued and D is trained after the
                        # generator's backward pass instead.
                        g_adv = g_adv_loss(disc(cur_f, pw_f))
                        loss = loss + args.gan_weight * g_adv
                        g_acc += float(g_adv.detach())
                        adv_n += 1
                        d_queue.append((cur_r.detach(), pw_r.detach(),
                                        cur_f.detach(), pw_f.detach()))
                    if args.num_experts > 1 and args.balance_weight > 0:
                        loss = loss + args.balance_weight * balance_core(model, args.loss_halo)
                    if args.cold_weight and warm == 0:
                        # Frames right after a reset have no history and are scored like any
                        # other; weight them up (4x for 0-3, 2x for 4-7, scaled to keep the
                        # clip's mean weight at 1).
                        age = start + fi
                        wf = (4.0 if age < 4 else 2.0 if age < 8 else 1.0) * 0.75
                        loss = loss_in + wf * (loss - loss_in)
                    prev_out, prev_gt = out, f["gt"]
                    prev_mask = f.get("mask")
                loss = loss / len(window)
                # Gradient accumulation = mini-batching without touching the model.
                # We train one crop-sequence at a time (effective batch 1), which makes
                # the gradient very noisy -- visible as ~1 dB swings between adjacent
                # eval points. Published neural-supersampling work uses batches of 8
                # clips; accumulating N windows before stepping buys the same variance
                # reduction with no change to the recurrent forward pass.
                (loss / args.accum_steps).backward()
                accum_i += 1
                if accum_i % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if ema is not None:
                        with torch.no_grad():
                            for k, v in model.state_dict().items():
                                if v.dtype.is_floating_point and k != "coverage_bias":
                                    ema[k].mul_(args.ema).add_(v, alpha=1.0 - args.ema)
                epoch_loss += loss.item()

                # Discriminator updates, now that the generator's graph is consumed.
                if disc is not None and d_queue:
                    for cr, pr, cf, pf in d_queue:
                        disc_opt.zero_grad()
                        d_loss = d_hinge_loss(disc(cr, pr), disc(cf, pf))
                        d_loss.backward()
                        disc_opt.step()
                        d_acc += float(d_loss.detach())
                    d_queue.clear()

                state = state.detach()
                prev_flicker = prev_flicker.detach()
                if t_state is not None:
                    t_state = t_state.detach()
                prev_out = prev_out.detach()
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            raw_sd = None
            if ema is not None:
                raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            p, s, ti, dv = eval_model()
            # Temporal is judged on |deviation from ground truth|, not raw instability.
            wins = int(p > b_psnr) + int(s > b_ssim) + int(dv < b_dev)
            flag = "  <-- BEATS FSR on all 3" if wins == 3 else (f"  <-- beats {wins}/3" if wins >= 2 else "")
            adv = ""
            if adv_n:
                dm, gm = d_acc / adv_n, g_acc / adv_n
                # Hinge-loss health. d_loss -> 0 means the discriminator has won outright
                # and the generator is getting no usable gradient. d_loss -> 2.0 means it
                # has collapsed and is labelling everything identically. Either way the
                # adversarial term has stopped doing anything useful and the run should be
                # restarted with a different --gan-weight / --gan-lr rather than left going.
                warn = ""
                if dm < 0.15:
                    warn = "  <-- D WON (no G gradient)"
                elif dm > 1.85:
                    # Hinge loss STARTS at exactly 2.0 (relu(1)+relu(1)) when D outputs
                    # zero, which is its initialisation -- so "near 2.0" is only a
                    # collapse if D has already had a fair number of updates. Flagging it
                    # immediately is a false positive on every run.
                    warn = ("  <-- D COLLAPSED" if epoch >= args.gan_warmup + 4
                            else "  (D warming up)")
                adv = f"  D {dm:.3f}  Gadv {gm:+.3f}{warn}"
            print(f"epoch {epoch:3d}  loss {epoch_loss:7.4f}  lr {sched.get_last_lr()[0]:.1e}   "
                  f"PSNR {p:6.2f} ({p-b_psnr:+.2f})  SSIM {s:.4f} ({s-b_ssim:+.4f})  "
                  f"temporal {ti:.5f}  |dev| {dv:.5f} ({dv-b_dev:+.5f}){flag}{adv}")
            # Best = (number of FSR metrics beaten), then total normalised margin.
            #
            # The old rule was `p + 30*s`, which **ignored temporal entirely**. That is not a
            # cosmetic flaw: v34 reached genuine temporal parity with FSR and the selector
            # could not see it, so a run whose whole point is temporal stability would happily
            # discard its own best checkpoint. The goal is to beat FSR on all three at once,
            # so win-count has to dominate the ranking.
            #
            # Margins are divided by a characteristic scale per metric (1 dB, 0.01 SSIM,
            # 0.001 temporal) so no single axis dominates the tiebreak just by having larger
            # raw numbers, and win-count is weighted far above any achievable margin sum.
            #
            # SSIM is weighted above the other two, deliberately. The goal of this
            # project is to *look* better than FSR, not to be more pixel-accurate than
            # it, and those are different things: PSNR is mean squared error, which a
            # slightly blurred image scores well on because being a little wrong
            # everywhere is cheap. SSIM's covariance term only pays out when the detail
            # is genuinely reconstructed, so it is the metric that tracks perceived
            # quality. We currently carry a large PSNR surplus (+4 dB at crop level) and
            # a near-zero SSIM margin, so trading PSNR for SSIM is close to free.
            margin = (3.0 * (s - b_ssim) / 0.001        # perceptual quality: dominant
                      + (b_dev - dv) / 0.001            # temporal: distance from GT's own
                      + (p - b_psnr) / 1.0)             # pixel accuracy: tiebreak only
            score = wins * 1000.0 + 2000.0 * int(s > b_ssim) + margin
            if score > best_score:
                best_score = score
                best_metrics = (p, s, ti, dv)
                if args.save:
                    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), args.save)
                    config.save_sidecar(args.save, config.model_kwargs(args))
            if raw_sd is not None:
                model.load_state_dict(raw_sd)

    if args.save:
        # Selection is on validation crops with a discontinuous SSIM bonus; keep the final
        # weights (the average, when --ema is on) so they can be scored full-frame as well.
        last = args.save[:-3] + "_last.pt" if args.save.endswith(".pt") else args.save + "_last"
        torch.save(ema if ema is not None else model.state_dict(), last)
        config.save_sidecar(last, config.model_kwargs(args))
    print("\n--- final (engine data, held-out scenes, multi-crop) ---")
    print(f"FSR (captured)   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}  "
          f"|dev| {b_dev:.5f}")
    p, s, ti, dv = eval_model()
    print(f"learned (last)   PSNR {p:6.2f}  SSIM {s:.4f}  temporal {ti:.5f}  |dev| {dv:.5f}")
    bp, bs, bti, bdv = best_metrics
    won = int(bp > b_psnr) + int(bs > b_ssim) + int(bdv < b_dev)
    print(f"learned (best)   PSNR {bp:6.2f} ({bp-b_psnr:+.2f})  SSIM {bs:.4f} ({bs-b_ssim:+.4f})  "
          f"temporal {bti:.5f}  |dev| {bdv:.5f} ({bdv-b_dev:+.5f})  -> beats FSR on {won}/3")
    if args.save:
        print(f"saved best model to {args.save}")

    # The honest, visible bar: full-frame over whole held-out scenes. Eval the
    # best checkpoint (the in-memory model is the last epoch).
    if args.save:
        model.load_state_dict(torch.load(args.save, map_location=dev))
    if args.no_full_eval:
        print("\n--- FULL-FRAME eval SKIPPED (--no-full-eval) ---")
        print("Score the saved checkpoint with scratchpad/eval_compare.py instead; it "
              "streams on CPU and finishes in minutes.")
        return
    (fp, fs, fti, fdv), (bfp, bfs, bfti, bfdv) = eval_full()
    fw = int(fp > bfp) + int(fs > bfs) + int(fdv < bfdv)
    print("\n--- FULL-FRAME (whole held-out scenes -- the real bar) ---")
    print(f"FSR (captured)   PSNR {bfp:6.2f}  SSIM {bfs:.4f}  temporal {bfti:.5f}  |dev| {bfdv:.5f}")
    print(f"learned (best)   PSNR {fp:6.2f} ({fp-bfp:+.2f})  SSIM {fs:.4f} ({fs-bfs:+.4f})  "
          f"temporal {fti:.5f}  |dev| {fdv:.5f} ({fdv-bfdv:+.5f})  -> beats FSR on {fw}/3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--bptt", type=int, default=6)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--state-channels", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dither-aug", type=float, default=0,
                    help="probability of stochastic GT coverage augmentation per clip")
    ap.add_argument("--edge-loss-weight", type=float, default=0,
                    help="extra L1 averaged over GT luma edges")
    ap.add_argument("--coverage-lr-mult", type=float, default=1,
                    help="Adam LR multiplier for coverage head and evidence input weights")
    ap.add_argument("--coverage-bias", type=float, default=-3.0,
                    help="initial coverage logit and neutral calibration for new heads")
    ap.add_argument("--temporal-weight", type=float, default=0.25)
    ap.add_argument("--flicker-weight", type=float, default=0.0,
                    help="Penalty for temporal change exceeding the motion-warped GT change")
    ap.add_argument("--carry-aux-weight", type=float, default=0.25,
                    help="weight on L1 of the raw carried colour to ground truth")
    ap.add_argument("--gt-supersample", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--train-scenes", type=int, default=6)
    ap.add_argument("--val-scenes", type=int, default=3)
    ap.add_argument("--num-experts", type=int, default=1)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--balance-weight", type=float, default=0.01)
    ap.add_argument("--engine-data", type=str, default="",
                    help="captures dir (<scene>/{fsr,gt}); trains on real engine data vs GT")
    ap.add_argument("--extra-engine-data", type=str, default="",
                    help="additional captures restricted to bases in the main training split")
    ap.add_argument("--extra-frames", type=int, default=48,
                    help="maximum frames in each random contiguous extra sequence")
    ap.add_argument("--extra-per-epoch", type=int, default=None,
                    help="extra sequences per epoch (default: ceil(main training scenes / 3))")
    ap.add_argument("--extra-lowpass", action=argparse.BooleanOptionalAction, default=True,
                    help="omit gradient, edge-gradient and SSIM losses on aliased extra GT")
    ap.add_argument("--crop", type=int, default=128, help="render-res train crop (engine mode)")
    ap.add_argument("--exposure-aug", type=exposure_range, default=None, metavar="LO,HI",
                    help="clip-wide log-uniform linear exposure gain (training only)")
    ap.add_argument("--augment", action="store_true", help="dihedral augmentation of engine training clips")
    ap.add_argument("--warmup-frames", type=int, default=0,
                    help="engine frames used only to warm recurrent states at each sequence start")
    ap.add_argument("--cold-start-prob", type=float, default=0.2,
                    help="probability of training an engine sequence without recurrent warm-up")
    ap.add_argument("--loss-halo", type=int, default=0,
                    help="exclude this many output pixels per side from engine training losses")
    ap.add_argument("--edge-crop-prob", type=float, default=0.0,
                    help="probability of drawing engine crop centres by GT gradient energy")
    ap.add_argument("--static-pan", type=int, default=0,
                    help="max synthetic pan (render px/frame) applied to frozen-world sequences")
    ap.add_argument("--eval-crops", type=int, default=8, help="seeded crops per val scene (engine mode)")
    ap.add_argument("--init-from", type=str, default="",
                    help="checkpoint to initialise weights from (warm start). Weights only "
                         "-- optimizer/scheduler state was never checkpointed, so use a "
                         "lower --lr than the original run to avoid a damaging re-warmup.")
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="accumulate gradients over N crop-sequence windows before "
                         "stepping. Effective batch size; 1 is very noisy (published "
                         "neural-supersampling work batches 8 clips).")
    ap.add_argument("--edge-grad-weight", type=float, default=0.0,
                    help="gradient L1 weighted by GROUND-TRUTH edge strength. Direct "
                         "measurement puts our deficit at edges specifically (edge "
                         "contrast 0.55-0.69 of GT vs FSR 0.82) while textured non-edge "
                         "regions already match or beat FSR, so the plain --grad-weight "
                         "term spends most of its penalty where nothing is wrong.")
    ap.add_argument("--distill-from", type=str, default="",
                    help="path to a TEACHER checkpoint. The student is trained to "
                         "reproduce the teacher's output in addition to the ground "
                         "truth. This is how you shrink a working model: the teacher's "
                         "output encodes which pixels it trusts and how it blends, which "
                         "is a far richer signal than GT alone for a small student, and "
                         "unlike --init-from it works across different architectures and "
                         "channel widths.")
    ap.add_argument("--distill-weight", type=float, default=1.0,
                    help="weight on the teacher-matching term.")
    ap.add_argument("--gan-weight", type=float, default=0.0, help="weight on the adversarial term. A temporal PatchGAN discriminator supplies a learned perceptual loss for texture; L1/PSNR ask for the conditional mean, which is a blur. Try 0.01-0.05 -- too high and it hallucinates detail that flickers through the recurrence.")
    ap.add_argument("--gan-lr", type=float, default=2e-4, help="discriminator learning rate.")
    ap.add_argument("--gan-warmup", type=int, default=5, help="epochs to train with GT losses only before enabling the adversarial term, so the discriminator does not shape a model that cannot yet reconstruct.")
    ap.add_argument("--no-full-eval", action="store_true",
                    help="skip the end-of-run full-frame eval. On 96-120 frame val "
                         "scenes it is pathologically slow -- it silently held the GPU "
                         "for 3 hours after one run had already saved its checkpoint, "
                         "starving the next run. scratchpad/eval_compare.py does the "
                         "same job on CPU in minutes and is what every result in "
                         "SCOREBOARD.md is actually measured with.")
    ap.add_argument("--crop-bank", type=str, default="",
                    help="path to a prebuilt crop bank (see fsrmamba/cropbank.py). "
                         "Replaces per-epoch disk cropping of the training scenes, "
                         "which on the capture HDD (~19 MB/s measured) costs ~20 "
                         "min/epoch. The bank must have been built from exactly this "
                         "run's training split or the run aborts.")
    ap.add_argument("--kernel-predict", action="store_true",
                    help="predict per-pixel softmax filter weights over "
                         "(9 raw LR taps + warped history + Lanczos resolve) "
                         "instead of a colour residual. Output becomes a convex "
                         "combination of real samples, so it cannot drift through "
                         "the recurrence and needs no tanh/box_stddev bound. "
                         "Initialised to FSR's 0.954/0.046 blend. See fsrmamba/kpn.py.")
    ap.add_argument("--unet", type=int, default=0, help="U-Net feature trunk with this many downsampling levels (0 = flat). Conv cost is channels^2 x pixels, so a level that doubles channels while quartering pixels costs the same -- capacity grows exponentially for linear cost. 4 levels + --separable is 918k params at 28.9 GFLOP vs the flat trunk 372k at 591.6. This is the shape DLSS 4.5 and FSR 4.1 use.")
    ap.add_argument("--separable", action="store_true", help="use depthwise-separable convolutions everywhere. A 64->64 dense 3x3 costs 36,864 MAC/px; separable costs 4,672 -- 7.9x cheaper for comparable capacity (the MobileNet factorisation). At v52 widths this takes the model from 592 to 191 GFLOP/frame, 34.6 to 11.2 ms.")
    ap.add_argument("--lr-film", action="store_true",
                    help="condition the LR upsampler on the jitter offset "
                         "multiplicatively (FiLM) instead of only as constant input "
                         "planes. ~4k extra parameters; see fsrmamba/lrnet.py.")
    ap.add_argument("--frames-per-seq", type=int, default=0,
                    help="crop-bank only: use a random contiguous slice of this many "
                         "frames from each window per epoch (0 = the whole window). "
                         "An epoch's cost is windows x frames, and the multi-path set "
                         "has 3x the scenes, so without this an epoch is 3x longer than "
                         "the runs it is being compared against. Also acts as temporal "
                         "augmentation, since the start offset is redrawn each epoch.")
    ap.add_argument("--eval-every", type=int, default=2,
                    help="run validation every N epochs. Each eval point costs "
                         "eval_crops x val_scenes x frames_per_scene model forwards, so on "
                         "96-frame scenes it is ~2400 forwards and dominates runtime "
                         "(measured ~30 min/epoch instead of the expected ~2).")
    ap.add_argument("--feature-channels", type=int, default=24, help="CNN width (model capacity)")
    ap.add_argument("--encoder-depth", type=int, default=2, help="number of conv layers in flat encoder (default 2)")
    ap.add_argument("--ssim-weight", type=float, default=0.0, help="weight on (1 - SSIM) loss")
    ap.add_argument("--grad-weight", type=float, default=0.0, help="weight on gradient-L1 (sharpness) loss")
    ap.add_argument("--freq-weight", type=float, default=0.0,
                    help="weight on log-magnitude FFT L1 loss (global blur penalty)")
    # Frequency Distribution Loss. Off by default: it costs ~5x a plain step, so
    # it belongs in a fine-tune, not a from-scratch run. In webvsr it was the
    # only one of nine imported techniques that measured better rather than
    # worse, and its gain was specifically on RENDERED content -- which is all
    # this project has. Suggested starting weight 0.75.
    ap.add_argument("--fdl-weight", type=float, default=0.0,
                    help="weight on Frequency Distribution Loss (sliced Wasserstein "
                         "on VGG feature FFTs; misalignment-robust, anti-blur)")
    ap.add_argument("--perceptual-weight", type=float, default=0.0,
                    help="weight on VGG16 feature-space L1 (perceptual, breaks L1 mean-blur)")
    ap.add_argument("--edge-bias", type=float, default=0.0,
                    help="probability a training crop is snapped to a frame edge "
                         "(trains border/disocclusion handling)")
    ap.add_argument("--sharpen", action="store_true", help="add learned RCAS-style sharpening head")
    ap.add_argument("--rectify", action="store_true",
                    help="clamp warped history into the neighbourhood colour box "
                         "(FSR RectifyHistory) with a learned per-pixel box width")
    ap.add_argument("--box-max", type=float, default=3.0,
                    help="upper bound on the learned rectification box width; "
                         "lower (e.g. 1.8) forces a tighter clamp -> sharper, more SSIM")
    ap.add_argument("--learned-resolve", action="store_true",
                    help="add a learned dilated-conv refiner on the Lanczos resolve "
                         "(reconstruct edge detail beyond a fixed spatial filter)")
    ap.add_argument("--swin-resolve", action="store_true",
                    help="windowed/shifted self-attention (Swin) refiner on the resolve "
                         "-- targets fine repetitive texture, the diagnosed failure")
    ap.add_argument("--swin-dim", type=int, default=48, help="embedding width of the Swin refiner")
    ap.add_argument("--lr-upsampler", action="store_true",
                    help="learned PixelShuffle upsampler reading the RAW low-res image "
                         "(the only path with access to detail Lanczos discarded)")
    ap.add_argument("--lr-dim", type=int, default=64, help="width of the LR upsampler trunk")
    ap.add_argument("--robust-disocc", action="store_true",
                    help="neighbourhood (3x3 min/max) depth test for disocclusion instead of "
                         "per-pixel -- stops spurious history rejection on geometric edges, "
                         "which is where temporal anti-aliasing comes from")
    ap.add_argument("--lock-feature", action="store_true",
                    help="feed FSR-style thin-feature (lock) confidence as an input, so the "
                         "model can protect fine detail instead of blending it away")
    ap.add_argument("--alpha-penalty", type=float, default=0.0,
                    help="loss penalty on the current-frame blend weight (excluding real "
                         "disocclusions) -- forces deeper temporal accumulation, which "
                         "averages out the resolve's per-phase bias (the 2x2 pixelation)")
    ap.add_argument("--alpha-target", type=float, default=0.0,
                    help="blend-weight the alpha penalty aims for (hinge target). FSR's "
                         "converged rate is ~0.045; our unconstrained model sits at ~0.265")
    ap.add_argument("--phase-residual", action="store_true",
                    help="phase: add a colour residual bounded by the local stddev")
    ap.add_argument("--logmse-weight", type=float, default=0.0,
                    help="weight on log10(MSE) per frame (optimises mean per-frame PSNR "
                         "directly; 0.02 moves the loss by 0.002 per dB)")
    ap.add_argument("--cold-weight", action="store_true",
                    help="weight the first frames after a state reset 4x (0-3) and 2x (4-7)")
    ap.add_argument("--fast-reset-lanczos", action="store_true",
                    help="fast (with --fast-base-gate): reset pixels take the Lanczos base")
    ap.add_argument("--select-full", action="store_true",
                    help="evaluate and select checkpoints on full validation frames "
                         "instead of crops")
    ap.add_argument("--jitter-sign", type=float, default=1.0,
                    help="-1: the captures place samples at texel centre + jitter (measured); "
                         "+1 keeps the ported baseline's centre - jitter convention")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an exponential moving average of the weights; the average "
                         "is what is evaluated and saved (0 = off, try 0.999)")
    ap.add_argument("--temporal-through", action="store_true",
                    help="let the temporal loss backpropagate into the previous frame "
                         "inside a BPTT window (it is detached by default)")
    ap.add_argument("--arch", choices=("mamba", "fast", "phase", "hq", "kpn"), default="mamba",
                    help="mamba: the full-resolution accumulator (v64). fast: the "
                         "render-resolution accumulator in fsrmamba/fast.py. "
                         "kpn: a render-resolution U-Net predicting per-output reconstruction filters.")
    ap.add_argument("--fast-widths", type=str, default="32,64",
                    help="fast: channels per pyramid level, from 1/2 render res downwards")
    ap.add_argument("--fast-depths", type=str, default="1,2",
                    help="fast: residual blocks per pyramid level")
    ap.add_argument("--fast-state", type=int, default=8,
                    help="fast: learned recurrent state channels (0 = no state, the ablation)")
    ap.add_argument("--fast-no-film", action="store_true", help="fast: no jitter conditioning")
    ap.add_argument("--fast-history-age", action="store_true", help="Reprojected frame count and history-weight floor")
    ap.add_argument("--fast-coverage", action="store_true", help="Temporal luma moments and coverage head")
    ap.add_argument("--fast-depth-soft", action="store_true",
                    help="Feed depth mismatch to the trunk without resetting history; requires --fast-depth-test.")
    ap.add_argument("--fast-depth-soft-osc", action="store_true",
                    help="Gate soft depth handling with coverage oscillation; requires depth-test, depth-soft and coverage.")
    ap.add_argument("--fast-depth-test", action="store_true",
                    help="fast: depth-envelope disocclusion test (off: frame border only)")
    for key in ("mv-dilate", "depth-dilate", "thin-lock"):
        ap.add_argument("--fast-" + key, action="store_true")
    ap.add_argument("--fast-stem-kernel", type=int, default=1, help="fast: stem conv size")
    ap.add_argument("--fast-nearest-sample", action="store_true",
                    help="fast: base each phase on the nearest low-res sample (may be a "
                         "neighbouring texel), matching the accumulation weight")
    ap.add_argument("--fast-conf-consistent", action="store_true",
                    help="fast: accumulate the corrected current weight that is blended")
    ap.add_argument("--fast-base-gate", action="store_true",
                    help="fast: gate between the deringed Lanczos resolve and the current sample")
    ap.add_argument("--fast-carry-raw", action="store_true",
                    help="fast (with --fast-accum): accumulate real samples and synthesise only for display")
    ap.add_argument("--fast-hist-residual", action="store_true",
                    help="fast: also predict a range-bounded correction to the history")
    ap.add_argument("--fast-conf-motion", action="store_true",
                    help="fast (with --fast-accum): divide the carried weight by "
                         "(1 + m * speed), learned m, so history loses authority under motion")
    ap.add_argument("--fast-accum", action="store_true",
                    help="fast: carry an accumulated sample weight and blend as a running "
                         "average (FSR's current / (history + current)), corrected by the net")
    ap.add_argument("--fast-hist-filter", choices=("bilinear", "bicubic"), default="bilinear",
                    help="fast: history reprojection filter")
    ap.add_argument("--fast-detail", type=int, default=0,
                    help="fast: channels of a thin render-resolution refinement branch (0 = off)")
    ap.add_argument("--fast-resolve", choices=("nearest", "lanczos"), default="nearest",
                    help="fast: base for the current frame -- the raw low-res sample, or "
                         "a jitter-aware Lanczos resolve of the 3x3 taps (FSR's)")
    ap.add_argument("--fast-learned-clamp", action="store_true",
                    help="fast: predict per sub-pixel how much of the history clamp to apply")
    ap.add_argument("--test-scenes", type=int, default=0,
                    help="hold out this many scenes (the last of the training split) as a "
                         "test set that is never trained on or used to select a checkpoint")
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="scale the blend weight on the current frame (<1 forces deeper "
                         "temporal accumulation, averaging out the resolve's per-phase bias)")
    config.add_arch_args(ap)
    args = ap.parse_args()
    if not math.isfinite(args.flicker_weight) or args.flicker_weight < 0:
        ap.error("--flicker-weight must be nonnegative and finite")
    if not 0 <= args.dither_aug <= 1:
        ap.error("--dither-aug must be in [0, 1]")
    if not math.isfinite(args.edge_loss_weight) or args.edge_loss_weight < 0:
        ap.error("--edge-loss-weight must be nonnegative and finite")
    if not math.isfinite(args.coverage_lr_mult) or args.coverage_lr_mult <= 0:
        ap.error("--coverage-lr-mult must be positive and finite")
    if args.coverage_lr_mult != 1 and (args.arch != "fast" or not args.fast_coverage or not args.engine_data):
        ap.error("--coverage-lr-mult requires engine training with --arch fast --fast-coverage")

    if args.engine_data:
        train_engine(args)
        return

    out_size = (args.height, args.width)
    render_size = (int(args.height / args.scale), int(args.width / args.scale))
    dev = torch.device(args.device)

    print(f"render {render_size[1]}x{render_size[0]} -> {out_size[1]}x{out_size[0]}")
    # Disjoint scene seeds. Train and val must not share a scene -- see
    # random_scene's docstring for why a frame-wise split would leak.
    train_seeds = list(range(args.train_scenes))
    val_seeds = list(range(1000, 1000 + args.val_scenes))
    print(f"pre-rendering {len(train_seeds)} train / {len(val_seeds)} val scenes...")
    t0 = time.time()
    train_seqs = [
        render_sequence(random_scene(s, out_size), args.seq_len, render_size, out_size,
                        args.gt_supersample)
        for s in train_seeds
    ]
    val_seqs = [
        render_sequence(random_scene(s, out_size), args.seq_len, render_size, out_size,
                        args.gt_supersample)
        for s in val_seeds
    ]
    print(f"  done in {time.time()-t0:.1f}s")

    def eval_all(model):
        rs = [evaluate(model, s, out_size, args.state_channels) for s in val_seqs]
        return tuple(sum(v) / len(rs) for v in zip(*rs))

    # --- baseline reference ---
    base = FSRAccumulator(render_size, out_size, device=dev)
    b_psnr, b_ssim, b_ti, b_dev = eval_all(base)
    print(f"\nFSR baseline   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  "
          f"temporal {b_ti:.5f}  |dev| {b_dev:.5f}")

    # --- train ---
    model = make_model(args, render_size, out_size, dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"learned model  {n_params:,} parameters, {args.num_experts} expert(s)\n")
    opt = training_optimizer(model, args.lr, args.coverage_lr_mult)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        d_acc, g_acc, adv_n = 0.0, 0.0, 0
        for seq in train_seqs:
            seq = augment_exposure(seq, args.exposure_aug, random)
            seq = augment_dither(seq, args.dither_aug, random)
            state = new_state(model, dev)
            prev_out, prev_depth, prev_gt = None, None, None
            prev_mask = None
            for start in range(0, len(seq), args.bptt):
                window = seq[start : start + args.bptt]
                if not window:
                    continue
                opt.zero_grad()
                loss = torch.zeros((), device=dev)
                for f in window:
                    out, state = model(
                        state, f["lr"], f["mv"], f["depth"], f["jitter"],
                        prev_depth_hr=prev_depth,
                    )
                    prev_depth = f["depth"]
                    loss = loss + F.l1_loss(out, f["gt"])
                    if args.edge_loss_weight > 0:
                        loss = loss + args.edge_loss_weight * edge_l1(out, f["gt"])
                    if prev_out is not None:
                        # Match ground truth's *temporal gradient*, not the
                        # previous frame itself. Penalising |out - warp(prev)|
                        # directly would reward a model that simply copies its
                        # history forward -- i.e. it would train ghosting in as
                        # the optimum. Comparing changes instead says: move the
                        # way the real scene moves.
                        if args.flicker_weight > 0:
                            loss = loss + args.flicker_weight * flicker_loss(
                                out, prev_out.detach(), f["gt"], prev_gt, f.get("gt_mv", f["mv"]),
                                getattr(model, "_last_reset", f.get("reset")), f.get("mask"), prev_mask)
                        d_out = out - warp_prev(prev_out.detach(), f["mv"])
                        d_gt = f["gt"] - warp_prev(prev_gt, f["mv"])
                        loss = loss + args.temporal_weight * F.l1_loss(d_out, d_gt)
                    if args.num_experts > 1 and args.balance_weight > 0:
                        loss = loss + args.balance_weight * model.load_balance_loss()
                    prev_out = out
                    prev_gt = f["gt"]
                    prev_mask = f.get("mask")
                loss = loss / len(window)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                # Truncate BPTT: carry state forward, drop the graph.
                state = state.detach()
                prev_out = prev_out.detach()

        if epoch % 2 == 0 or epoch == args.epochs - 1:
            model.eval()
            p, s, ti, dv = eval_all(model)
            flag = ""
            if p > b_psnr and ti < b_ti:
                flag = "  <-- beats baseline on both"
            print(
                f"epoch {epoch:3d}  loss {epoch_loss:7.4f}   "
                f"PSNR {p:6.2f}  SSIM {s:.4f}  temporal {ti:.5f}  |dev| {dv:.5f}{flag}"
            )

    print("\n--- final ---")
    print(f"FSR baseline   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  "
          f"temporal {b_ti:.5f}  |dev| {b_dev:.5f}")
    p, s, ti, dv = eval_all(model)
    print(f"learned        PSNR {p:6.2f}  SSIM {s:.4f}  "
          f"temporal {ti:.5f}  |dev| {dv:.5f}")
    print("\n|dev| is distance from the ground truth's OWN temporal instability;\n"
          "0 is the target. Raw 'temporal' alone is minimised by a blurrier\n"
          "output, so it cannot be selected on -- see metrics.temporal_deviation.")

    if args.save:
        torch.save(model.state_dict(), args.save)
        config.save_sidecar(args.save, args)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
