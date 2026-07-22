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
import pathlib
import random
import time

import torch
import torch.nn.functional as F

from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.mamba import MambaAccumulator, MambaState
from fsrmamba.metrics import psnr, ssim, temporal_instability
from fsrmamba.synth import halton_jitter, random_scene
from fsrmamba.engine_data import crop_sequence, list_scenes, load_engine_scene


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


def evaluate(model, frames, out_size, state_channels):
    """Run a model over a sequence and return (psnr, ssim, temporal instability)."""
    is_learned = isinstance(model, MambaAccumulator)
    state = (
        MambaState.zeros(out_size, state_channels)
        if is_learned
        else FSRState.zeros(out_size)
    )
    prev_out, prev_depth = None, None
    tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0}
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
                n_ti += 1
            prev_out = out
    n = len(frames)
    return tot["psnr"] / n, tot["ssim"] / n, tot["ti"] / max(1, n_ti)


def evaluate_captured(frames) -> tuple[float, float, float]:
    """Baseline metrics from the *captured* FSR output vs ground truth.

    In engine mode the baseline is the real FSR 3.1.4 output we captured, not the
    ported accumulator -- so this just scores frames["fsr_out"] against the GT.
    """
    tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0}
    n_ti = 0
    prev = None
    for f in frames:
        tot["psnr"] += psnr(f["fsr_out"], f["gt"])
        tot["ssim"] += ssim(f["fsr_out"], f["gt"])
        if prev is not None:
            tot["ti"] += temporal_instability(f["fsr_out"], prev, f["mv"])
            n_ti += 1
        prev = f["fsr_out"]
    n = len(frames)
    return tot["psnr"] / n, tot["ssim"] / n, tot["ti"] / max(1, n_ti)


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

    scenes = list_scenes(args.engine_data)
    if len(scenes) < 2:
        raise SystemExit(f"need >=2 scenes in {args.engine_data}, found {scenes}")
    n_val = max(1, args.val_scenes)
    val_names, train_names = scenes[:n_val], scenes[n_val:]
    print(f"engine data: {len(train_names)} train / {len(val_names)} val scenes")
    print(f"  train: {train_names}")
    print(f"  val:   {val_names}")
    print(f"crop render {crop_r}x{crop_r} -> out {crop_o}x{crop_o}")

    # Full sequences stay on CPU; crops move to the GPU per use (1080p x all
    # scenes will not fit in VRAM, but a crop is tiny).
    t0 = time.time()
    train_full = [load_engine_scene(f"{args.engine_data}/{s}", device="cpu") for s in train_names]
    val_full = [load_engine_scene(f"{args.engine_data}/{s}", device="cpu") for s in val_names]
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
        state = MambaState.zeros(out_size, args.state_channels, device=dev) if run_model else None
        prev_out, prev_depth = None, None
        tot = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0}
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
                n_ti += 1
            prev_out = out
        n = len(frames)
        return (tot["psnr"] / n, tot["ssim"] / n, tot["ti"] / max(1, n_ti))

    def eval_model():
        model.eval()
        with torch.no_grad():
            rs = [_score(c, model) for c in val_crops]
        return tuple(sum(v) / len(rs) for v in zip(*rs))

    with torch.no_grad():
        rs = [_score(c, None) for c in val_crops]
    b_psnr, b_ssim, b_ti = (sum(v) / len(rs) for v in zip(*rs))
    print(f"\nFSR (captured) PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}")

    model = MambaAccumulator(render_size, out_size, state_channels=args.state_channels,
                             num_experts=args.num_experts, device=dev)
    print(f"learned model  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{args.num_experts} expert(s)\n")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    best_psnr = -1.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for full in train_full:
            seq = to_dev(crop_sequence(full, crop_r, int(args.scale)))  # fresh crop = augmentation
            state = MambaState.zeros(out_size, args.state_channels, device=dev)
            prev_out, prev_depth, prev_gt = None, None, None
            for start in range(0, len(seq), args.bptt):
                window = seq[start:start + args.bptt]
                if not window:
                    continue
                opt.zero_grad()
                loss = torch.zeros((), device=dev)
                for f in window:
                    out, state = model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                                       prev_depth_hr=prev_depth)
                    prev_depth = f["depth"]
                    loss = loss + F.l1_loss(out, f["gt"])
                    if prev_out is not None:
                        d_out = out - warp_prev(prev_out.detach(), f["mv"])
                        d_gt = f["gt"] - warp_prev(prev_gt, f["mv"])
                        loss = loss + args.temporal_weight * F.l1_loss(d_out, d_gt)
                    if args.num_experts > 1 and args.balance_weight > 0:
                        loss = loss + args.balance_weight * model.load_balance_loss()
                    prev_out, prev_gt = out, f["gt"]
                loss = loss / len(window)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
                state = state.detach()
                prev_out = prev_out.detach()
        sched.step()

        if epoch % 2 == 0 or epoch == args.epochs - 1:
            p, s, ti = eval_model()
            gap = p - b_psnr
            flag = "  <-- beats FSR" if gap > 0 else ""
            print(f"epoch {epoch:3d}  loss {epoch_loss:7.4f}  lr {sched.get_last_lr()[0]:.1e}   "
                  f"PSNR {p:6.2f} ({gap:+.2f} vs FSR)  SSIM {s:.4f}  temporal {ti:.5f}{flag}")
            if p > best_psnr:
                best_psnr = p
                if args.save:
                    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), args.save)

    print("\n--- final (engine data, held-out scenes, multi-crop) ---")
    print(f"FSR (captured)   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}")
    p, s, ti = eval_model()
    print(f"learned (last)   PSNR {p:6.2f}  SSIM {s:.4f}  temporal {ti:.5f}")
    print(f"learned (best)   PSNR {best_psnr:6.2f}   gap to FSR {best_psnr - b_psnr:+.2f} dB")
    if args.save:
        print(f"saved best model to {args.save}")


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
    ap.add_argument("--temporal-weight", type=float, default=0.25)
    ap.add_argument("--gt-supersample", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--train-scenes", type=int, default=6)
    ap.add_argument("--val-scenes", type=int, default=3)
    ap.add_argument("--num-experts", type=int, default=1)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--balance-weight", type=float, default=0.01)
    ap.add_argument("--engine-data", type=str, default="",
                    help="captures dir (<scene>/{fsr,gt}); trains on real engine data vs GT")
    ap.add_argument("--crop", type=int, default=128, help="render-res train crop (engine mode)")
    ap.add_argument("--eval-crops", type=int, default=8, help="seeded crops per val scene (engine mode)")
    args = ap.parse_args()

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
    b_psnr, b_ssim, b_ti = eval_all(base)
    print(f"\nFSR baseline   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}")

    # --- train ---
    model = MambaAccumulator(
        render_size, out_size, state_channels=args.state_channels,
        num_experts=args.num_experts, device=dev
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"learned model  {n_params:,} parameters, {args.num_experts} expert(s)\n")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for seq in train_seqs:
            state = MambaState.zeros(out_size, args.state_channels, device=dev)
            prev_out, prev_depth, prev_gt = None, None, None
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
                    if prev_out is not None:
                        # Match ground truth's *temporal gradient*, not the
                        # previous frame itself. Penalising |out - warp(prev)|
                        # directly would reward a model that simply copies its
                        # history forward -- i.e. it would train ghosting in as
                        # the optimum. Comparing changes instead says: move the
                        # way the real scene moves.
                        d_out = out - warp_prev(prev_out.detach(), f["mv"])
                        d_gt = f["gt"] - warp_prev(prev_gt, f["mv"])
                        loss = loss + args.temporal_weight * F.l1_loss(d_out, d_gt)
                    if args.num_experts > 1 and args.balance_weight > 0:
                        loss = loss + args.balance_weight * model.load_balance_loss()
                    prev_out = out
                    prev_gt = f["gt"]
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
            p, s, ti = eval_all(model)
            flag = ""
            if p > b_psnr and ti < b_ti:
                flag = "  <-- beats baseline on both"
            print(
                f"epoch {epoch:3d}  loss {epoch_loss:7.4f}   "
                f"PSNR {p:6.2f}  SSIM {s:.4f}  temporal {ti:.5f}{flag}"
            )

    print("\n--- final ---")
    print(f"FSR baseline   PSNR {b_psnr:6.2f}  SSIM {b_ssim:.4f}  temporal {b_ti:.5f}")
    p, s, ti = eval_all(model)
    print(f"learned        PSNR {p:6.2f}  SSIM {s:.4f}  temporal {ti:.5f}")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
