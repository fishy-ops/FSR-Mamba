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
import time

import torch
import torch.nn.functional as F

from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.mamba import MambaAccumulator, MambaState
from fsrmamba.metrics import psnr, ssim, temporal_instability
from fsrmamba.synth import halton_jitter, random_scene


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
    args = ap.parse_args()

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
