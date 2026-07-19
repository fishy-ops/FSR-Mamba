"""Is the router actually routing, or has it collapsed?

An aggregate quality metric cannot distinguish two very different failures:

  (a) the router specialises correctly, but routing simply does not help on this
      data -- a real null result;
  (b) the router collapsed to near-uniform weights, so "4 experts" is really one
      averaged expert wearing a hat -- an untested hypothesis, not a null result.

These call for opposite responses, so measure before concluding. This script
reports, over held-out scenes:

  mean max weight   1/K means uniform (collapsed); 1.0 means hard specialisation
  entropy ratio     H(weights) / log(K); 1.0 is uniform, 0.0 is fully committed
  per-expert usage  the share of pixels each expert wins -- catches the case
                    where one expert takes everything
  flip rate         share of pixels changing winner between consecutive frames,
                    which is the router-flicker the hysteresis term exists to
                    suppress
"""

from __future__ import annotations

import argparse

import torch

from fsrmamba.mamba import MambaAccumulator, MambaState
from fsrmamba.synth import halton_jitter, random_scene
from train import render_sequence


@torch.no_grad()
def probe(model, frames, out_size, state_channels):
    k = model.num_experts
    state = MambaState.zeros(out_size, state_channels)
    prev_depth, prev_win = None, None
    max_w, ent, usage, flips, n = 0.0, 0.0, torch.zeros(k), 0.0, 0

    for f in frames:
        # Re-run the front half of forward() to recover the routing weights.
        upsampled, _, box_center, box_stddev = model._resolve._upsample(f["lr"], f["jitter"])
        uv = model._resolve._hr_uv + f["mv"]
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
        if prev_depth is not None:
            warped_prev = torch.nn.functional.grid_sample(
                prev_depth[None, None], (uv * 2 - 1).unsqueeze(0),
                mode="nearest", padding_mode="border", align_corners=False)[0, 0]
            rel = (warped_prev - f["depth"]).abs() / f["depth"].clamp(min=1e-3)
            disocc = (rel > 0.1).float()
        else:
            disocc = torch.zeros_like(f["depth"])
        disocc = torch.maximum(disocc, ((~inside) | (state.frame_index == 0)).float())

        vel = torch.linalg.vector_norm(
            f["mv"] * torch.tensor([3840.0, 2160.0]), dim=-1)
        feats = torch.cat((upsampled, box_center, box_stddev,
                           disocc.unsqueeze(-1),
                           (vel / 20.0).clamp(0, 1).unsqueeze(-1),
                           (1.0 / f["depth"].clamp(min=1e-2)).unsqueeze(-1)), dim=-1)
        enc = model.encoder(feats.permute(2, 0, 1).unsqueeze(0))

        warped = model._warp(state.hidden, f["mv"]).unsqueeze(0)
        prev_logits = warped[:, 3 : 3 + model.n_router]
        reset = disocc.view(1, 1, *disocc.shape)
        inertia = torch.sigmoid(model.router_inertia) * (1.0 - reset)
        logits = torch.lerp(model.router(enc), prev_logits, inertia)
        w = torch.softmax(logits, dim=1)

        max_w += w.max(dim=1).values.mean().item()
        p = w.clamp(min=1e-9)
        ent += (-(p * p.log()).sum(dim=1).mean().item()) / torch.tensor(float(k)).log().item()
        win = w.argmax(dim=1)
        usage += torch.bincount(win.flatten(), minlength=k).float() / win.numel()
        if prev_win is not None:
            flips += (win != prev_win).float().mean().item()
        prev_win = win
        n += 1

        # Advance the real model so state stays consistent.
        _, state = model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                         prev_depth_hr=prev_depth)
        prev_depth = f["depth"]

    return max_w / n, ent / n, usage / n, flips / max(1, n - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--state-channels", type=int, default=12)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=10)
    args = ap.parse_args()

    out_size = (args.height, args.width)
    render_size = (int(args.height / args.scale), int(args.width / args.scale))
    model = MambaAccumulator(render_size, out_size,
                             state_channels=args.state_channels,
                             num_experts=args.num_experts)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"loaded {args.checkpoint}")
    else:
        print("NOTE: no --checkpoint given, probing an untrained model")
    model.eval()

    frames = render_sequence(random_scene(1000, out_size), args.seq_len,
                             render_size, out_size, 3)
    mx, ent, usage, flips = probe(model, frames, out_size, args.state_channels)
    k = args.num_experts

    print(f"\nexperts            {k}")
    print(f"mean max weight    {mx:.4f}   (uniform = {1/k:.4f}, committed = 1.0)")
    print(f"entropy ratio      {ent:.4f}   (1.0 = uniform, 0.0 = fully committed)")
    print(f"per-expert usage   {[round(u, 3) for u in usage.tolist()]}")
    print(f"router flip rate   {flips:.4f}   (share of pixels changing expert per frame)")

    if ent > 0.95:
        print("\n=> COLLAPSED. The router is near-uniform: this is one averaged")
        print("   expert, not four. The routing hypothesis has not been tested.")
    elif max(usage.tolist()) > 0.95:
        print("\n=> DEGENERATE. One expert wins nearly everything; the rest are dead.")
    else:
        print("\n=> Router is specialising. A null quality result here is a real one.")


if __name__ == "__main__":
    main()
