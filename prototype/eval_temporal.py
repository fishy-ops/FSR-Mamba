"""Evaluate saved accumulators with the ground-truth-referenced temporal metric.

The point of this script is one question: is the learned accumulator's low
temporal instability a genuine stability win, or is it sitting BELOW the ground
truth's own instability -- which would mean it is over-smoothing, the failure a
learned temporal filter is most prone to and the one raw instability cannot see.
"""
import argparse, torch
from fsrmamba.synth import random_scene
# reuse train.py's renderer verbatim so eval and training cannot diverge
from train import render_sequence
from fsrmamba.baseline import FSRAccumulator, FSRState
from fsrmamba.mamba import MambaAccumulator, MambaState
from fsrmamba.metrics import psnr, ssim, temporal_instability, temporal_deviation


def run(model, frames, out_size, state_channels):
    learned = isinstance(model, MambaAccumulator)
    state = (MambaState.zeros(out_size, state_channels) if learned
             else FSRState.zeros(out_size))
    prev_out = prev_depth = prev_gt = None
    acc = {"psnr": 0.0, "ssim": 0.0, "ti": 0.0, "dev": 0.0, "gt_ti": 0.0}
    n = ntemp = 0
    with torch.no_grad():
        for f in frames:
            out, state = model(state, f["lr"], f["mv"], f["depth"], f["jitter"],
                               prev_depth_hr=prev_depth)
            prev_depth = f["depth"]
            acc["psnr"] += psnr(out, f["gt"]); acc["ssim"] += ssim(out, f["gt"]); n += 1
            if prev_out is not None:
                acc["ti"] += temporal_instability(out, prev_out, f["mv"])
                d, g = temporal_deviation(out, prev_out, f["gt"], prev_gt, f["mv"])
                acc["dev"] += d          # SIGNED: negative = more stable than truth
                acc["gt_ti"] += g
                ntemp += 1
            prev_out, prev_gt = out, f["gt"]
    t = max(1, ntemp)
    return (acc["psnr"]/n, acc["ssim"]/n, acc["ti"]/t, acc["dev"]/t, acc["gt_ti"]/t)


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="*", default=["moe4.pt", "moe4_bal.pt"])
ap.add_argument("--height", type=int, default=180)
ap.add_argument("--width", type=int, default=320)
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--seq-len", type=int, default=12)
ap.add_argument("--val-scenes", type=int, default=3)
ap.add_argument("--state-channels", type=int, default=8)
ap.add_argument("--gt-supersample", type=int, default=4)
a = ap.parse_args()

out_size = (a.height, a.width)
render_size = (int(a.height / a.scale), int(a.width / a.scale))
seqs = [render_sequence(random_scene(s, out_size), a.seq_len, render_size,
                        out_size, a.gt_supersample)
        for s in range(1000, 1000 + a.val_scenes)]        # same held-out seeds as train.py
print(f"render {render_size[1]}x{render_size[0]} -> {out_size[1]}x{out_size[0]}, "
      f"{a.val_scenes} held-out scenes x {a.seq_len} frames\n")

models = {"FSR baseline (ported)": (FSRAccumulator(render_size, out_size), a.state_channels)}
for c in a.ckpt:
    sd = torch.load(c, map_location="cpu", weights_only=False)
    # expert count comes from router.weight's output channels. Match that key
    # exactly -- "router" also matches the scalar router_inertia, which has no
    # shape[0] at all.
    # Infer the geometry from the checkpoint rather than trusting CLI defaults:
    # these files were trained at state_channels=12, not the default 8, and a
    # mismatch fails loudly here but would silently mis-evaluate if forced.
    # Infer geometry by SEARCHING for the config whose state_dict matches
    # exactly. Reading it off individual tensors does not work -- the decoder's
    # input width is a derived quantity, not state_channels, so a plausible
    # single-tensor read gives 8 where the truth is 12. These checkpoints were
    # trained at state_channels=12, not the CLI default of 8.
    want = {k: tuple(v.shape) for k, v in sd.items()}
    ne = sd["router.weight"].shape[0] if "router.weight" in sd else 1
    m = sc = None
    for cand_sc in (4, 8, 12, 16, 24, 32):
        for cand_fc in (16, 24, 32, 48):
            try:
                cand = MambaAccumulator(render_size, out_size,
                                        state_channels=cand_sc,
                                        feature_channels=cand_fc,
                                        num_experts=ne)
            except Exception:
                continue
            if {k: tuple(v.shape) for k, v in cand.state_dict().items()} == want:
                m, sc = cand, cand_sc
                break
        if m is not None:
            break
    if m is None:
        raise SystemExit(f"could not match {c}'s geometry -- refusing to guess")
    m.load_state_dict(sd); m.eval()
    models[f"learned ({c}, {ne}ex s{sc})"] = (m, sc)

print(f"{'model':<34}{'PSNR':>7}{'SSIM':>8}{'instab.':>10}{'GT instab.':>12}{'dev':>11}  verdict")
print("-" * 96)
for name, (m, sc_) in models.items():
    ps = ss = ti = dv = gt = 0.0
    for s in seqs:
        p_, s_, t_, d_, g_ = run(m, s, out_size, sc_)
        ps += p_; ss += s_; ti += t_; dv += d_; gt += g_
    k = len(seqs)
    v = ("OVER-SMOOTHED (more stable than reality)" if dv/k < -1e-4
         else "flickers more than truth" if dv/k > 1e-4 else "matches truth")
    print(f"{name:<34}{ps/k:>7.2f}{ss/k:>8.4f}{ti/k:>10.5f}{gt/k:>12.5f}{dv/k:>+11.5f}  {v}")
print("\ndev = output instability - ground truth instability. 0 is the target.")
