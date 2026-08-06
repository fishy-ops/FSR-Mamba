"""Build a crop bank from the capture set (see fsrmamba/cropbank.py for why).

The training split must match what train.py computes, or a validation scene can end
up in the bank -- so the split is derived the same way here: list_scenes() sorted,
first --val-scenes held out.

usage:
  python build_bank.py --engine-data D:/FSR-Mamba/captures_long \
      --out D:/FSR-Mamba/cropbank/long_w2_384.pt --windows 2 --win-render 384
"""
import argparse

from fsrmamba.cropbank import build_bank
from fsrmamba.engine_data import list_scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-scenes", type=int, default=3)
    ap.add_argument("--windows", type=int, default=2,
                    help="stored windows per scene. Each is sub-cropped at use time, "
                         "so this is not the number of distinct training crops.")
    ap.add_argument("--win-render", type=int, default=384,
                    help="window size at RENDER resolution. Must exceed --crop used "
                         "in training, or there is nothing to sub-crop.")
    ap.add_argument("--frames", type=int, default=0, help="0 = all frames in the scene")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = list_scenes(args.engine_data)
    val, train = scenes[:max(1, args.val_scenes)], scenes[max(1, args.val_scenes):]
    print(f"{len(train)} training scenes (holding out {val})")
    print(f"train: {train}")
    build_bank(args.engine_data, train, args.out, windows=args.windows,
               win_render=args.win_render, scale=args.scale,
               frames=(args.frames or None), seed=args.seed)


if __name__ == "__main__":
    main()
