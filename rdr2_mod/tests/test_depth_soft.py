"""Export depth-soft/coverage shader fixtures and compare FAST/generic D3D12 paths on Windows."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_sequence import export_sequence
import torch
from fsrmamba.config import save_sidecar
from fsrmamba.fast import FastAccumulator


def source(render):
    h, w = render
    rgb = torch.rand(h, w, 3)
    mv = torch.randn(h, w, 2) * .001
    mv[0, :, 1] = -1
    for frame in range(8):
        depth = torch.randint(0, 4, (h, w)).float() * .2 + .1
        depth[4:9, 7:12] = .8 if frame % 2 else .2
        color = rgb.clone()
        color[12:22, 20:35] = .1 if frame % 2 else .9
        depth[12:22, 20:35] = .4 if frame < 2 or frame % 2 else .48
        color[25:35, 40:55] = .1 if frame < 3 else .9
        left = 4 + 2 * frame
        color[40:50] = .05
        depth[40:50] = .2
        color[40:50, left:left+8] = .9
        depth[40:50, left:left+8] = .8
        motion = mv.clone()
        motion[40:50] = 0
        motion[40:50, left:left+8, 0] = -2 / w
        yield color, motion, depth, ((.4, -.4), (-.4, .4), (.25, -.25))[frame % 3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--temp-dir", required=True, type=Path)
    p.add_argument("--render", help="Override the two default render sizes with HxW, e.g. 720x1280 for pack timing")
    p.add_argument("--prepare-only", action="store_true", help="Export fixtures without running D3D12")
    a = p.parse_args()
    a.temp_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(417)
    renders = ((64, 96), (63, 95))
    if a.render:
        try:
            render = tuple(int(v) for v in a.render.lower().split("x"))
            if len(render) != 2 or min(render) < 8:
                raise ValueError
        except ValueError:
            p.error("--render must be HxW with both >= 8")
        renders = (render,)
    count = fixtures = 0
    for render, soft, dilate, bias, osc, age in itertools.product(renders, (False, True), (False, True), (None, -6., -3.), (False, True), (False, True)):
        coverage = bias is not None
        if osc and not (soft and coverage):
            continue
        stem = f"depth-soft-{render[0]}x{render[1]}-soft{int(soft)}-mv{int(dilate)}-cov{int(coverage)}-osc{int(osc)}-bias{bias}-age{int(age)}"
        checkpoint = a.temp_dir / (stem + ".pt")
        sequence = a.temp_dir / (stem + ".seq.bin")
        weights = a.temp_dir / (stem + ".weights.bin")
        cfg = dict(arch="fast", widths=[8, 16], depths=[1, 1], n_state=0, accum=True,
                   carry_raw=True, base_gate=True, nearest_sample=True, conf_consistent=True,
                   hist_filter="bicubic", depth_test=True, depth_soft=soft, depth_soft_osc=osc, mv_dilate=dilate, coverage=coverage,
                   coverage_bias=bias if coverage else -6., history_age=age)
        model = FastAccumulator(render, tuple(2 * v for v in render),
                                **{k: v for k, v in cfg.items() if k != "arch"}).eval()
        with torch.no_grad():
            model.out.weight.normal_(0, .008)
            model.out.bias.normal_(0, .03)
        torch.save(model.state_dict(), checkpoint)
        save_sidecar(checkpoint, cfg)
        export_sequence(checkpoint, sequence, 8, weights, source(render), render, reset_frames=(4,))
        fixtures += 1
        data = weights.read_bytes()
        length = int.from_bytes(data[8:12], "little")
        exported = json.loads(data[12:12+length])
        assert exported["depth_soft"] is soft and exported["coverage"] is coverage
        assert exported["depth_soft_osc"] is osc
        assert exported["history_age"] is age
        assert exported["coverage_bias"] == cfg["coverage_bias"]
        if sys.platform == "win32" and not a.prepare_only:
            # Engine-space input allows FAST; model-space input always selects generic.
            for extra in ([], ["--no-fast-path"]):
                subprocess.run([a.binary.resolve(), weights.resolve(), sequence.resolve(),
                                "--engine-space", *extra], check=True)
                count += 1
    if a.prepare_only or sys.platform != "win32":
        print(f"SKIP: D3D12 runtime parity and pack timing require Windows; {fixtures} fixture configurations exported")
    else:
        print(f"PASS: {count} FAST/generic D3D12 runs, 8 frames each, reset at frame 4")


if __name__ == "__main__":
    main()
