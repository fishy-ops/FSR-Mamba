"""End-to-end CLI smoke tests using temporary captures and checkpoints."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_regions import accumulate, new_totals, region_masks
from fsrmamba.config import build_model, save_sidecar
from fsrmamba.engine_data import list_scenes, load_engine_scene
from fsrmamba.evalkit import full_frame


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        data = tmp / "captures"
        output = run("tools/make_fake_engine.py", "--out", data, "--scenes", 2,
                     "--frames", 4, "--render", "8x12")
        assert "Wrote scene_01" in output
        extra = tmp / "extra"
        run("tools/make_fake_engine.py", "--out", extra, "--scenes", 1,
            "--frames", 3, "--render", "12x16")
        (extra / "scene_00").rename(data / "scene_02")
        assert list_scenes(data) == ["scene_00", "scene_01", "scene_02"]
        scene = load_engine_scene(str(data / "scene_00"), half=True, mmap=True)
        assert len(scene) == 4
        assert all(v.dtype == torch.float16 for frame in scene for v in frame.values()
                   if torch.is_tensor(v))
        assert scene[0]["lr"].shape == (8, 12, 3)
        assert scene[0]["mv"].shape == (8, 12, 2)
        assert scene[0]["depth"].shape == (8, 12)
        assert scene[0]["gt"].shape == (16, 24, 3)
        paths = []
        for i, cfg in enumerate(({}, dict(feature_channels=16, rectify=True))):
            model = build_model(cfg, (8, 12), (16, 24))
            path = tmp / f"model_{i}.pt"
            torch.save(model.state_dict(), path)
            if i == 1:
                save_sidecar(path, cfg)
            paths.append(path)
        report_path = tmp / "report.json"
        output = run("eval_full.py", "--engine-data", data, "--ckpt", *paths,
                     "--lpips", "--block", 2, "--json", report_path, "--robust-disocc", 0)
        for phrase in ("FSR (captured)", "Mean over scenes", "GT instab", "|dev|", "95% CI", "tie"):
            assert phrase in output, (phrase, output)
        assert f"{paths[1]} - {paths[0]}" in output
        report = json.loads(report_path.read_text())
        assert len(report["scenes"]) == 3
        for sources in report["scenes"].values():
            for scores in sources.values():
                assert len(scores["per_frame"]["ti"]) == len(scores["per_frame"]["psnr"]) - 1
                assert "abs_dev" in scores["summary"]
        assert all(cfg["robust_disocc"] is False for cfg in report["configs"].values())
        output = run("eval_full.py", "--ckpt", *paths, "--drift", 30,
                     "--height", 8, "--width", 12)
        assert "FSR (ported)" in output and "slope" in output and "stable" in output
        output = run("diagnose_regions.py", "--engine-data", data, "--ckpt", paths[0])
        assert all(phrase in output for phrase in ("disoccluded", "edge", "moving", "flat",
                                                 "share of total", "GT instab"))
        model = build_model({}, (8, 12), (16, 24))
        totals = new_totals()
        regional = accumulate(scene, model, totals=totals)
        for source, regions in regional.items():
            assert math.isclose(sum(r["pixel_fraction"] for r in regions.values()), 1.0, abs_tol=1e-12)
            assert sum(r["pixels"] for r in regions.values()) == 3 * 16 * 24
            assert math.isclose(sum(r["squared_error_share"] for r in regions.values()), 1.0)
        assert all(regional["model"][r]["pixel_fraction"] == regional["FSR"][r]["pixel_fraction"]
                   for r in regional["model"])
        frame = full_frame(scene[1], "cpu")
        model._last_reset = torch.ones(1, 1, 16, 24)
        masks = region_masks(frame, model)
        assert masks["disoccluded"].all()
        assert not any(masks[name].any() for name in ("edge", "moving", "flat"))
        model._last_reset.zero_()
        frame["gt"] = torch.zeros(16, 24, 3)
        frame["mv"] = torch.zeros(16, 24, 2)
        assert region_masks(frame, model)["flat"].all()
        frame["mv"][..., 0] = 3 / 24
        assert region_masks(frame, model)["moving"].all()
        output = run("bench_latency.py", "--render", "32x48", "--frames", 3, "--warmup", 1,
                     "--fp16", "--channels-last", "--onnx", tmp / "step.onnx")
        for phrase in ("Parameters:", "mean", "median", "p95", "frames/s", "Per-module",
                       "encoder", "_resolve._upsample", "_warp", "other", "ONNX", "running fp32"):
            assert phrase in output, (phrase, output)
        output = run("bench_latency.py", "--ckpt", paths[0], "--render", "8x12",
                     "--frames", 2, "--warmup", 0)
        assert "Latency" in output
    print("test_scripts passed")


if __name__ == "__main__":
    main()
