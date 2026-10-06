"""Capture scoring protocols, native auxiliaries, and fused evaluation plumbing."""

from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_full
from fsrmamba.config import build_model, load_checkpoint, new_state, save_sidecar
from fsrmamba.engine_data import load_engine_scene
from fsrmamba.evalkit import full_frame, paired_scene_bootstrap, run_scene, summarise
from fsrmamba.metrics import psnr, ssim, temporal_deviation, temporal_instability


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout


@torch.no_grad()
def legacy_scores(scene, model):
    scores = {key: [] for key in ("psnr", "ssim", "ti", "gt_ti", "dev")}
    state = new_state(model) if model is not None else None
    prev_out, prev_gt, prev_depth = None, None, None
    for raw in scene:
        frame = full_frame(raw, "cpu")
        if model is None:
            out = frame["fsr_out"]
        else:
            out, state = model(state, frame["lr"], frame["mv"], frame["depth"],
                               frame["jitter"], prev_depth_hr=prev_depth)
        out = out.float()
        scores["psnr"].append(float(psnr(out, frame["gt"])))
        scores["ssim"].append(float(ssim(out, frame["gt"])))
        if prev_out is not None:
            dev, gt_ti = temporal_deviation(out, prev_out, frame["gt"], prev_gt, frame["mv"])
            scores["ti"].append(float(temporal_instability(out, prev_out, frame["mv"])))
            scores["gt_ti"].append(float(gt_ti))
            scores["dev"].append(float(dev))
        prev_out, prev_gt, prev_depth = out, frame["gt"], frame["depth"]
    return scores


def check_skip(tmp, data, paths, scene):
    reports = []
    for skip in (0, 1, 3, 6):
        path = tmp / f"skip_{skip}.json"
        output = run("eval_full.py", "--engine-data", data, "--ckpt", *paths,
                     "--skip", skip, "--block", 2, "--report-ranges", "--json", path)
        assert f"frames >= {skip} scored" in output
        report = json.loads(path.read_text())
        assert report["skip"] == skip
        reports.append(report)
        sources = report["scenes"]["scene_00"]
        for source, scores in sources.items():
            values = scores["per_frame"]
            assert len(values["psnr"]) == 6 and len(values["dev"]) == 5
            for key, series in values.items():
                kept = series[max(skip - 1, 0):] if key in ("ti", "gt_ti", "dev") else series[skip:]
                expected = sum(kept) / len(kept) if kept else 0.0
                assert scores["summary"][key] == expected
                assert report["summary"][source][key] == expected
            kept = values["dev"][max(skip - 1, 0):]
            assert scores["summary"]["abs_dev"] == (sum(map(abs, kept)) / len(kept) if kept else 0.0)
        for label, comparison in report["comparisons"].items():
            later, earlier = label.split(" - ")
            for key, row in comparison.items():
                start = max(skip - 1, 0) if key == "abs_dev" else skip
                def series(source):
                    values = sources[source]["per_frame"]["dev" if key == "abs_dev" else key][start:]
                    return list(map(abs, values)) if key == "abs_dev" else values
                expected = paired_scene_bootstrap([series(later)], [series(earlier)], block=2)
                for actual, value in zip((row["mean_diff"], row["lo"], row["hi"]), expected):
                    assert actual == value or (math.isnan(actual) and math.isnan(value))
    original = reports[0]["scenes"]["scene_00"]
    for source in original:
        model = load_checkpoint(source, (8, 12), (16, 24))[0] if source in paths else None
        before = legacy_scores(scene, model)
        assert original[source]["per_frame"] == before
        old_summary = {key: sum(values) / len(values) for key, values in before.items()}
        old_summary["abs_dev"] = sum(map(abs, before["dev"])) / len(before["dev"])
        assert original[source]["summary"] == old_summary
        for report in reports[1:]:
            assert report["scenes"]["scene_00"][source]["per_frame"] == before


def check_native(data, paths, scene):
    h, w = scene[0]["lr"].shape[:2]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    motion = torch.stack((0.002 + x * 0.0001, -0.001 + y * 0.0001), dim=-1)
    smooth = [dict(frame, mv=motion.half()) for frame in scene]
    for path in paths[:2]:
        model = load_checkpoint(path, (8, 12), (16, 24))[0]
        seen, outputs = [], []
        handle = model.register_forward_pre_hook(lambda module, args: seen.append((args[2].shape, args[3].shape)))
        native = run_scene(smooth, model, native_lr=True,
                           on_frame=lambda i, f, out, *rest: outputs.append(out.clone()))
        handle.remove()
        assert seen == [((8, 12, 2), (8, 12))] * 6
        default = run_scene(smooth, model)
        assert max(abs(a - b) for a, b in zip(native["psnr"], default["psnr"])) < 1
        assert native["gt_ti"] == default["gt_ti"]
        for i in range(1, 6):
            mv_hr = full_frame(smooth[i], "cpu")["mv"]
            assert native["ti"][i - 1] == float(temporal_instability(outputs[i], outputs[i - 1], mv_hr))
    output = run("eval_full.py", "--engine-data", data, "--ckpt", *paths, "--native-lr")
    assert output.count("--native-lr ignored for MambaAccumulator") == 1


def check_fused(data, paths, scene):
    model = load_checkpoint(paths[0], (8, 12), (16, 24))[0]
    for accum, hist in ((False, "bilinear"), (True, "bicubic")):
        if accum:
            model = build_model(dict(arch="fast", widths=[8], depths=[1], n_state=0,
                                     accum=accum, hist_filter=hist), (8, 12), (16, 24))
            with torch.no_grad():
                model.out.weight.normal_(0, 0.01)
        expected, actual = [], []
        forward = run_scene(scene, model, native_lr=True,
                            on_frame=lambda i, f, out, *rest: expected.append(out.clone()))
        fused = run_scene(scene, model, fused_backend="reference",
                          on_frame=lambda i, f, out, *rest: actual.append(out.clone()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
        assert abs(summarise(forward)["psnr"] - summarise(fused)["psnr"]) < 1e-4
    output = run("eval_full.py", "--engine-data", data, "--ckpt", *paths, "--fused")
    assert output.count("Fused fp16 skipped: requires CUDA and CuPy") == 1
    assert "[fused fp16]" not in output
    args = SimpleNamespace(engine_data=data, ckpt=paths, scenes=None, frames=None, device="cuda",
                           lpips=False, robust_disocc=None, amp=True, fused=True, native_lr=False,
                           skip=3, report_ranges=False, block=2)
    def cpu_load(path, render, output, device, overrides):
        return load_checkpoint(path, render, output, "cpu", overrides)
    def reference_run(*args, **kwargs):
        scene, model, device, *rest = args
        if kwargs["fused_backend"] is not None:
            kwargs["fused_backend"] = "reference"
        kwargs["amp"] = False
        return run_scene(scene, model, "cpu", *rest, **kwargs)
    stream = io.StringIO()
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.dict(sys.modules, cupy=SimpleNamespace()), \
            patch.object(eval_full, "load_checkpoint", side_effect=cpu_load), \
            patch.object(eval_full, "run_scene", side_effect=reference_run), redirect_stdout(stream):
        report = eval_full.evaluate(args)
    sources = list(report["summary"])
    assert sources[:3] == ["FSR (captured)", paths[0] + " [amp]", paths[0] + " [fused fp16]"]
    assert paths[0] + " [fused fp16] - " + paths[0] + " [amp]" in report["comparisons"]
    assert stream.getvalue().count("[fused fp16] skipped: FusedFast requires arch fast") == 2


def main():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        data = tmp / "captures"
        run("tools/make_fake_engine.py", "--out", data, "--scenes", 1, "--frames", 6, "--render", "8x12")
        scene = load_engine_scene(str(data / "scene_00"), half=True, mmap=True)
        paths = []
        for arch in ("fast", "phase", "mamba"):
            cfg = (dict(arch=arch, widths=[8], depths=[1], n_state=0) if arch != "mamba"
                   else dict(arch=arch, state_channels=4, feature_channels=8))
            model = build_model(cfg, (8, 12), (16, 24))
            with torch.no_grad():
                if arch == "fast":
                    model.out.weight.normal_(0, 0.01)
                elif arch == "phase":
                    model.gates.weight.normal_(0, 0.01)
            path = tmp / f"{arch}.pt"
            torch.save(model.state_dict(), path)
            save_sidecar(path, cfg)
            paths.append(str(path))
        check_skip(tmp, data, paths, scene)
        check_native(data, paths, scene)
        check_fused(data, paths, scene)
        amp_json = tmp / "amp.json"
        run("eval_full.py", "--engine-data", data, "--ckpt", paths[0], "--amp", "--json", amp_json)
        assert paths[0] + " [amp]" in json.loads(amp_json.read_text())["summary"]
        longer = [scene[i % 6] for i in range(34)]
        torch.save(longer, data / "scene_00" / "_cache_tm1.pt")
        output = run("eval_full.py", "--engine-data", data, "--ckpt", paths[0], "--report-ranges")
        assert all(f"  {label}:" in output for label in ("0-3", "4-7", "8-15", "16-31", "32+"))
        output = run("eval_full.py", "--engine-data", data, "--ckpt", paths[0], "--skip", 16, "--report-ranges")
        assert all(f"  {label}:" not in output for label in ("0-3", "4-7", "8-15"))
        assert all(f"  {label}:" in output for label in ("16-31", "32+"))
    print("test_eval_full passed")


if __name__ == "__main__":
    main()
