"""Fast checkpoint reconstruction, streaming tools, export, and latency sweep."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_latency import make_inputs, run_pass, smooth_motion
from diagnose_regions import accumulate, region_masks
from fsrmamba.config import (FAST_DEFAULTS, add_arch_args, build_model, infer_config,
                             load_checkpoint, load_sidecar, model_kwargs, new_state, save_sidecar)
from fsrmamba.engine_data import load_engine_scene
from fsrmamba.evalkit import full_frame, run_scene
from fsrmamba.fast import FastAccumulator, FastState
from test_scripts import run
from tools.onnx_fast import FastStep, export_fast, export_inputs, feature_buffer
from tools.sweep_latency import configs


CASES = [dict(arch="fast", widths=[32, 64], depths=[1, 2], n_state=8),
         dict(arch="fast", widths=[24], depths=[2], n_state=0),
         dict(arch="fast", widths=[16, 32, 64], depths=[0, 1, 2], n_state=4,
              film=False, stem_kernel=3)]


def test_config(tmp):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("mamba", "fast"), default="mamba")
    ap.add_argument("--fast-state", type=int, default=8)
    add_arch_args(ap)
    args = ap.parse_args(["--arch", "fast", "--fast-widths", "16,32,64", "--fast-depths",
                          "0,1,2", "--fast-state", "4", "--fast-no-film",
                          "--fast-stem-kernel", "3", "--fast-depth-test"])
    kwargs = model_kwargs(args)
    assert kwargs == dict(CASES[2], widths=(16, 32, 64), depths=(0, 1, 2), depth_test=True, learned_clamp=False, resolve="nearest", detail_ch=0, hist_filter="bilinear", accum=False, conf_motion=False, hist_residual=False, nearest_sample=False, conf_consistent=False, carry_raw=False, base_gate=False, reset_lanczos=False, mv_dilate=False, depth_dilate=False, thin_lock=False, depth_soft=False, depth_soft_osc=False, coverage=False, history_age=False, coverage_bias=-3.0, jitter_sign=1.0)
    for render in ((32, 48), (35, 64)):
        output = tuple(v * 2 for v in render)
        for i, cfg in enumerate(CASES):
            model = build_model(cfg, render, output)
            assert isinstance(model, FastAccumulator) and not model.training
            state = new_state(model, "cpu")
            assert isinstance(state, FastState)
            assert state.feat.shape == (1, cfg["n_state"], *model._tr_size)
            sd = model.state_dict()
            expected = dict(FAST_DEFAULTS, **cfg)
            inferred = infer_config(sd)
            assert inferred == expected, (inferred, expected)
            build_model(inferred, render, output).load_state_dict(sd, strict=True)
            path = tmp / f"fast_{render[0]}_{i}.pt"
            torch.save(sd, path)
            loaded, loaded_cfg = load_checkpoint(path, render, output)
            assert loaded_cfg == inferred
            save_sidecar(path, cfg)
            assert load_sidecar(path) == expected
            loaded, loaded_cfg = load_checkpoint(path, render, output)
            assert loaded_cfg == expected
            assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in sd.items())
            save_sidecar(path, dict(cfg, depth_test=True))
            assert load_checkpoint(path, render, output)[0].depth_test
            assert not load_checkpoint(path, render, output, overrides={"depth_test": False})[0].depth_test
            assert infer_config(sd, {"depth_test": True})["depth_test"]
            save_sidecar(path, cfg)
    sd = build_model(CASES[0], (32, 48), (64, 96)).state_dict()
    for extra in (1, 16, 64):
        broken = dict(sd)
        broken["out.weight"] = torch.zeros(sd["out.weight"].shape[0] + extra, 32, 1, 1)
        try:
            infer_config(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scale accepted")
    broken = dict(sd)
    broken.pop("fuse.bias")
    path = tmp / "broken.pt"
    torch.save(broken, path)
    try:
        load_checkpoint(path, (32, 48), (64, 96))
    except RuntimeError as exc:
        assert "fuse.bias" in str(exc)
    else:
        raise AssertionError("Strict fast load accepted a missing parameter")
    for scale in (1, 3):
        model = build_model(CASES[0], (8, 12), (8 * scale, 12 * scale))
        assert infer_config(model.state_dict())["n_state"] == 8


@torch.no_grad()
def test_wrapper(tmp):
    for render in ((32, 48), (35, 64)):
        output = tuple(v * 2 for v in render)
        for cfg in CASES:
            for depth_test in (False, True):
                model = build_model(dict(cfg, depth_test=depth_test), render, output)
                # Exercise the learned residual, recurrent update, and jitter modulation.
                for parameter in model.parameters():
                    parameter.uniform_(-0.08, 0.08)
                state = new_state(model)
                state.frame_index = 3
                state.color.uniform_(0, 1)
                state.feat.uniform_(-0.5, 0.5)
                state.depth.uniform_(0.6, 1.6)
                lr, mv, depth = make_inputs(model, torch.device("cpu"))
                mv_hr = F.interpolate(mv.permute(2, 0, 1)[None], size=output,
                                       mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                hist = model.reproject(state, mv_hr)
                jitter = torch.tensor([0.125, -0.375])
                rgb, new = model(state, lr, mv, depth, jitter, hist=hist)
                tensors = (hist, lr.permute(2, 0, 1)[None], mv[None], depth[None, None],
                           jitter, state.feat)
                if depth_test:
                    tensors += (state.depth,)
                wrapper = FastStep(model)
                out, feat = wrapper(*tensors)
                torch.testing.assert_close(out, rgb.permute(2, 0, 1)[None], atol=1e-5, rtol=0)
                torch.testing.assert_close(feat, new.feat, atol=1e-5, rtol=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", torch.jit.TracerWarning)
                    traced = torch.jit.trace(wrapper, tensors, check_trace=False)
                changed = tuple(t.clone() for t in tensors)
                changed[0].uniform_(0, 1)
                changed[1].uniform_(0, 1)
                changed[4].copy_(torch.tensor([-0.25, 0.4]))
                expected = wrapper(*changed)
                actual = traced(*changed)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    model = build_model(dict(CASES[0], depth_test=True), (35, 64), (70, 128))
    inputs = make_inputs(model, torch.device("cpu"))
    tensors, names = export_inputs(model, inputs, torch.device("cpu"))
    assert names[-1] == "prev_depth" and tensors[1].shape == (1, 3, 35, 64)
    path = tmp / "fast.onnx"
    exported = export_fast(model, inputs, path, torch.device("cpu"))
    if importlib.util.find_spec("onnx") is None:
        assert exported is None
        print("ONNX graph test skipped: onnx unavailable; PyTorch wrapper and tracing passed.")
    else:
        assert exported is not None and path.exists()
        import onnx
        graph = onnx.load(path)
        assert graph.opset_import[0].version >= 17
        assert [out.name for out in graph.graph.output] == ["out", "feat_new"]


def test_streaming(tmp):
    for render in ((32, 48), (35, 64)):
        tag = f"{render[0]}x{render[1]}"
        data = tmp / tag
        run("tools/make_fake_engine.py", "--out", data, "--scenes", 1,
            "--frames", 3, "--render", tag)
        path = tmp / f"random_{tag}.pt"
        cfg = dict(CASES[0], depth_test=True)
        model = build_model(cfg, render, tuple(v * 2 for v in render))
        torch.save(model.state_dict(), path)
        save_sidecar(path, cfg)
        scene = load_engine_scene(str(data / "scene_00"), half=True, mmap=True)
        scores = run_scene(scene, model)
        assert len(scores["psnr"]) == 3 and len(scores["ti"]) == 2
        assert all(math.isfinite(v) for values in scores.values() for v in values)
        regional = accumulate(scene, model)
        for regions in regional.values():
            assert sum(r["pixels"] for r in regions.values()) == 2 * render[0] * render[1] * 4
            assert math.isclose(sum(r["pixel_fraction"] for r in regions.values()), 1.0)
        frame = full_frame(scene[1], "cpu")
        model._last_reset = torch.zeros(1, 1, *render)
        model._last_reset[..., :3, :5] = 1
        masks = region_masks(frame, model)
        assert masks["disoccluded"].sum() == 3 * 5 * 4
        report = tmp / f"report_{tag}.json"
        text = run("eval_full.py", "--engine-data", data, "--ckpt", path,
                   "--frames", 3, "--json", report)
        assert "Mean over scenes" in text
        assert json.loads(report.read_text())["configs"][str(path)]["arch"] == "fast"
        text = run("eval_full.py", "--ckpt", path, "--drift", 3,
                   "--height", render[0], "--width", render[1])
        assert "slope" in text and "stable" in text
        text = run("diagnose_regions.py", "--engine-data", data, "--ckpt", path)
        assert all(region in text for region in ("disoccluded", "edge", "moving", "flat"))
    return path


def test_latency(tmp, ckpt):
    for dtype in (torch.float32, torch.float16):
        tensor = feature_buffer(torch.empty(1, 0, 16, 24, dtype=dtype))
        assert tensor.shape == (1, 0, 16, 24) and tensor.dtype == dtype
        assert tensor.is_contiguous() and tensor.untyped_storage().data_ptr() != 0
    text = run("bench_latency.py", "--arch", "fast", "--render", "32x48", "--frames", 3,
               "--warmup", 1, "--fp16", "--channels-last", "--onnx", tmp / "bench.onnx", "--ort")
    for phrase in ("full step", "network + blend", "reprojection alone", "stem", "enc[0]",
                   "enc[1]", "down[0]", "up[0]", "fuse", "out", "film", "median", "mean", "p95"):
        assert phrase in text, (phrase, text)
    assert text.count("p95") >= 3
    assert "ONNX" in text
    if importlib.util.find_spec("onnxruntime") is None:
        assert text.count("ONNX Runtime unavailable") == 1
    assert "full step" in run("bench_latency.py", "--ckpt", ckpt, "--render", "35x64",
                              "--frames", 3, "--warmup", 0)
    model = build_model(CASES[0], (32, 48), (64, 96))
    inputs = make_inputs(model, torch.device("cpu"))
    assert inputs[1].shape == (32, 48, 2) and inputs[2].shape == (32, 48)
    mv_pixels = smooth_motion((32, 48), (32, 48), "cpu") * torch.tensor([48, 32])
    assert (mv_pixels[1:] - mv_pixels[:-1]).abs().max() < 0.05
    assert (mv_pixels[:, 1:] - mv_pixels[:, :-1]).abs().max() < 0.05
    counts = [0]
    original = model.reproject
    def reproject(*args):
        counts[0] += 1
        return original(*args)
    model.reproject = reproject
    times = run_pass(model, inputs, torch.device("cpu"), 3, 1, mode="network")
    assert len(times) == 3 and counts[0] == 4
    all_configs = list(configs())
    assert len(all_configs) == 144 and len({json.dumps(c, sort_keys=True) for c in all_configs}) == 144
    assert {tuple(c["widths"]) for c in all_configs} == {
        (16,), (24,), (32,), (16, 32), (24, 48), (32, 64), (48, 96), (32, 64, 128)}
    csv_path = tmp / "sweep.csv"
    text = run("tools/sweep_latency.py", "--render", "32x48", "--device", "cpu",
               "--frames", 3, "--warmup", 1, "--limit", 3, "--budget-ms", 1000,
               "--csv", csv_path)
    assert "network + blend" in text and "full step" in text
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and all(row["within_budget"] == "True" for row in rows)
    medians = [float(row["network_ms"]) for row in rows]
    assert medians == sorted(medians) and min(medians) > 0
    assert all(int(row["parameters"]) > 0 and float(row["full_ms"]) > 0 for row in rows)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_config(tmp)
        test_wrapper(tmp)
        ckpt = test_streaming(tmp)
        test_latency(tmp, ckpt)
    print("test_fast_tooling passed")


if __name__ == "__main__":
    main()
