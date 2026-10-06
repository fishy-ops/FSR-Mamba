"""Metric equivalence, paired scene bootstrap, and streamed capture scoring."""

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import build_model
from fsrmamba.evalkit import (full_frame, paired_block_bootstrap, paired_scene_bootstrap,
                              run_scene, summarise, temporal_diff_map)
from fsrmamba.metrics import temporal_instability


def fake_scene(frames=4, render=(8, 12), output=(16, 24)):
    generator = torch.Generator().manual_seed(12)
    return [{"lr": torch.rand((*render, 3), generator=generator).half(),
             "mv": (torch.randn((*render, 2), generator=generator) * 0.005).half(),
             "depth": (1 + torch.rand(render, generator=generator)).half(),
             "gt": torch.rand((*output, 3), generator=generator).half(),
             "fsr_out": torch.rand((*output, 3), generator=generator).half(),
             "jitter": (0.0, 0.0) if i % 2 else torch.tensor([0.1, -0.2])}
            for i in range(frames)]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    curr, prev = torch.rand(16, 24, 3), torch.rand(16, 24, 3)
    for mv in (torch.randn(16, 24, 2) * 0.05, torch.zeros(16, 24, 2), torch.ones(16, 24, 2) * 2):
        diff, valid = temporal_diff_map(curr, prev, mv)
        value = diff[valid].mean().item() if valid.any() else 0.0
        assert abs(value - temporal_instability(curr, prev, mv)) < 1e-6
    a = list(range(20))
    assert paired_block_bootstrap(a, a) == (0.0, 0.0, 0.0)
    assert paired_block_bootstrap([x + 1 for x in a], a) == (1.0, 1.0, 1.0)
    result = paired_block_bootstrap(a, list(reversed(a)), block=3)
    assert result == paired_block_bootstrap(a, list(reversed(a)), block=3)
    assert result[1] < 0 < result[2]
    assert paired_block_bootstrap([2], [1], block=100) == (1.0, 1.0, 1.0)
    assert paired_scene_bootstrap([[0] * 9, [100] * 3], [[0] * 9, [0] * 3], block=2) == (25, 25, 25)
    try:
        paired_block_bootstrap([1], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal lengths accepted")
    scene = fake_scene()
    snapshots = [{k: v.clone() for k, v in f.items() if torch.is_tensor(v)} for f in scene]
    f = full_frame(scene[0], "cpu")
    assert f["mv"].shape == (16, 24, 2) and f["depth"].shape == (16, 24)
    assert f["jitter"] is scene[0]["jitter"]
    assert all(v.dtype == torch.float32 for k, v in f.items() if k != "jitter")
    tensor_jitter = dict(scene[0], jitter=torch.tensor([0.25, -0.25], dtype=torch.float16))
    assert full_frame(tensor_jitter, "cpu")["jitter"].dtype == torch.float32
    assert torch.equal(full_frame(tensor_jitter, "cpu")["jitter"], tensor_jitter["jitter"].float())
    for model in (None, build_model({}, (8, 12), (16, 24))):
        calls = []
        def hook(i, frame, out, prev_out, prev_gt, passed_model):
            assert passed_model is model
            assert (prev_out is None) == (i == 0)
            assert (prev_gt is None) == (i == 0)
            calls.append(i)
        scores = run_scene(iter(scene), model, on_frame=hook)
        assert calls == list(range(4))
        for key, values in scores.items():
            assert len(values) == (4 if key in ("psnr", "ssim") else 3)
            assert all(isinstance(v, float) and math.isfinite(v) for v in values)
        assert all(abs(ti - gt_ti - dev) < 1e-8 for ti, gt_ti, dev in
                   zip(scores["ti"], scores["gt_ti"], scores["dev"]))
        summary = summarise(scores)
        assert summary["abs_dev"] == sum(abs(v) for v in scores["dev"]) / 3
        assert len(run_scene(iter(scene), model, max_frames=2)["ti"]) == 1
    for snapshot, frame in zip(snapshots, scene):
        assert all(torch.equal(snapshot[k], frame[k]) for k in snapshot)
    print("test_evalkit passed")


if __name__ == "__main__":
    main()
