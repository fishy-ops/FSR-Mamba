"""Extra capture split guards, epoch sampling, loss gating, and CPU training."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.engine_data import list_scenes, load_extra_engine_scenes
from tools.make_fake_engine import make_fake_engine
import train


ROOT = Path(__file__).resolve().parents[1]


def epoch_plans():
    for seed in range(20):
        for main_count in (0, 1, 3, 10):
            before, after = random.Random(seed), random.Random(seed)
            expected = list(range(main_count))
            before.shuffle(expected)
            actual = train.plan_epoch(main_count, 0, 4, after)
            assert actual == [("main", i) for i in expected]
            assert before.getstate() == after.getstate()
        for per_epoch in (0, 1, 3, 10):
            plan = train.plan_epoch(3, 5, per_epoch, random.Random(seed))
            assert plan == train.plan_epoch(3, 5, per_epoch, random.Random(seed))
            assert sorted(i for kind, i in plan if kind == "main") == [0, 1, 2]
            extras = [i for kind, i in plan if kind == "extra"]
            assert len(extras) == min(5, per_epoch)
            assert len(set(extras)) == len(extras)
            assert all(0 <= i < 5 for i in extras)
    draws = [train.plan_epoch(3, 8, 2, random.Random(seed)) for seed in range(20)]
    assert len({tuple(p) for p in draws}) > 1
    for counts in ((-1, 0, 0), (1, -1, 0), (1, 1, -1)):
        try:
            train.plan_epoch(*counts, random.Random(0))
        except ValueError:
            pass
        else:
            raise AssertionError("negative sequence count accepted")

    frames = list(range(10))
    starts = set()
    for seed in range(100):
        window = train.sequence_window(frames, 6, random.Random(seed))
        assert len(window) == 6
        assert window == list(range(window[0], window[0] + 6))
        starts.add(window[0])
    assert starts == set(range(5))
    for count in (10, 48):
        rng = random.Random(9)
        state = rng.getstate()
        assert train.sequence_window(frames, count, rng) is frames
        assert rng.getstate() == state


def load_extras(extra, training=("bbb", "ccc", "ddd")):
    output = io.StringIO()
    with redirect_stdout(output):
        loaded = load_extra_engine_scenes(extra, training, ["aaa"], ["eee"], 2, 6)
    return loaded, output.getvalue()


def split_guards(main, extra):
    assert list_scenes(main) == ["aaa", "bbb", "ccc", "ddd", "eee"]
    loaded, output = load_extras(extra)
    assert [name for name, _ in loaded] == ["bbb", "ccc_slow", "ddd_fast"]
    assert "excluded (validation/test base): ['aaa_fast', 'eee']" in output
    assert "excluded (unknown base): ['zzz']" in output
    assert loaded[0][1][0]["lr"].shape == (16, 20, 3)
    assert all(v.dtype == torch.float16 for _, frames in loaded for f in frames
               for v in f.values() if torch.is_tensor(v))
    try:
        load_extras(extra, training=("aaa", "bbb", "ccc", "ddd", "eee"))
    except SystemExit as error:
        assert "would leak" in str(error)
    else:
        raise AssertionError("overlapping training and held-out bases accepted")

    wrong = extra / "ccc_slow" / "_cache_tm1.pt"
    frames = torch.load(wrong, weights_only=True)
    frames[0]["gt"] = frames[0]["gt"][:, :-1]
    torch.save(frames, wrong)
    short = extra / "ddd_fast" / "_cache_tm1.pt"
    torch.save(torch.load(short, weights_only=True)[:3], short)
    loaded, output = load_extras(extra)
    assert [name for name, _ in loaded] == ["bbb"]
    assert "extra skip ccc_slow: render 12x16 -> output 24x31" in output
    assert "does not match --scale 2" in output
    assert "extra skip ddd_fast: 3 frames < --bptt 6" in output

    # Both output axes must match, even when the other axis has the right ratio.
    frames[0]["gt"] = torch.zeros(23, 32, 3, dtype=torch.float16)
    torch.save(frames, wrong)
    _, output = load_extras(extra)
    assert "extra skip ccc_slow: render 12x16 -> output 23x32" in output


def training(main, extra):
    command = ["train.py", "--engine-data", str(main), "--extra-engine-data", str(extra),
               "--extra-frames", "6", "--arch", "fast", "--fast-accum",
               "--jitter-sign", "-1", "--epochs", "1", "--crop", "8", "--bptt", "6",
               "--val-scenes", "1", "--test-scenes", "1", "--eval-crops", "1",
               "--device", "cpu", "--no-full-eval", "--augment", "--edge-bias", "0.5",
               "--edge-crop-prob", "0.5", "--logmse-weight", "0.01",
               "--ssim-weight", "0.1", "--grad-weight", "0.1", "--edge-grad-weight", "0.1"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "train: ['bbb', 'ccc', 'ddd']" in result.stdout
    assert "val:   ['aaa']" in result.stdout and "held-out test ['eee']" in result.stdout
    assert "1 extra sequences used" in result.stdout
    assert "learned (last)" in result.stdout

    for options, expected in (([], 18), (["--no-extra-lowpass"], 24),
                              (["--extra-per-epoch", "0"], 18)):
        with patch.object(sys, "argv", command + options), patch.object(train, "train_engine") as run:
            train.main()
        args = run.call_args.args[0]
        assert args.extra_lowpass == ("--no-extra-lowpass" not in options)
        assert args.extra_per_epoch == (0 if options == ["--extra-per-epoch", "0"] else None)
        random.seed(7)
        torch.manual_seed(7)
        with patch.object(train, "gradient_l1", wraps=train.gradient_l1) as grad, \
             patch.object(train, "edge_gradient_l1", wraps=train.edge_gradient_l1) as edge, \
             patch.object(train, "ssim_map_mean", wraps=train.ssim_map_mean) as ssim, \
             redirect_stdout(io.StringIO()):
            train.train_engine(args)
        assert grad.call_count == edge.call_count == ssim.call_count == expected


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    epoch_plans()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        main_data, extra = root / "main", root / "extra"
        make_fake_engine(main_data, scenes=5, frames=6, render=(12, 16))
        for i, name in enumerate(("aaa", "bbb", "ccc", "ddd", "eee")):
            (main_data / f"scene_{i:02d}").rename(main_data / name)
        make_fake_engine(extra, scenes=6, frames=10, render=(12, 16))
        for i, name in enumerate(("aaa_fast", "bbb", "ccc_slow", "ddd_fast", "eee", "zzz")):
            (extra / f"scene_{i:02d}").rename(extra / name)
        different_size = root / "different_size"
        make_fake_engine(different_size, scenes=1, frames=10, render=(16, 20))
        (extra / "bbb" / "_cache_tm1.pt").unlink()
        (different_size / "scene_00" / "_cache_tm1.pt").rename(extra / "bbb" / "_cache_tm1.pt")
        split_guards(main_data, extra)
        training(main_data, extra)
    print("test_extra_engine passed")


if __name__ == "__main__":
    main()
