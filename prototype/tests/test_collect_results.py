"""Synthetic reporting, metadata joins, temporal indexing, and bootstrap checks."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.evalkit import paired_scene_bootstrap
from tools.collect_results import BASELINE, paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]
BETTER = "checkpoints/better.pt [fused fp16]"
WORSE = "checkpoints/worse.pt [amp]"


def write_json(path, data):
    path.write_text(json.dumps(data))


def evaluation():
    scenes = {}
    for i, scene in enumerate(("a", "b")):
        sources = {}
        for source, offset, scale in ((WORSE, -2, 2), (BASELINE, 0, 1), (BETTER, 2, 0.5)):
            scores = dict(psnr=[30 + 2 * i + t / 10 + offset for t in range(12)],
                          ssim=[0.8 + 0.02 * i + t / 1000 + offset / 100 for t in range(12)],
                          dev=[(-1) ** t * (0.01 + t / 1000) * scale for t in range(1, 12)])
            scores["gt_ti"] = [0.1] * 11
            scores["ti"] = [gt + dev for gt, dev in zip(scores["gt_ti"], scores["dev"])]
            sources[source] = dict(per_frame=scores, summary=dict(psnr=-999, ssim=-999))
        scenes[scene] = sources
    return dict(scenes=scenes, summary={}, comparisons={}, skip=8,
                configs={"different/path/better.pt": dict(arch="fast", widths=[9], depths=[9]),
                         "different/path/worse.pt": dict(arch="fast", widths=[24, 48],
                                                         depths=[0, 2], accum=True)})


def run(*args, success=True):
    result = subprocess.run([sys.executable, "tools/collect_results.py", *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr
    return result.stdout if success else result.stderr


def table_rows(section):
    lines = section.split("### Per scene")[0].splitlines()
    return [[part.strip() for part in line.split("|")[1:-1]]
            for line in lines if line.startswith("| ")][2:]


def test_bootstrap():
    a = [[t / 10 for t in range(12)], [t * t / 50 for t in range(12)]]
    b = [[0.2] * 12, [0.3] * 12]
    actual = paired_bootstrap(a, b)
    expected = paired_scene_bootstrap(a, b)
    assert all(math.isclose(x, y, abs_tol=1e-14) for x, y in zip(actual, expected))
    assert actual == paired_bootstrap(a, b)
    assert paired_bootstrap([[0] * 12, [100] * 4], [[0] * 12, [0] * 4]) == (50, 50, 50)
    assert paired_bootstrap([[1]], [[0]]) == (1, 1, 1)
    try:
        paired_bootstrap([[1, 2]], [[1]])
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal paired lengths accepted")


def main():
    torch.set_num_threads(1)
    test_bootstrap()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        a, b = tmp / "eval_a.json", tmp / "eval_b.json"
        data = evaluation()
        write_json(a, data)
        duplicate = copy.deepcopy(data)
        for sources in duplicate["scenes"].values():
            for entry in sources.values():
                entry["per_frame"]["psnr"] = [99] * 12
        write_json(b, duplicate)
        sidecars = tmp / "sidecars"
        sidecars.mkdir()
        write_json(sidecars / "better.pt.json",
                   dict(arch="fast", widths=[16, 32], depths=[1, 2], accum=True,
                        depth_test=True, hist_filter="bicubic", nearest_sample=True,
                        conf_consistent=True, hist_residual=True))
        old, latest = tmp / "old.log", tmp / "latest.log"
        old.write_text("SUMMARY widths=16,32 depths=1,2 total_with_display=9.0000\n"
                       "E2E widths=(16, 32) depths=(1, 2) {}: TOTAL TRT 8.00 ms\n")
        latest.write_text(
            "SUMMARY widths=16,32 depths=1,2 pack=0.4 trunk=1 resolve=0.2 "
            "total=1.6 total_with_display=2.3000\n"
            "SUMMARY widths=16,32 depths=1,2 pack=0.4 trunk=1 resolve=0.2 "
            "total=1.6 total_with_display=2.3450\n"
            "SUMMARY widths=24,48 depths=0,2 total_with_display=3.4560\n"
            "E2E widths=(16, 32) depths=(1, 2) {'ns': True}: pack 0.46 | "
            "trunk TRT 0.79 (cuDNN graph 0.99) | resolve 0.22 | TOTAL TRT 1.60 ms | "
            "TOTAL cuDNN 1.72 ms\n"
            "E2E widths=(16, 32) depths=(1, 2) {}: TOTAL TRT 1.50 ms | TOTAL cuDNN 1.72 ms\n"
            "TRUNK widths=(16, 32) depths=(1, 2): TensorRT fp16 99.0 ms | cuDNN 99.0 ms\n"
            "TRUNK widths=(24, 48) depths=(0, 2): TensorRT fp16 99.0 ms | cuDNN 99.0 ms\n")
        args = ["--eval", a, b, "--sidecars", sidecars, "--bench", old, latest,
                "--skip", "0", "--skip", "4", "--title", "Synthetic results"]
        output = run(*args)
        assert output == run(*args)
        assert output.startswith("# Synthetic results\n")
        first, second = output.split("## Frames >= ")[1:]
        rows, skipped = table_rows(first), table_rows(second)
        assert [r[0] for r in rows] == [BASELINE, BETTER, WORSE]
        assert [r[0] for r in skipped] == [BASELINE, BETTER, WORSE]
        assert rows[1][1] == "fast 16,32/1,2 accum dt bicubic ns cc hres"
        assert rows[1][2:4] == ["2.35", "1.50"]
        assert rows[2][1] == "fast 24,48/0,2 accum"
        assert rows[2][2:4] == ["3.46", "-"]
        for index, verdict, wins in ((0, "tie", "0"), (1, "better", "3"), (2, "worse", "0")):
            for metric in rows[index][4:7]:
                assert f", {verdict}; CI [" in metric
            assert rows[index][7] == wins
        assert rows[0][4].startswith("31.55 (+0.00, tie;")
        assert rows[1][4].startswith("33.55 (+2.00, better;")
        assert rows[0][5].startswith("0.8155 (+0.0000, tie;")
        assert rows[0][6].startswith("0.01600 (+0.00000, tie;")
        assert rows[1][6].startswith("0.00800 (-0.00800, better;")
        assert skipped[0][4].startswith("31.75 (+0.00, tie;")
        assert skipped[0][5].startswith("0.8175 (+0.0000, tie;")
        assert skipped[0][6].startswith("0.01750 (+0.00000, tie;")
        assert skipped[1][6].startswith("0.00875 (-0.00875, better;")
        assert f"| a | {BASELINE} | 30.55 | 0.8055 |" in first
        assert f"| b | {BETTER} | 34.75 | 0.8475 |" in second
        assert "frames >= 0" in first and "t >= 1" in first
        assert "frames >= 4" in second and "t >= 4" in second
        for source in (BASELINE, BETTER, WORSE):
            assert f"- {source}: {a}; scenes a, b" in output
        assert str(b) not in output
        out = tmp / "results.md"
        assert run(*args, "--out", out) == ""
        assert out.read_text() == output
        filtered = run("--eval", a, "--scenes", "b", "--skip", "4")
        assert table_rows(filtered)[0][4].startswith("32.75 (+0.00, tie;")
        assert f"| a | {BASELINE}" not in filtered
        default = run("--eval", a)
        assert table_rows(default)[0][4].startswith("31.55 (+0.00, tie;")
        assert table_rows(default)[1][2:4] == ["-", "-"]
        empty = run("--eval", a, "--skip", "16")
        assert all(", n/a;" in metric for r in table_rows(empty) for metric in r[4:7])
        assert "nonnegative" in run("--eval", a, "--skip", "-1", success=False)
        assert "Requested scenes" in run("--eval", a, "--scenes", "missing", success=False)
        local = copy.deepcopy(duplicate)
        for entries in local["scenes"].values():
            entries["local.pt"] = copy.deepcopy(entries[BASELINE])
            entries["local.pt"]["per_frame"]["psnr"] = [101] * 12
        write_json(b, local)
        combined = run("--eval", a, b)
        local_row = next(r for r in table_rows(combined) if r[0] == "local.pt")
        assert local_row[4].startswith("101.00 (+2.00, better;")
        assert f"- local.pt: {b}; scenes a, b" in combined
        broken = copy.deepcopy(data)
        broken["scenes"]["a"][BETTER]["per_frame"]["dev"].pop()
        write_json(b, broken)
        assert "equal lengths" in run("--eval", b, success=False)
    print("test_collect_results passed")


if __name__ == "__main__":
    main()
