"""Run every plain assertion test in a fresh Python process."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    failures = []
    paths = sorted((root / "tests").glob("test_*.py"))
    assert root / "tests" / "test_kpn.py" in paths
    for path in paths:
        print(f"Running {path.name}", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=root, env=env)
        if result.returncode:
            failures.append(path.name)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
