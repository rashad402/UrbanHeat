"""Run every test suite and exit non-zero if any of them fails.

Usage:  python tests/run_all.py

Each suite is a standalone script with its own PASS/FAIL accounting (no pytest required, so it
runs anywhere the project runs). They are executed in separate processes because tests/test_api.py
installs a fake `ee` module into sys.modules, which must not leak into the others.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUITES = [
    ("SEB physics", "test_sebal.py"),
    ("Physics augmentation", "test_synthetic.py"),
    ("Planner API", "test_api.py"),
]


def main():
    failed = []
    for label, script in SUITES:
        print(f"\n{'=' * 70}\n{label}  ({script})\n{'=' * 70}")
        r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=ROOT)
        if r.returncode != 0:
            failed.append(label)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"SUITES FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(SUITES)} suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
