# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Run SpectralLM's correctness gates and record what they measured.

Usage (from the repository root): python3 papers/spectral-lm/scripts/correctness.py

Runs the finite-difference gradient checks and the model gates with their
output shown, collects every `GATE {...}` line they print (each gate asserts
its own tolerance as well), and writes papers/spectral-lm/data/correctness/
correctness.json with the commit, the commands and the device. Fails if any
gate failed or no gate line was printed.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMANDS = [
    ["cargo", "test", "--release", "--offline", "-p", "brain-gradcheck", "--test", "spectrallm_fd", "--", "--nocapture", "--test-threads=1"],
    ["cargo", "test", "--release", "--offline", "-p", "brain-spectrallm", "--test", "model", "--", "--nocapture", "--test-threads=1"],
]


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip():
        sys.exit("the tracked tree has uncommitted changes; the record must name a commit that reproduces it")
    checks = []
    for cmd in COMMANDS:
        run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        # The test harness prints "test <name> ... " without a newline before a
        # test's own output, so a gate line need not start the line.
        checks += [json.loads(line[line.index("GATE {") + 5:]) for line in run.stdout.splitlines() if "GATE {" in line]
        if run.returncode != 0:
            sys.exit(f"{' '.join(cmd)} failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
    if not checks or not all(c["pass"] for c in checks):
        sys.exit(f"gates missing or failed: {checks}")
    device = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = ROOT / "papers" / "spectral-lm" / "data" / "correctness" / "correctness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commit": head, "command": " && ".join(" ".join(c) for c in COMMANDS), "device": f"gpu0 {device} (and the CPU backend)", "checks": checks}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(checks)} gates")


if __name__ == "__main__":
    main()
