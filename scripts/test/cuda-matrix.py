#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements validation matrices that show which parts of a
# large codebase actually run on a given accelerator, for its clients. If your
# team needs expertise in bringing a model catalogue to a new GPU backend then
# you can procure our services by sending an email to info@swedishembedded.com.

"""Run every workspace crate's tests on the CUDA backend and record the result.

Each crate is run on its own with `BRAIN_BACKEND=cuda`, so a crate that cannot
build, hangs, or fails is reported by name rather than hiding the rest. The
output is one JSON document: per crate a status (`ok`, `failed`, `timeout`,
`build-failed`), the count of passed, failed and ignored tests, and the names of
the failing tests. A crate whose tests do not touch the GPU is still run: that
is what lets "this model has no CUDA-dependent test" be told apart from "this
model passes on CUDA".

    scripts/test/cuda-matrix.py                     # every crate, to cuda-matrix.json
    scripts/test/cuda-matrix.py -o out.json qwen3 gpt2
    scripts/test/cuda-matrix.py --timeout 1800 --jobs 1

It needs a CUDA device and a toolkit lane on the environment (see
`make cuda/install`); without them every GPU test skips and the matrix says so
by reporting zero failures on tests that never ran.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESULT = re.compile(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored")
FAILED = re.compile(r"^test (\S+) \.\.\. FAILED", re.M)


def workspace_crates() -> list[str]:
    """Package names of every workspace member that has tests or a lib."""
    meta = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    packages = json.loads(meta.stdout)["packages"]
    return sorted(p["name"] for p in packages if any(t["kind"][0] in ("lib", "test", "bin") for t in p["targets"]))


def run_crate(package: str, timeout: int, extra_env: dict) -> dict:
    env = {**os.environ, "BRAIN_BACKEND": "cuda", **extra_env}
    started = time.monotonic()
    try:
        proc = subprocess.run(
            ["cargo", "test", "--release", "--offline", "--no-fail-fast", "-p", package],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return {"status": "timeout", "seconds": timeout, "tail": (e.stdout or b"")[-400:].decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")[-400:]}
    out = proc.stdout + proc.stderr
    results = RESULT.findall(out)
    passed = sum(int(r[1]) for r in results)
    failed = sum(int(r[2]) for r in results)
    ignored = sum(int(r[3]) for r in results)
    if not results and proc.returncode != 0:
        status = "build-failed"
    elif failed or proc.returncode != 0:
        status = "failed"
    else:
        status = "ok"
    return {
        "status": status,
        "seconds": round(time.monotonic() - started),
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "failed_tests": sorted(set(FAILED.findall(out))),
        "build_error": out[-600:] if status == "build-failed" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crates", nargs="*", help="package names; default every workspace package")
    ap.add_argument("-o", "--output", default="cuda-matrix.json")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per crate")
    args = ap.parse_args()

    crates = args.crates or workspace_crates()
    report = {}
    for name in crates:
        print(f"[{len(report) + 1}/{len(crates)}] {name} ...", flush=True)
        report[name] = run_crate(name, args.timeout, {})
        r = report[name]
        print(f"    {r['status']}: {r.get('passed', 0)} passed, {r.get('failed', 0)} failed", flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    bad = {k: v for k, v in report.items() if v["status"] != "ok"}
    print(f"\n{len(report) - len(bad)} of {len(report)} crates pass on CUDA")
    for k, v in sorted(bad.items()):
        print(f"  {k}: {v['status']} {v.get('failed_tests', [])[:4]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
