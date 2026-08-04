#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""CodeFormer blind face restoration over brain's D-Bus surface, sweeping `w`.

`w` is CodeFormer's identity-fidelity dial: 0 = maximum quality (the code
prediction alone drives the generator), 1 = maximum fidelity to the input (the
encoder features are injected at full strength). It is a one-element device
buffer read by `scale_add`, NOT a recorded graph constant — so a whole sweep runs
on ONE resident instance with one buffer write per value and no rebuild. The
timings below make that visible: the first call pays the import + upload, every
later `w` costs one forward.

    BRAIN_RESTORE_WEIGHTS=/path/to/codeformer \\
      dbus-run-session -- bash -c '
        brain serve --dbus & sleep 3
        python3 examples/restore/restore_face.py --image face.ppm'

The action takes an ALIGNED face and returns a 512x512 one (the reference CLI's
`cropped_faces/` -> `restored_faces/` step). Pair it with `facenet detect`
(`examples/vision/face_id.py`) to find the face in a full photo first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus, read_fd, sealed_memfd  # noqa: E402
from brain_py.image import load_ppm, save_ppm  # noqa: E402


def mean_abs_diff(a: bytes, b: bytes) -> float:
    import array

    x, y = array.array("f"), array.array("f")
    x.frombytes(a)
    y.frombytes(b)
    n = min(len(x), len(y))
    return sum(abs(x[i] - y[i]) for i in range(n)) / n if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) of an aligned face")
    ap.add_argument("--w", default="0,0.25,0.5,0.75,1", help="comma-separated fidelity values to sweep")
    ap.add_argument("--out", default="/tmp", help="directory for the restored faces")
    args = ap.parse_args()

    img, w, h = load_ppm(args.image)
    meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with BrainDBus() as brain:
        if "restore" not in brain.models():
            print("FATAL: 'restore' not served (set BRAIN_RESTORE_WEIGHTS)", file=sys.stderr)
            return 2

        print(f"{args.image}: {w}x{h} -> 512x512 restored face")
        for value in [float(v) for v in args.w.split(",")]:
            t = time.perf_counter()
            r = brain.run(
                "restore", "restore_face", {"w": value},
                in_fds={"image": sealed_memfd(img)}, in_meta=meta,
            )
            dt = (time.perf_counter() - t) * 1000
            blob = read_fd(r.fds["image"])
            ow, oh = r.result["width"], r.result["height"]
            path = out / f"restored_w{value:.2f}.ppm"
            save_ppm(path, blob, ow, oh, 3)
            # Only comparable when the input is already 512x512; otherwise the
            # geometry differs and the number would be meaningless, so skip it.
            drift = f"  mean|out-in| {mean_abs_diff(img, blob):.5f}" if (w, h) == (ow, oh) else ""
            print(f"  w = {value:.2f}  {ow}x{oh}  {dt:8.1f} ms{drift}  -> {path.name}")

        print("\nHigher w should track the input more closely; lower w should look cleaner.")
        print("scheduler:", brain.stats())
        print("  ('builds' counts every model this server built, not just this one. What the sweep")
        print("   shows is that it did NOT grow: w is a device-buffer write, so every value after")
        print("   the first is answered by the already-resident 512x512 graph — see the timings.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
