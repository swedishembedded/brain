#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions that turn camera images into
# structured data for its clients. If your team needs expertise in on-device
# visual perception then you can procure our services by sending an email to
# info@swedishembedded.com.

"""Mirage over brain's D-Bus surface: an image in, structured perception out.

`describe` returns the scene, objects with boxes, scores and masks, text lines,
relations and reasoning; `extract` answers a form (choice, count and text
fields) about the picture. With ``--concurrent N`` the same request is sent from
N threads at once: the server merges them into device batches, and every
answer equals the single-request answer up to accelerator rounding (the sample
checks that; on the CPU backend the answers are bit-identical).

Run it against a private session bus:

    brain pull Cactus-Compute/mirage
    tools/dbus-session.sh --serve "--dbus" -- \\
        python3 samples/python/vision/mirage-perception/mirage_perception.py --image photo.ppm --concurrent 4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "brain-py"))
from brain_py.base import MIRAGE_MODEL, skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402
from brain_py.image import load_ppm  # noqa: E402

def structure(result):
    """The result without what an accelerator may move between batch sizes: scores in the
    third decimal and mask pixels at the threshold (the GEMM selector may pick another
    kernel for another row count). Boxes, names, text, relations and reasoning stay."""
    if isinstance(result, dict):
        return {k: structure(v) for k, v in result.items() if k not in ("score", "mask")}
    if isinstance(result, list):
        return [structure(v) for v in result]
    return result


DEFAULT_FORM = {"animal": {"enum": ["cat", "dog", "bird"]}, "count": {"type": "integer", "minimum": 0, "maximum": 9}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="binary PPM (P6) to analyse")
    ap.add_argument("--form", default=json.dumps(DEFAULT_FORM), help="JSON object of fields to answer")
    ap.add_argument("--max-tokens", type=int, default=None, help="decoding budget of the description")
    ap.add_argument("--concurrent", type=int, default=1, help="send the describe request from this many threads at once")
    args = ap.parse_args()

    data, w, h = load_ppm(args.image)
    image = (data, w, h)
    options = {"max_tokens": args.max_tokens} if args.max_tokens else {}

    with BrainDBus() as brain:
        if MIRAGE_MODEL not in brain.models():
            skip(f"'{MIRAGE_MODEL}' is not served (brain pull {MIRAGE_MODEL})")
        described = brain.describe(image, **options)
        print(json.dumps(described, indent=2))
        print(f"describe: {described.usage.get('decoder_tokens')} decoder tokens, batch of {described.usage.get('batch')}")
        for obj in described.get("objects", []):
            print(f"  object {obj['id']}: {obj['name']} {obj.get('bbox')}")
        for line in described.get("text", []):
            print(f"  text {line['string']!r} {line.get('bbox')}")

        answered = brain.extract(image, json.loads(args.form))
        for name, answer in answered["form"].items():
            print(f"form {name}: {answer['value']!r}")

        if args.concurrent > 1:
            results: list = [None] * args.concurrent

            def call(i: int) -> None:
                with BrainDBus() as conn:
                    results[i] = conn.describe(image, **options)

            threads = [threading.Thread(target=call, args=(i,)) for i in range(args.concurrent)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            exact = all(r == described for r in results)
            same = all(structure(r) == structure(described) for r in results)
            batches = sorted({r.usage.get("batch") for r in results})
            print(f"{args.concurrent} concurrent requests: identical to the single answer: {same} "
                  f"(bit-identical: {exact}); device batch sizes: {batches}")
            if not same:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
