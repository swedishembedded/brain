#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Scheduler batching + eviction assertions over brain's D-Bus surface.

Isolated from the generate -> detect -> annotate demo (that demo IS
``examples/dbus/detect_pipeline.py`` — ``scheduler.bats`` runs it directly rather
than maintaining a second, drifting copy). This file is only the scheduler
validation this repo needs automated coverage of:

1. **batch**  — several same-shape requests fired concurrently must coalesce into
   one scheduler group (``Stats.max_batch >= 2``);
2. **evict**  — requesting more distinct instance shapes than fit forces an LRU
   eviction (``Stats.evictions >= 1``).

Both read live ``Stats`` counters — no timing heuristics, no assumptions about
scheduler internals. Exits 0 iff both pass; prints a VALIDATION line
``scheduler.bats`` greps for (``batching=PASS``/``eviction=PASS``).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import BrainError  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402


def fire_concurrent(model: str, action: str, params: dict, count: int) -> None:
    """Submit `count` identical-shape requests at once (own connection each, since
    a jeepney blocking connection is not thread-safe for concurrent calls)."""

    def one(i: int) -> None:
        with BrainDBus() as brain:
            try:
                brain.run(model, action, {**params, "seed": params.get("seed", 0) + i})
            except BrainError as e:
                print(f"  request {i} failed: {e}", file=sys.stderr)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="z-image", help="a streaming-image-capable model (or `mock`)")
    ap.add_argument("--action", default="text2image")
    ap.add_argument("--size", type=int, default=256, help="base width/height")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-n", type=int, default=4, help="concurrent requests fired for the batching probe")
    args = ap.parse_args()

    with BrainDBus() as brain:
        models = brain.models()
        if args.model not in models:
            print(f"FATAL: '{args.model}' not served (models: {models})", file=sys.stderr)
            return 2

        base = {"prompt": "scheduler probe", "width": args.size, "height": args.size, "steps": args.steps, "seed": 0}

        # Warm-up outside the timed/measured section, so the first request's cold
        # build (weight upload + graph build for this shape) never gets counted as
        # "no batching happened" — it is by definition NOT concurrent with itself.
        brain.run(args.model, args.action, base)

        t0 = time.perf_counter()
        fire_concurrent(args.model, args.action, base, args.batch_n)
        batch_wall_s = round(time.perf_counter() - t0, 2)
        after = brain.stats()
        print(
            f"[batch] {args.batch_n} concurrent requests in {batch_wall_s}s; "
            f"max_batch={after['max_batch']} batches={after['batches']} builds={after['builds']}"
        )

        # Eviction: more distinct instance shapes than fit -> an LRU swap.
        t0 = time.perf_counter()
        for delta in (32, 64):
            side = args.size + delta
            brain.run(args.model, args.action, {**base, "width": side, "height": side})
        evict_s = round(time.perf_counter() - t0, 2)
        final = brain.stats()
        print(f"[evict] +2 shapes in {evict_s}s; evictions={final['evictions']} builds={final['builds']} resident={final['resident']}")

    print("\n=== FINAL STATS ===\n ", json.dumps(final))
    ok_batch = after["max_batch"] >= 2
    ok_evict = final["evictions"] >= 1
    print(
        f"\nVALIDATION: batching={'PASS' if ok_batch else 'FAIL'} (max_batch={after['max_batch']}) | "
        f"eviction={'PASS' if ok_evict else 'FAIL'} (evictions={final['evictions']})"
    )
    return 0 if ok_batch and ok_evict else 1


if __name__ == "__main__":
    sys.exit(main())
