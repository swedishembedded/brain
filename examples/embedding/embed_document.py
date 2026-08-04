#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Long-context embeddings over brain's D-Bus interface (LFM2.5-Encoder).

The document travels as a **file descriptor** (sealed memfd), not as bytes
marshalled through D-Bus — the same fd path every brain blob uses — and the
per-token hidden states come back the same way. Concurrency goes through
brain's residency executor: equal-length requests batch into one true batched
forward; different devices run in parallel lanes.

Run under a private session bus (weights + tokenizer via env):

    dbus-run-session -- bash -c '
      BRAIN_LFM=out/lfm-230m.weights \
      BRAIN_LFM_TOKENIZER=/path/to/tokenizer.json \
      brain serve --dbus & sleep 2
      python3 examples/embedding/embed_document.py --input README.md --concurrent 4'

Requires: jeepney (the same dependency as examples/dbus) — `pip install -e brain-py`.
"""
from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.dbus import BrainDBus  # noqa: E402


def embed_once(brain: BrainDBus, text: bytes, label: str) -> tuple[int, int, float]:
    """One embed request: text in as an input blob, [n, d] f32 hidden states out.

    `run()` seals the input into a memfd and reads the output fd back into bytes
    for us — no manual memfd/ctypes plumbing needed on the client side (brain-py
    already implements that once, in `brain_py.dbus.sealed_memfd`/`read_fd`)."""
    t0 = time.monotonic()
    out = brain.run("lfm", "embed", params={}, blobs={"text": text}, meta={"text": {"media": "text"}})
    dt = time.monotonic() - t0
    emb_meta = out.meta["embeddings"]
    shape = (emb_meta.get("meta") or {}).get("shape", [0, 0])
    raw = out.blobs["embeddings"]
    n, d = int(shape[0]), int(shape[1])
    assert len(raw) == n * d * 4, f"fd payload {len(raw)} != {n}x{d} f32"
    # Mean-pool on the client as a sanity check on the fd payload.
    floats = struct.unpack(f"<{n * d}f", raw)
    mean0 = sum(floats[i * d] for i in range(n)) / max(n, 1)
    print(
        f"  [{label}] {n} tokens x {d} dim over {emb_meta['transport']} "
        f"({len(raw) >> 10} KiB) in {dt:.2f}s; mean[0]={mean0:+.4f} "
        f"(service: {out.outputs.get('tokens')} tokens)"
    )
    return n, d, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="text file to embed (long context welcome)")
    ap.add_argument("--concurrent", type=int, default=1, help="issue N identical requests concurrently")
    args = ap.parse_args()

    text = Path(args.input).read_bytes()
    print(f"document: {args.input} ({len(text) >> 10} KiB)")

    with BrainDBus() as brain:
        models = brain.models()
        if "lfm" not in models:
            print(f"lfm not served (models: {models}); set BRAIN_LFM + BRAIN_LFM_TOKENIZER", file=sys.stderr)
            return 1

        # Warm-up (weight upload + graph build for this length) — never timed.
        print("warm-up:")
        embed_once(brain, text, "warm")

        print(f"{args.concurrent} concurrent request(s):")
        t0 = time.monotonic()
        times: list[float] = []
        errs: list[str] = []

        def worker(i: int) -> None:
            # One bus connection per thread (a jeepney blocking connection is not
            # thread-safe for concurrent calls).
            try:
                with BrainDBus() as b:
                    times.append(embed_once(b, text, f"req{i}")[2])
            except Exception as e:  # noqa: BLE001 — report, don't crash the demo
                errs.append(f"req{i}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrent)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0
        for e in errs:
            print(f"  ERROR {e}", file=sys.stderr)
        if times:
            print(
                f"wall {wall:.2f}s for {len(times)} requests "
                f"(mean per-request {sum(times) / len(times):.2f}s; batching/lanes = wall < sum)"
            )
        return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
