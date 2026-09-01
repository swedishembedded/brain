#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The cooling-loop scenario (see `tools/forecast/make_cooling_loop.py` and
`crates/timesfm3/examples/cooling_loop.rs`) through brain's SERVED path: a
running `brain serve --dbus` process, not the library API.

The generic D-Bus `forecast` action carries one series - it has no wire
format yet for TimesFM-3's own past/known-future covariates, so this shows
the univariate slice of the same problem (the return-temperature series
alone) against a seasonal-naive baseline computed locally. For the full
multivariate story - the ambient forecast and the shift schedule actually
informing the prediction - see the Rust example, which calls the model
directly and needs no server.

Run under a private session bus (needs no system config):

    python3 tools/forecast/make_cooling_loop.py --out cooling_loop.csv
    dbus-run-session -- bash -c '
      BRAIN_TIMESFM3=timesfm3.safetensors brain serve --dbus --device cpu &
      sleep 2
      python3 examples/forecast/cooling_loop.py cooling_loop.csv
    '
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402

CONTEXT, HORIZON, TRIP = 576, 128, 45.0  # patch-aligned, matching the Rust example


def f32le(values) -> bytes:
    return struct.pack("<%df" % len(values), *[float(v) for v in values])


def trip_hour(path) -> str:
    for h, v in enumerate(path):
        if v > TRIP:
            return f"hour {h}"
    return "never"


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "cooling_loop.csv"
    rows = [line.split(",") for line in open(csv_path).read().splitlines()[1:]]
    t_return = [float(r[1]) for r in rows]
    if len(t_return) < CONTEXT + HORIZON:
        sys.exit(f"{csv_path}: need >= {CONTEXT + HORIZON} hourly bars, got {len(t_return)}")
    ctx, actual = t_return[:CONTEXT], t_return[CONTEXT : CONTEXT + HORIZON]

    seasonal = [ctx[len(ctx) - 24 + (h % 24)] for h in range(HORIZON)]

    with BrainDBus() as brain:
        served = brain.models()
        if "brain/timesfm3" not in served:
            skip(f"brain/timesfm3 is not served (served: {served}); start it with: BRAIN_TIMESFM3=<weights> brain serve --dbus")
        out = brain.run(
            "brain/timesfm3", "forecast", {"horizon": HORIZON},
            blobs={"context": f32le(ctx)}, meta={"context": {"media": "bytes", "shape": [len(ctx)]}},
        )
    shape, levels = out.meta["forecast"]["meta"]["shape"], out.meta["forecast"]["meta"]["levels"]
    data = list(struct.unpack("<%df" % (len(out.blobs["forecast"]) // 4), out.blobs["forecast"]))
    h, nq = shape
    j = min(range(len(levels)), key=lambda k: abs(levels[k] - 0.5))
    median = [data[t * nq + j] for t in range(h)]

    mae = lambda p: sum(abs(a - b) for a, b in zip(p, actual)) / len(actual)  # noqa: E731
    print(f"cooling loop (served, univariate): {CONTEXT}h context -> {HORIZON}h forecast, trip threshold {TRIP} C")
    print(f"  actual                              predicted trip: {trip_hour(actual)}")
    print(f"  seasonal naive       MAE {mae(seasonal):6.2f}   predicted trip: {trip_hour(seasonal)}")
    print(f"  timesfm3 (served)    MAE {mae(median):6.2f}   predicted trip: {trip_hour(median)}")


if __name__ == "__main__":
    main()
