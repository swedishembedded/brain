#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Time-series forecasting over brain's D-Bus surface.

Sends a context series to `brain serve --dbus` as a sealed memfd and reads the
forecast distribution back as a file descriptor — the same generic `Run` + fd
path every brain model uses (see .agents/rules/serving-contract.md). Works for all three
forecasting foundation models:

  chronos2  probabilistic universal forecaster  -> [levels, horizon] quantiles
  fincast   financial forecaster                 -> [horizon, 1+levels] (col 0 mean)
  kronos    autoregressive OHLCV forecaster      -> [horizon, feat] sample bars

The wire format is raw little-endian float32 with an explicit `shape` in the
blob meta — no per-model protocol.

Examples:
  # chronos2 (needs `BRAIN_CHRONOS2=... brain serve --dbus` running):
  python3 examples/forecast/forecast_client.py --model chronos2 --horizon 64
  # fincast on the daily bucket:
  python3 examples/forecast/forecast_client.py --model fincast --freq 0
  # kronos on synthetic OHLCV:
  python3 examples/forecast/forecast_client.py --model kronos --horizon 32
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

try:
    import brain_py  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain-py"))
from brain_py.base import skip  # noqa: E402
from brain_py.dbus import BrainDBus  # noqa: E402


def f32le(values) -> bytes:
    return struct.pack("<%df" % len(values), *[float(v) for v in values])


def unpack_f32(b: bytes) -> list[float]:
    return list(struct.unpack("<%df" % (len(b) // 4), b))


def synth_series(n: int) -> list[float]:
    """A gentle trend + seasonality — enough for a foundation model to lock onto."""
    return [100.0 + 0.05 * i + 5.0 * math.sin(i * 0.1) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="brain/chronos2", choices=["brain/chronos2", "brain/fincast", "brain/kronos", "brain/mock"],
                     help="a forecast-capable model (brain/mock needs no weights — a quick, deterministic check)")
    ap.add_argument("--horizon", type=int, default=64, help="steps to forecast")
    ap.add_argument("--context", type=int, default=256, help="synthetic context length (ignored with --series)")
    ap.add_argument("--freq", type=int, default=0, help="fincast frequency bucket (0 daily / 1 weekly / 2 monthly)")
    ap.add_argument("--series", help="file of whitespace/comma-separated f32 to use as context (else synthetic)")
    ap.add_argument("--bus", default="SESSION", help="SESSION or SYSTEM")
    args = ap.parse_args()

    ctx = (
        [float(x) for x in open(args.series).read().replace(",", " ").split()]
        if args.series
        else synth_series(args.context)
    )
    if len(ctx) < 8:
        sys.exit("context too short (need >= 8 points)")

    # Every model takes a plain f32 series here. kronos expands a univariate
    # series to OHLCV bars server-side; pass full [T, feat] bars via --series-2d
    # for a real OHLCV forecast (not needed for the demo).
    payload, shape = f32le(ctx), [len(ctx)]

    params: dict = {"horizon": args.horizon}
    if args.model == "brain/fincast":
        params["freq"] = args.freq

    with BrainDBus(bus=args.bus) as brain:
        served = brain.models()
        if args.model not in served:
            env = {"brain/chronos2": "BRAIN_CHRONOS2=<weights>", "brain/fincast": "BRAIN_FINCAST=<weights>",
                   "brain/kronos": "BRAIN_KRONOS_TOKENIZER=<dir> BRAIN_KRONOS_DECODER=<dir>",
                   "brain/mock": "BRAIN_MOCK=1"}[args.model]
            skip(f"model {args.model!r} is not served (served: {served}); start it with: {env} brain serve --dbus")
        out = brain.run(
            args.model, "forecast", params,
            blobs={"context": payload},
            meta={"context": {"media": "bytes", "shape": shape}},
        )

    fmeta = out.meta["forecast"]["meta"]
    fshape, kind, levels = fmeta["shape"], fmeta["kind"], fmeta.get("levels", [])
    data = unpack_f32(out.blobs["forecast"])

    print(f"model={out.outputs.get('model')}  device={out.outputs.get('device')}  horizon={out.outputs.get('horizon')}")
    print(f"forecast: shape={fshape}  kind={kind}  levels={levels if levels else '(samples)'}")

    # Extract a single point path for a readable summary.
    if kind == "quantiles" and levels:  # chronos2: [Q, horizon] quantile-major -> median row
        q, h = fshape
        j = min(range(len(levels)), key=lambda k: abs(levels[k] - 0.5))
        path = [data[j * h + t] for t in range(h)]
        label = "median"
    elif kind == "mean+quantiles":  # fincast: [horizon, num_outputs], col 0 = mean
        h, no = fshape
        path = [data[t * no + 0] for t in range(h)]
        label = "mean"
    else:  # kronos samples: [horizon, feat] -> close column (index 3)
        h, f = fshape
        col = min(3, f - 1)
        path = [data[t * f + col] for t in range(h)]
        label = "close"

    show = [round(v, 3) for v in path[: min(12, len(path))]]
    print(f"{label} path (first {len(show)} of {len(path)}): {show}")


if __name__ == "__main__":
    main()
