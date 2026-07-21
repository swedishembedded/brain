# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Forecasting types for the Python client: :class:`Panel` (the input) and
:class:`Forecast` (the distribution-valued output).

The wire format carries bulk numeric fields as base64 of little-endian float32
with an explicit ``shape`` — the same layout as ``events::bytes`` on the Rust
side. Small metadata stays plain JSON.

The honesty contract of the API surfaces here: :attr:`Forecast.native_representation`
is what the model actually emitted, and any field brain *derived* is flagged, so
a tool can refuse to size a position off, say, an interpolated tail.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass, field
from typing import Optional


def encode_f32(values) -> str:
    """Base64 of little-endian float32 — the wire encoding for numeric arrays."""
    buf = struct.pack("<%df" % len(values), *[float(v) for v in values])
    return base64.b64encode(buf).decode("ascii")


def decode_f32(b64: str) -> list[float]:
    """Inverse of :func:`encode_f32`."""
    raw = base64.b64decode(b64)
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


@dataclass
class Variate:
    """One named series within a panel item."""

    name: str
    role: str = "target"  # target | past_covariate | known_future | static
    kind: str = "continuous"  # continuous | categorical
    data: list = field(default_factory=list)
    future: Optional[list] = None  # known_future only, horizon-length
    observed: Optional[list] = None
    cardinality: Optional[int] = None

    def to_wire(self) -> dict:
        o = {
            "name": self.name,
            "role": self.role,
            "kind": self.kind,
            "data": encode_f32(self.data),
        }
        if self.future is not None:
            o["future"] = encode_f32(self.future)
        if self.observed is not None:
            o["observed"] = encode_f32(self.observed)
        if self.cardinality is not None:
            o["cardinality"] = int(self.cardinality)
        return o


class Panel:
    """The forecasting input: named, role-tagged series for one or more items."""

    def __init__(self, freq: str, start: Optional[str] = None) -> None:
        self.freq = freq
        self.start = start
        # items: item_id -> list[Variate]
        self._items: dict[str, list[Variate]] = {}

    # -- builders ------------------------------------------------------------

    @classmethod
    def from_series(cls, values, item_id: str = "series", freq: str = "1d",
                    name: str = "y", start: Optional[str] = None) -> "Panel":
        """A univariate panel from one target series."""
        p = cls(freq, start)
        p.add_target(item_id, name, values)
        return p

    @classmethod
    def from_ohlcv(cls, df, item_id: str, freq: str = "1d",
                   start: Optional[str] = None,
                   columns=("open", "high", "low", "close", "volume")) -> "Panel":
        """A panel from an OHLCV table (any object with column access like a
        pandas DataFrame or a dict of sequences). OHLCV is not special — it is
        just named continuous variates; ``close`` etc. become targets and
        ``volume`` a past covariate.
        """
        p = cls(freq, start)
        for col in columns:
            series = df[col]
            values = list(series)
            role = "past_covariate" if col == "volume" else "target"
            p.add_variate(item_id, Variate(name=col, role=role, data=values))
        return p

    # -- mutation ------------------------------------------------------------

    def add_variate(self, item_id: str, variate: Variate) -> "Panel":
        self._items.setdefault(item_id, []).append(variate)
        return self

    def add_target(self, item_id: str, name: str, values) -> "Panel":
        return self.add_variate(item_id, Variate(name=name, role="target", data=list(values)))

    def add_covariate(self, name: str, values, role: str = "past",
                      future=None, item_id: Optional[str] = None,
                      kind: str = "continuous", cardinality: Optional[int] = None) -> "Panel":
        """Attach a covariate. ``role`` is ``"past"`` or ``"known_future"``
        (``future`` supplies its horizon values). Defaults to the sole item when
        the panel has exactly one.
        """
        if item_id is None:
            if len(self._items) != 1:
                raise ValueError("item_id required when the panel has != 1 item")
            item_id = next(iter(self._items))
        wire_role = {"past": "past_covariate", "known_future": "known_future"}.get(role, role)
        self.add_variate(item_id, Variate(
            name=name, role=wire_role, kind=kind, data=list(values),
            future=list(future) if future is not None else None,
            cardinality=cardinality,
        ))
        return self

    def add_calendar(self, ctx_dates, fut_dates, item_id: Optional[str] = None) -> "Panel":
        """Attach Kronos's time features (minute/hour/weekday/day/month) computed
        from the context bar dates and the forecast-horizon dates. ``ctx_dates`` /
        ``fut_dates`` are sequences of datetime-like objects (``datetime`` or
        pandas ``Timestamp``). Without this, Kronos runs calendar-agnostic (a touch
        more extreme than the reference). weekday is Monday=0, matching the ref.
        """
        def feats(dates):
            return {
                "minute": [int(d.minute) for d in dates],
                "hour": [int(d.hour) for d in dates],
                "weekday": [int(d.weekday()) for d in dates],
                "day": [int(d.day) for d in dates],
                "month": [int(d.month) for d in dates],
            }
        cf, ff = feats(ctx_dates), feats(fut_dates)
        for name in ("minute", "hour", "weekday", "day", "month"):
            self.add_covariate(name, cf[name], role="known_future", future=ff[name],
                               item_id=item_id, kind="categorical")
        return self

    def to_wire(self) -> dict:
        return {
            "freq": self.freq,
            "start": self.start,
            "items": [
                {"item_id": iid, "variates": [v.to_wire() for v in vs]}
                for iid, vs in self._items.items()
            ],
        }


@dataclass
class Block:
    """A shaped numeric block, with provenance."""

    shape: list
    data: list
    derived: bool = False
    method: str = ""

    @classmethod
    def from_wire(cls, o: Optional[dict]) -> Optional["Block"]:
        if not o:
            return None
        return cls(
            shape=list(o.get("shape", [])),
            data=decode_f32(o.get("data", "")),
            derived=bool(o.get("derived", False)),
            method=o.get("method", ""),
        )

    def row(self, t: int) -> list:
        """Row ``t`` of a 2-D block."""
        cols = self.shape[1] if len(self.shape) > 1 else 1
        return self.data[t * cols:(t + 1) * cols]


class TargetForecast:
    """The forecast for one target series — a distribution over the horizon."""

    def __init__(self, wire: dict) -> None:
        self.item_id = wire.get("item_id", "")
        self.name = wire.get("name", "")
        self.levels = list(wire.get("levels", []))
        self.quantiles = Block.from_wire(wire.get("quantiles"))
        self.samples = Block.from_wire(wire.get("samples"))
        self.mean = Block.from_wire(wire.get("mean"))
        self.distribution = Block.from_wire(wire.get("distribution"))
        self.dist_family = wire.get("dist_family", "")
        self.classes = Block.from_wire(wire.get("classes"))
        self.class_labels = list(wire.get("class_labels", []))

    @property
    def horizon(self) -> int:
        if self.quantiles:
            return self.quantiles.shape[0]
        if self.mean:
            return self.mean.shape[0]
        if self.samples:
            return self.samples.shape[1]
        return 0

    def _level_index(self, level: float) -> int:
        # nearest quantile level
        best, bi = 1e9, 0
        for i, lv in enumerate(self.levels):
            if abs(lv - level) < best:
                best, bi = abs(lv - level), i
        return bi

    def median(self) -> list:
        """Median (or mean) point path."""
        if self.mean:
            return list(self.mean.data)
        if self.quantiles and self.levels:
            j = self._level_index(0.5)
            q = self.quantiles.shape[1]
            return [self.quantiles.data[t * q + j] for t in range(self.horizon)]
        raise ValueError("no point representation available")

    def quantile(self, level: float) -> list:
        """The predicted path at a given quantile level."""
        if not self.quantiles:
            raise ValueError("no quantile representation available")
        j = self._level_index(level)
        q = self.quantiles.shape[1]
        return [self.quantiles.data[t * q + j] for t in range(self.horizon)]

    def interval(self, level: float = 0.8):
        """Central interval covering ``level`` probability, as ``(lo, hi)`` paths.
        Uses the outermost available quantiles when the exact tails aren't
        present.
        """
        lo = (1.0 - level) / 2.0
        hi = 1.0 - lo
        return self.quantile(lo), self.quantile(hi)

    def sample_paths(self):
        """The ``(n_samples, horizon)`` sample matrix, or ``None``."""
        if not self.samples:
            return None
        n, h = self.samples.shape[0], self.samples.shape[1]
        return [self.samples.data[i * h:(i + 1) * h] for i in range(n)]


class Forecast:
    """A complete forecast result, with per-target access and single-target
    conveniences (the common case)."""

    def __init__(self, wire: dict) -> None:
        self.model = wire.get("model", "")
        self.model_version = wire.get("model_version", "")
        self.native_representation = wire.get("native_representation", "")
        self.horizon = int(wire.get("horizon", 0))
        self.freq = wire.get("freq", "")
        self.targets = [TargetForecast(t) for t in wire.get("targets", [])]

    def target(self, name: Optional[str] = None) -> TargetForecast:
        if name is None:
            if not self.targets:
                raise ValueError("forecast has no targets")
            return self.targets[0]
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(name)

    # single-target conveniences
    def median(self, name: Optional[str] = None) -> list:
        return self.target(name).median()

    def interval(self, level: float = 0.8, name: Optional[str] = None):
        return self.target(name).interval(level)

    def samples(self, name: Optional[str] = None):
        return self.target(name).sample_paths()

    @property
    def derived(self) -> dict:
        """Which representations of the first target were derived, and how."""
        t = self.target()
        out = {}
        for rep, blk in (("quantiles", t.quantiles), ("samples", t.samples),
                         ("mean", t.mean), ("distribution", t.distribution)):
            if blk and blk.derived:
                out[rep] = blk.method
        return out
