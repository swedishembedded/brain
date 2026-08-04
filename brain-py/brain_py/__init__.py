# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""brain-py: the Python client for the ``brain`` edge-AI runtime.

brain serves a **capability model** — every model advertises actions
(``generate``, ``embed``, ``text2image``, ``transcribe``, …) taking typed params
and named binary blobs. This package speaks that model over two transports that
share ONE high-level API, so you can switch transport without rewriting:

* :class:`~brain_py.dbus.BrainDBus` — **the default**: the ``com.swedishembedded.Brain1``
  D-Bus surface of ``brain serve --dbus``, exchanging bulk data as file descriptors.
* :class:`~brain_py.client.BrainStdio` — the JSONL-on-stdio transport that drives a
  ``brain run`` subprocess and correlates requests by ``req_id``. It additionally
  offers the ``brain run`` legacy verbs :meth:`~brain_py.client.BrainStdio.detect`,
  :meth:`~brain_py.client.BrainStdio.converse`, ``forecast`` and ``backtest``.

Pick one with the top-level :func:`Brain` factory (D-Bus unless you ask for JSONL):

    from brain_py import Brain

    with Brain() as brain:                      # D-Bus (the default)
        print(brain.models())
        print(brain.generate(prompt="hello", model="mock"))

    with Brain(transport="jsonl") as brain:      # spawn `brain run` over stdio
        print(brain.generate(prompt="hello", model="mock"))

Both expose the same :meth:`~brain_py.base.BrainBase.run` /
:meth:`~brain_py.base.BrainBase.subscribe` primitives and the ``generate`` /
``chat`` / ``embed`` / ``text2image`` convenience wrappers (see
:class:`~brain_py.base.BrainBase`); every action returns an
:class:`~brain_py.base.Outcome`.

Helpers: :mod:`brain_py.image` saves/annotates brain's raw HWC-f32 image blobs
(no PIL); :func:`brain_py.annotate.annotate` draws boxes onto a PIL image.
"""

from typing import Any

from .annotate import annotate
from .base import BrainBase, BrainError, Outcome
from .client import BrainClient, BrainStdio, Detection
from .dbus import BrainDBus, RunResult, read_fd, sealed_memfd
from .forecast import Forecast, Panel, Variate
from .image import draw_boxes, save_ppm


def Brain(transport: str = "dbus", **kwargs: Any) -> BrainBase:
    """Construct a brain client for the requested transport (**D-Bus by default**).

    * ``transport="dbus"`` (default) → :class:`~brain_py.dbus.BrainDBus`, connecting
      to ``com.swedishembedded.Brain1`` on the session bus. Extra kwargs
      (e.g. ``bus="SYSTEM"``) pass through.
    * ``transport="jsonl"`` (aliases: ``"stdio"``) → :class:`~brain_py.client.BrainStdio`,
      spawning a ``brain run`` subprocess. Extra kwargs (``yolo=``, ``device=``,
      ``forecast=``, …) pass through.

    Both share the high-level capability API, so switching transport needs no
    other code change.
    """
    t = transport.lower()
    if t in ("dbus", "d-bus", "bus"):
        return BrainDBus(**kwargs)
    if t in ("jsonl", "stdio", "stdout"):
        return BrainStdio(**kwargs)
    raise ValueError(f"unknown transport {transport!r}; use 'dbus' (default) or 'jsonl'")


__all__ = [
    # entry points
    "Brain",
    "BrainDBus",
    "BrainStdio",
    "BrainClient",
    # shared model
    "BrainBase",
    "BrainError",
    "Outcome",
    # dbus fd helpers
    "RunResult",
    "read_fd",
    "sealed_memfd",
    # detection / imaging
    "Detection",
    "annotate",
    "draw_boxes",
    "save_ppm",
    # forecasting
    "Panel",
    "Variate",
    "Forecast",
]
