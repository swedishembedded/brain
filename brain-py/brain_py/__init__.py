# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""brain-py: Python clients + helpers for the ``brain`` executable.

- :class:`~brain_py.client.BrainClient` spawns ``brain run`` and speaks its
  JSONL-over-stdio event protocol, correlating requests/responses by ``req_id``.
- :class:`~brain_py.dbus.BrainDBus` (in the ``dbus`` extra — needs ``jeepney``)
  drives ``brain serve --dbus``, exchanging images/streams/results as file
  descriptors.
- :mod:`brain_py.image` saves/annotates brain's raw HWC-f32 image blobs (no PIL);
  :func:`brain_py.annotate.annotate` draws boxes onto a PIL image.
"""

from .annotate import annotate
from .client import BrainClient, Detection
from .forecast import Forecast, Panel, Variate
from .image import draw_boxes, save_ppm

# `brain_py.dbus` is intentionally NOT imported here: it depends on the optional
# `jeepney` package, so importing the base library never requires it. Use
# ``from brain_py.dbus import BrainDBus``.

__all__ = [
    "BrainClient",
    "Detection",
    "annotate",
    "draw_boxes",
    "save_ppm",
    "Panel",
    "Variate",
    "Forecast",
]
