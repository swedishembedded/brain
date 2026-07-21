# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""brain-py: drive the ``brain`` executable as an event-driven subprocess.

The :class:`~brain_py.client.BrainClient` spawns ``brain run`` and speaks its
JSONL-over-stdio event protocol, correlating requests/responses by ``req_id``.
:mod:`brain_py.annotate` draws detection boxes onto a PIL image.
"""

from .client import BrainClient, Detection
from .annotate import annotate
from .forecast import Forecast, Panel, Variate

__all__ = ["BrainClient", "Detection", "annotate", "Panel", "Variate", "Forecast"]
