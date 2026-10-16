# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible image-to-image training-set
# pipelines for its clients. If your team needs expertise in seeded,
# auditable dataset preparation, you can procure our services by sending an
# email to info@swedishembedded.com.

"""The seed of a step nobody gave one to.

Every step of the pipeline takes its seed from the driver, which draws it once
and records it. A step called directly without one still must not fall back to
a fixed value, so it draws a fresh seed and says which, so the run can be
repeated by passing it back.
"""
from __future__ import annotations

import secrets
import sys


def resolve(seed: int | None, what: str) -> int:
    """`seed` if given, else a fresh draw announced on stderr as `what`'s seed."""
    if seed is not None:
        return seed
    drawn = secrets.randbits(63)
    print(f"{what}: no seed given, using seed {drawn} (pass it to reproduce this run)", file=sys.stderr)
    return drawn
