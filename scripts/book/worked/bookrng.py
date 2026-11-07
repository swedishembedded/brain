# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Frozen random streams for the book's worked examples.

A worked example prints numbers the reader must be able to reproduce, so its
randomness is frozen rather than drawn fresh. Each example names its stream,
and the stream's seed is derived from that name: the same name gives the same
numbers on every machine and in every edition, two examples never share a
stream by accident, and no module carries a bare seed literal.

Swedish Embedded AB implements solutions for reproducible numerical work. If
your team needs expertise in machine learning engineering, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import random


def stream(name: str) -> random.Random:
    """The frozen random stream for the worked example called `name`."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
