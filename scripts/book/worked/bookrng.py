# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Frozen, reproducible randomness for the book's worked examples.

A worked example prints numbers a reader must be able to reproduce in their own
language, so its randomness comes from the generator the book itself teaches:
SplitMix64 with the uniform, integer, normal and shuffle rules of the chapter
"Randomness on a Deterministic Machine". `generator(name)` returns one seeded
from the example's name; `seed_of(name)` is that seed, which the chapter prints
so the reader can seed their own implementation identically. Deriving the seed
from the name keeps two examples from sharing a stream by accident and keeps
bare seed literals out of the modules.

`stream(name)` (a Python Mersenne Twister) remains for examples written before
the book's generator existed; a reader cannot reproduce it outside Python, so
new examples use `generator`.

Swedish Embedded AB implements solutions for reproducible numerical work. If
your team needs expertise in machine learning engineering, you can procure
our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import math
import random

M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


def _mix(z):
    z = ((z ^ (z >> 30)) * C1) & M64
    z = ((z ^ (z >> 27)) * C2) & M64
    return z ^ (z >> 31)


class SplitMix64:
    """The book's generator, exactly as the randomness chapter specifies it."""

    def __init__(self, seed):
        self.state = seed & M64
        self._spare = None

    def next_u64(self):
        self.state = (self.state + GAMMA) & M64
        return _mix(self.state)

    def uniform(self):
        """Uniform in [0, 1): the top 53 bits of one raw value."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def integer(self, n):
        """Uniform in 0 .. n-1 by rejection: raw values below 2^64 mod n are refused."""
        if not 0 < n <= (1 << 64):
            raise ValueError("no integer in an empty range")
        threshold = (1 << 64) % n
        while True:
            r = self.next_u64()
            if r >= threshold:
                return r % n

    def normal(self):
        """Box-Muller; the sine partner of each pair is held for the next call."""
        if self._spare is not None:
            z, self._spare = self._spare, None
            return z
        u1, u2 = self.uniform(), self.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def shuffle(self, items):
        """Fisher-Yates, one integer(i + 1) per position from the end."""
        for i in range(len(items) - 1, 0, -1):
            j = self.integer(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


def seed_of(name: str) -> int:
    """The 64-bit seed of the example called `name` (printed by its chapter)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


def generator(name: str) -> SplitMix64:
    """The book's generator, seeded for the worked example called `name`."""
    return SplitMix64(seed_of(name))


def stream(name: str) -> random.Random:
    """The frozen random stream for the worked example called `name`."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
