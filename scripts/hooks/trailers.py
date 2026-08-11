# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Strip Co-Authored-By:/Claude-Session: trailer lines from a commit message.

Shared by the commit-msg hook (silently cleans every new commit message) and
the pre-push hook (fails the push if one somehow survived — e.g. a commit
made without the hook installed, or merged/cherry-picked in from elsewhere).
"""
import re

TRAILER_RE = re.compile(rb"(?i)^(Co-Authored-By|Claude-Session):")


def strip_trailers(message: bytes) -> bytes:
    lines = message.split(b"\n")
    kept = [ln for ln in lines if not TRAILER_RE.match(ln)]
    new = b"\n".join(kept)
    new = re.sub(rb"\n{3,}", b"\n\n", new)
    new = new.rstrip(b"\n") + b"\n" if new.strip() else new
    return new


def has_trailer(message: bytes) -> bool:
    return any(TRAILER_RE.match(ln) for ln in message.split(b"\n"))
