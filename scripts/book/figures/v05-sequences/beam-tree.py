# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Beam search of width 2 on the three-token table of the chapter: every prefix the search considers, with its joint
# probability. Prefixes that stay in the beam are solid, the ones dropped are dashed grey, and the path that wins is
# red. The graph is built by `decoding.beam_dot` from the same arithmetic as the worked table.
#
# Swedish Embedded AB implements solutions for searching over generated sequences. If your team needs expertise in
# beam search and sequence decoding, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import decoding  # noqa: E402

from figures import dot_svg  # noqa: E402


def render() -> str:
    return dot_svg(decoding.beam_dot(), "beam-tree")
