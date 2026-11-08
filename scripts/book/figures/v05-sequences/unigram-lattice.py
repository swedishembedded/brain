# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The lattice of segmentations of "bakers" under the chapter's unigram vocabulary: one node per position between
# letters, one edge for every vocabulary piece that matches there, labelled with the piece and its natural log
# probability. The path Viterbi's search finds is drawn in red. The graph is built by
# `tokenization.lattice_dot` from the same probabilities as the worked table.
#
# Swedish Embedded AB implements solutions for segmenting text under a probabilistic vocabulary. If your team
# needs expertise in subword tokenizers, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tokenization  # noqa: E402

from figures import dot_svg  # noqa: E402


def render() -> str:
    return dot_svg(tokenization.lattice_dot("bakers"), "unigram-lattice")
