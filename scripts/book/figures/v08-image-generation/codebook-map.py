# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two-number latent space of the chapter's tokenizer after training: the encoder rows of the 128 test readings (one point for each
# of the four positions) and the 16 codes that they are replaced by. The data are `vector_quantization.latent_map_data`.
#
# Swedish Embedded AB implements solutions for image tokenizers whose codebooks are trained, checked and kept alive. If your team
# needs expertise in vector quantization and discrete latent image models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vector_quantization as vq  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts, codes = vq.latent_map_data()
    xs = [p[1] for p in pts] + [c[0] for c in codes]
    ys = [p[2] for p in pts] + [c[1] for c in codes]
    lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    p = Plot(lim, lim, width=560, height=420, xlabel="first entry of a row", ylabel="second entry of a row",
             title="Encoder rows and the codes that replace them", legend_loc="ur")
    for t in range(vq.T):
        sel = [q for q in pts if q[0] == t]
        p.points([q[1] for q in sel], [q[2] for q in sel], color=t, label=f"rows of position {t}", r=2.6)
    p.points([c[0] for c in codes], [c[1] for c in codes], color="#1a202c", label="the 16 codes", r=5.5)
    return p.svg()
