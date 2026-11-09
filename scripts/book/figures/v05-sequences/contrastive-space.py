# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Thirty-two sentence vectors of the toy synonym language (four meanings, eight sentences each: every combination of synonyms) projected on their
# first two principal directions, before and after the 100 steps of contrastive training of `encoders.contrastive_training`. Each panel has
# its own directions (`encoders.pca_points`). A meaning is one colour; its eight sentences share no content word with some of the others.
#
# Swedish Embedded AB implements solutions for inspecting what an embedding model has learned. If your team needs expertise in embedding
# analysis and retrieval evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import encoders  # noqa: E402
from svgplot import Plot  # noqa: E402

PW, PH, GAP = 400, 360, 10
COLOURS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]


def _name(m):
    a, b = encoders.MEANINGS[m]
    return f"{encoders.ANIMALS_B[a][0]} in the {encoders.PLACES_B[b][0]}"


def _panel(stage, show_legend):
    groups, kept = encoders.pca_points(stage)
    xs = [x for g in groups for x, _ in g]
    ys = [y for g in groups for _, y in g]
    pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys))
    p = Plot((min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad), width=PW, height=PH, xlabel="first direction",
             ylabel="second direction", title=f"{stage} training ({100 * kept:.0f}% of variance in 2 directions)",
             legend_loc="below" if show_legend else None)
    for k, (m, g) in enumerate(zip(encoders.PCA_MEANINGS, groups)):
        p.points([x for x, _ in g], [y for _, y in g], color=COLOURS[k], r=4.5, label=_name(m) if show_legend else "")
    return p.svg()


def render() -> str:
    panels = [_panel("before", True), _panel("after", False)]
    inner = lambda s: s.split("\n", 1)[1].rsplit("</svg>", 1)[0]  # noqa: E731
    sizes = []
    for s in panels:
        head = s.split("\n", 1)[0]
        vb = head.split('viewBox="')[1].split('"')[0].split()
        sizes.append((float(vb[2]), float(vb[3])))
    w = sizes[0][0] + sizes[1][0] + GAP
    h = max(s[1] for s in sizes)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}">',
           f'<rect width="{w:g}" height="{h:g}" fill="#ffffff"/>']
    x = 0
    for body, (sw, sh) in zip(panels, sizes):
        out.append(f'<svg x="{x:g}" y="0" width="{sw:g}" height="{sh:g}" viewBox="0 0 {sw:g} {sh:g}">\n{inner(body)}</svg>')
        x += sw + GAP
    out.append("</svg>")
    return "\n".join(out) + "\n"
