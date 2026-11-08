# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 15 embedding rows of the toy grammar projected to their first two principal directions, before and after the
# 300 steps of training of `embeddings.training`. Each panel has its own directions (power iteration on the
# covariance of its rows, `embeddings.pca_points`). Tokens of one class share a colour.
#
# Swedish Embedded AB implements solutions for inspecting what a model has learned about its vocabulary. If your team
# needs expertise in embedding analysis, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import embeddings  # noqa: E402
from svgplot import Plot, text_width  # noqa: E402

PW, PH, GAP = 400, 360, 10
COLOURS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]
OTHER = "#718096"


def _name(i):
    return "full stop" if embeddings.VOCAB[i] == "." else embeddings.VOCAB[i]


def _x_data(p, px):
    return p.xlim[0] + (px - p.ml) / (p.w - p.ml - p.mr) * (p.xlim[1] - p.xlim[0])


def _y_data(p, py):
    return p.ylim[0] + (p.h - p.mb - py) / (p.h - p.mt - p.mb) * (p.ylim[1] - p.ylim[0])


def _label(p, xs, ys, text, ids, colour, taken):
    """Put `text` above or below the points `ids`, centred or starting at their left end, where it covers no point."""
    wpx, hpx = text_width(text, 10), 12
    px = [p.x(xs[i]) for i in ids]
    py = [p.y(ys[i]) for i in ids]
    for dy in (-1, 1):
        for anchor in ("middle", "start", "end"):
            cx = sum(px) / len(px) if anchor == "middle" else (min(px) if anchor == "start" else max(px))
            x0 = cx - wpx / 2 if anchor == "middle" else cx if anchor == "start" else cx - wpx
            y = (min(py) - 9) if dy < 0 else (max(py) + 18)
            box = (x0 - 4, y - hpx, x0 + wpx + 4, y + 3)
            inside = p.ml + 2 <= box[0] and box[2] <= p.w - p.mr - 2 and p.mt + 2 <= box[1] and box[3] <= p.h - p.mb - 2
            clear = all(not (box[0] - 6 < qx < box[2] + 6 and box[1] - 6 < qy < box[3] + 6)
                        for qx, qy in zip([p.x(v) for v in xs], [p.y(v) for v in ys]))
            free = all(box[2] < t[0] or t[2] < box[0] or box[3] < t[1] or t[3] < box[1] for t in taken)
            if inside and clear and free:
                taken.append(box)
                p.text(_x_data(p, cx), _y_data(p, y), text, anchor=anchor, size=10, color=colour)
                return
    raise RuntimeError(f"no place for the label {text!r}")


def _panel(stage, show_legend):
    pts, kept = embeddings.pca_points(stage)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)
    title = f"{stage} training ({100 * kept:.0f}% of variance in 2 directions)"
    p = Plot(xlim, ylim, width=PW, height=PH, xlabel="first direction", ylabel="second direction", title=title,
             legend_loc="below" if show_legend else None)
    members = set()
    for k, (name, ids) in enumerate(embeddings.CLASSES):
        members |= set(ids)
        p.points([xs[i] for i in ids], [ys[i] for i in ids], color=COLOURS[k], r=4.5, label=name if show_legend else "")
    rest = [i for i in range(embeddings.NV) if i not in members]
    p.points([xs[i] for i in rest], [ys[i] for i in rest], color=OTHER, r=4.5, label="<s> and full stop" if show_legend else "")
    # one label per token while the rows are spread out; one per class once the rows of a class have collapsed
    # onto each other. Each label goes above or below its points, centred or starting or ending at them,
    # where it covers no point and no other label.
    if stage == "before":
        colour_of = {i: COLOURS[k] for k, (_, ids) in enumerate(embeddings.CLASSES) for i in ids}
        groups = [(_name(i), [i], colour_of.get(i, OTHER)) for i in range(embeddings.NV)]
    else:
        groups = [(", ".join(embeddings.VOCAB[i] for i in ids), ids, COLOURS[k]) for k, (_, ids) in enumerate(embeddings.CLASSES)]
        groups += [(_name(i), [i], OTHER) for i in rest]
    taken = []
    for text, ids, colour in groups:
        _label(p, xs, ys, text, ids, colour, taken)
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
