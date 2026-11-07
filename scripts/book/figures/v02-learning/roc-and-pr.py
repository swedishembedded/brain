# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The ROC curve with its area shaded, beside the precision-recall curve, for the same 3000 test parts.

Both are computed from the model's test logits by the chapter's procedure: lower the cut past one distinct score at a
time and record the counts. The ROC curve is the same whatever the share of defective parts; the precision-recall curve
is drawn twice, for the test set's own share and for a share of 1 percent made by re-weighting the good parts.

Swedish Embedded AB implements solutions for evaluating classifiers on imbalanced data. If your team needs expertise in
ROC and precision-recall analysis, you can procure our services by sending an email to info@swedishembedded.com.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metrics as m  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402


def _thin(plot, xs, ys):
    """Drop points that land on the same whole drawn pixel as their predecessor (and keep the last point)."""
    keep, last = [], None
    for a, b in zip(xs, ys):
        key = (round(plot.x(a)), round(plot.y(b)))
        if key != last:
            keep.append((a, b))
            last = key
    if keep[-1] != (xs[-1], ys[-1]):
        keep.append((xs[-1], ys[-1]))
    return [a for a, _ in keep], [b for _, b in keep]


def _inner(svg: str, x_offset: int, suffix: str) -> str:
    """One Plot's SVG as a nested <svg> placed at x_offset, with its ids made unique."""
    svg = svg.replace('id="plot"', f'id="plot{suffix}"').replace("url(#plot)", f"url(#plot{suffix})")
    svg = svg.replace('id="arrow"', f'id="arrow{suffix}"').replace("url(#arrow)", f"url(#arrow{suffix})")
    return re.sub(r"<svg ", f'<svg x="{x_offset}" y="0" ', svg, count=1)


def render() -> str:
    _, _, z, y, *_ = m._scores()
    pos = sum(y)
    neg = len(y) - pos
    pts = m._roc_points(z, y)
    auc = m._trapezoid(pts)
    w, h = 400, 380

    roc = Plot((0, 1), (0, 1), width=w, height=h, xlabel="false positive rate", ylabel="true positive rate",
               title="ROC curve")
    xs, ys = _thin(roc, [a for a, _ in pts], [b for _, b in pts])
    poly = " ".join(f"{fmt(roc.x(a))},{fmt(roc.y(b))}" for a, b in zip(xs, ys))
    base = f"{fmt(roc.x(1))},{fmt(roc.y(0))} {fmt(roc.x(0))},{fmt(roc.y(0))}"
    roc.items.append(f'<polygon points="{poly} {base}" fill="#2b6cb0" fill-opacity="0.18" stroke="none"/>')
    roc.line([0, 1], [0, 1], color="#a0aec0", dash=True, width=1.5)
    roc.line(xs, ys, color=0, width=2.4)
    roc.text(0.38, 0.30, f"area = {auc:.4f}", color="#2b6cb0", size=13)
    roc.text(0.62, 0.52, "chance: 0.5", color="#718096")

    pr = Plot((0, 1), (0, 1), width=w, height=h, xlabel="recall (true positive rate)", ylabel="precision",
              title="Precision-recall curve")
    own = m._pr_points(z, y)
    xs2, ys2 = _thin(pr, [a for a, _ in own], [b for _, b in own])
    w1 = 0.99 / 0.01 * pos / neg                    # a 1 percent share: each good part counts w1
    order = sorted(range(len(y)), key=lambda i: -z[i])
    tp = fp = 0.0
    low = []
    for i in order:
        if y[i] == 1:
            tp += 1
        else:
            fp += w1
        low.append((tp / pos, tp / (tp + fp)))
    xs3, ys3 = _thin(pr, [a for a, _ in low], [b for _, b in low])
    pr.line(xs2, ys2, color=0, width=2.4)
    pr.line(xs3, ys3, color=1, width=2.4)
    pr.hline(pos / len(y), color="#2b6cb0")
    pr.hline(0.01, color="#c05621")
    pr.text(0.30, 0.76, f"{pos / len(y):.1%} defective (the test set)", color="#2b6cb0")
    pr.text(0.38, 0.22, "1% defective", color="#c05621")
    pr.text(0.60, 0.13, "dashed: no-skill level", color="#4a5568", size=11)

    gap = 10
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * w + gap} {h}" width="{2 * w + gap}" height="{h}">\n'
            f'<rect width="{2 * w + gap}" height="{h}" fill="#ffffff"/>\n'
            + _inner(roc.svg(), 0, "a") + _inner(pr.svg(), w + gap, "b") + "</svg>\n")
