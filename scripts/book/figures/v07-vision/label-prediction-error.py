# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 8 by 8 scene of the dense-prediction chapter: its true label map, the prediction of the best stride-2 predictor (each 2 by 2
# block takes its majority class) and the error map, with the class IoU of that prediction (`dense_prediction.iou_example`).
#
# Swedish Embedded AB implements solutions for segmentation evaluation that is checked against a hand computation. If your team needs
# expertise in semantic segmentation metrics, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dense_prediction as dp  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 26
PANEL_W, PANEL_H = 250, 296
TOP = 70
SHADE = {0: "#ebf4ff", 1: "#c6f6d5", 2: "#fbd38d"}
NAMES = ("sky", "ground", "box")


def _cell(d, left, r, c, fill, text=None, bold=False):
    d.items.append(f'<rect x="{fmt(left + c * CELL)}" y="{fmt(TOP + r * CELL)}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="{MUTED}" stroke-width="0.8"/>')
    if text is not None:
        d.items.append(f'<text x="{fmt(left + (c + 0.5) * CELL)}" y="{fmt(TOP + (r + 0.5) * CELL + 4)}" {FONT} font-size="12" fill="{INK}" '
                       f'text-anchor="middle" font-weight="{"bold" if bold else "normal"}">{text}</text>')


def _panel(d, x, title, grid, errors=None):
    d.panel(x, 20, PANEL_W, PANEL_H, title)
    left = x + (PANEL_W - dp.W * CELL) / 2
    for r in range(dp.H):
        for c in range(dp.W):
            if errors is None:
                _cell(d, left, r, c, SHADE[grid[r][c]], grid[r][c])
            else:
                bad = errors[r][c]
                _cell(d, left, r, c, "#feb2b2" if bad else "#ffffff", "x" if bad else None, bold=True)
    return left


def render() -> str:
    lab, pred, conf, iou = dp.figure_data()
    d = Diagram(3 * PANEL_W + 4 * 18, PANEL_H + 76)
    err = [[lab[r][c] != pred[r][c] for c in range(dp.W)] for r in range(dp.H)]
    n_err = sum(sum(r) for r in err)
    _panel(d, 18, "truth", lab)
    _panel(d, 18 + PANEL_W + 18, "prediction (2 by 2 block vote)", pred)
    _panel(d, 18 + 2 * (PANEL_W + 18), "wrong pixels", lab, err)
    base = TOP + dp.H * CELL + 26
    d.text(18 + PANEL_W / 2, base, ["0 sky, 1 ground, 2 box"], size=12, color=MUTED)
    d.text(18 + PANEL_W + 18 + PANEL_W / 2, base, ["same classes"], size=12, color=MUTED)
    d.text(18 + 2 * (PANEL_W + 18) + PANEL_W / 2, base, [f"{n_err} of 64 pixels wrong"], size=12, color=MUTED)
    ious = "   ".join(f"{NAMES[c]} {iou[c].numerator}/{iou[c].denominator}" for c in range(dp.N_CLS))
    d.text(d.w / 2, 20 + PANEL_H + 34, [f"IoU per class: {ious}; mean IoU {dp.mean_iou(iou).numerator}/{dp.mean_iou(iou).denominator}"], size=13)
    return d.svg()
