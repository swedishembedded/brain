# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The three registered contrasts as effect sizes with their 95 percent intervals, at the two lengths of training sequence: the mean difference of
# population cross-entropy (first arm minus second, nats per token) over the eight paired runs, the t interval of the mean, and the registered margin
# of 0.02 on both sides of zero. Numbers from `spectral_lm.effect_data` (the unnudged runs).
#
# Swedish Embedded AB implements solutions for testing new model architectures against matched baselines. If your team needs expertise in
# experiment design for neural architectures, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spectral_lm as sl  # noqa: E402
from svgplot import Plot  # noqa: E402

LABELS = {"S - Q_par": "S - Q_par (H1)", "S - Q_flop": "S - Q_flop (H2)", "S - R": "S - R (H3)"}
OFFSET = {"S - Q_par": 0.8, "S - Q_flop": 1.0, "S - R": 1.25}   # multiplicative offsets on the log axis, so the three do not overlap


def render() -> str:
    p = Plot((-0.09, 0.07), (500, 30000), width=620, height=330, xlabel="difference of population cross-entropy, first minus second (nats per token)",
             ylabel="training tokens", title="Effect sizes with 95% intervals; dashed: the margin 0.02", ylog=True, legend_loc="auto")
    p.vline(0.0, color="#718096", dash=False)
    p.vline(-sl.MARGIN)
    p.vline(sl.MARGIN)
    colors = {"S - Q_par": 0, "S - Q_flop": 1, "S - R": 2}
    seen = set()
    for cfg, n_tok in (("main", 1200), ("more", 12000)):
        for label, mu, lo, hi in sl.effect_data(cfg):
            y = n_tok * OFFSET[label]
            c = colors[label]
            p.line([lo, hi], [y, y], color=c, width=2.2)
            for e in (lo, hi):
                p.line([e, e], [y / 1.1, y * 1.1], color=c, width=2.2)
            p.points([mu], [y], color=c, r=4.5, label="" if label in seen else LABELS[label])
            seen.add(label)
    return p.svg()
