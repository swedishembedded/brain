# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Test accuracy against the distance of the subject for the three recurrent cells (mean of three runs each) and the
# ceiling of any model that reads only the last four tokens. Numbers from `recurrent_networks.accuracy_by_distance`;
# distances to the right of the dotted line were not in the training set.
#
# Swedish Embedded AB implements solutions for sequence models that must remember across long inputs. If your team
# needs expertise in evaluating the memory of a language model, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import recurrent_networks as rn  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    acc, ceil = rn.accuracy_data()
    ds = [rn.distance(k) for k in rn.TEST_KS]
    p = Plot((0, 100), (0.4, 1.04), width=640, height=380, xlabel="distance d of the subject from the verb (tokens)",
             ylabel="test accuracy", title="Memory beyond a window of four tokens")
    # the ceiling is a step (1 while the subject is inside the window, 1/2 from the first distance beyond it), so draw the
    # two levels as separate segments instead of a slope between d = 4 and d = 7 that no sequence has
    inside = [d for d in ds if d <= rn.WINDOW]
    beyond = [d for d in ds if d > rn.WINDOW]
    p.line([inside[0], inside[-1]], [1.0, 1.0], color="#718096", dash=True, label="window of 4 tokens (ceiling)")
    p.line([beyond[0], beyond[-1]], [0.5, 0.5], color="#718096", dash=True)
    p.points(ds, ceil, color="#718096", r=3)
    for kind, color in (("rnn", 3), ("gru", 5), ("lstm", 2)):
        label = rn.CELL_NAMES[kind]
        p.line(ds, acc[kind], color=color, label=label)
        p.points(ds, acc[kind], color=color, r=3)
    p.vline(rn.distance(max(rn.TRAIN_KS)) + 12)
    p.text(rn.distance(max(rn.TRAIN_KS)) + 13.5, 0.43, "not trained on", size=12, color="#4a5568")
    return p.svg()
