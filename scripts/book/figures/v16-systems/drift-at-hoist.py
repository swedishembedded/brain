# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The copies model of the chapter "How brain Grew": the expected number of drifted copies that a hoist must reconcile,
# against the model whose arrival triggers the hoist, with the hoist points of the chapter's table marked. The values
# are the worked module's own expectations, so the picture cannot disagree with the table.
#
# Swedish Embedded AB implements solutions for consolidating duplicated code in machine learning engines before it
# drifts for its clients. If your team needs expertise in the long-term maintenance of such engines then you can
# procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evolution as ev  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ks = list(range(1, ev.N_MODELS + 1))
    p = Plot((0, ev.N_MODELS + 1), (0, 25), width=600, height=360,
             xlabel="model whose arrival triggers the hoist",
             ylabel="expected drifted copies", title="What a late hoist has to reconcile", legend_loc="below")
    p.line(ks, [ev._drifted_at(k) for k in ks], color=0, label="drifted copies at the hoist")
    p.points(list(ev.HOISTS), [ev._drifted_at(k) for k in ev.HOISTS], color=1, label="hoist points of the table")
    p.points([ev.N_MODELS], [ev._drifted_after(ev.N_MODELS)], color=3, r=4.5,
             label=f"never hoisted, after model {ev.N_MODELS}")
    return p.svg()
