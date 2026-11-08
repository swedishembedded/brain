# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The five stages of kernel selection, for one request on the GPU unit and the same request on the CPU unit: the
# candidates, the capability filter (a correctness gate), the reference gate (a candidate must match the reference
# before it may win), the measurement (a race with a cap) and the cache. The counts under the boxes are the first
# request of the worked module's `pipeline_trace`, so the picture and the table cannot disagree.
#
# Swedish Embedded AB implements solutions for kernels that choose themselves on hardware nobody tuned them for. If
# your team needs expertise in kernel selection for machine learning inference, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kernel_selection as ks  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 860, 358
COLS = [88 + 171 * i for i in range(5)]          # centres of the five boxes
BOX_W, BOX_H = 112, 54
PANEL_Y = (14, 188)
PANEL_H = 156
TITLES = ("Request m = 200 on the GPU unit", "The same request on the CPU unit")
STAGES = (["Candidates"], ["Capability", "filter"], ["Reference", "gate"], ["Measure", "(a race)"], ["Selection", "cache"])


def _notes(e, cpu):
    if cpu:
        return (["reg, tiled, naive,", "reg-notail"],
                [f"{e['n_cap']} of {e['n_cat']} pass", "3 refused: no", "workgroup barriers"],
                ["one candidate left:", "nothing to check"],
                ["one candidate left:", "nothing to time"],
                [f"stores {e['choice']}", "under the CPU key"])
    refused = e["n_cap"] - e["n_gate"]
    return (["reg, tiled, naive,", "reg-notail"],
            [f"{e['n_cap']} of {e['n_cat']} pass"],
            [f"{e['gate_launches']} gate launches", f"{refused} refused: reg-notail", "wrong at m = 129"],
            [f"{e['launches']} launches", "dropped: " + ", ".join(e["dropped"])],
            [f"stores {e['choice']}", "key: device | driver |", "digest | bucket 256"])


def render() -> str:
    gpu, cpu = ks.pipeline_numbers()
    d = Diagram(W, H)
    for i, (e, is_cpu) in enumerate(((gpu, False), (cpu, True))):
        y0 = PANEL_Y[i]
        d.panel(20, y0, W - 40, PANEL_H, TITLES[i])
        cy = y0 + 62
        boxes = [d.box(x, cy, BOX_W, BOX_H, STAGES[j], "accent" if j in (1, 2) else "model") for j, x in enumerate(COLS)]
        counts = [str(e["n_cat"]), str(e["n_cap"]), str(e["n_gate"]), e["choice"]]
        for j in range(4):
            d.arrow(boxes[j]["r"], boxes[j + 1]["l"], counts[j])
        for j, lines in enumerate(_notes(e, is_cpu)):
            d.text(COLS[j], cy + BOX_H / 2 + 34, lines, size=11, color="#4a5568")
    return d.svg()
