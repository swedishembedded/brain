# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Loads of a weight group into a window of slots, for a model of 32 groups run 4 times over its groups in a fixed order, against the
# number of slots in the window, for the policy that plans over the known schedule and for least recently used, in the chapter
# "Memory Budgets and Model Residency". The data are the worked module's `window_series`, so the curves cannot disagree with its table.
#
# Swedish Embedded AB implements memory-budgeted, multi-model serving for teams whose models do not all fit resident at once.
# If your team needs expertise in streaming model weights through a bounded window then you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import residency as rs  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    slots = list(range(2, 41))
    scan, lru = rs.window_series(slots)
    p = Plot((2, 40), (0, 140), width=620, height=380, xlabel="slots in the window (linear axis)",
             ylabel="loads over 4 passes of 32 groups (linear axis)", title="Loads against the width of the window",
             legend_loc="ll")
    p.line(slots, scan, color=0, label="plan over the known schedule")
    p.line(slots, lru, color=1, label="least recently used", dash=True)
    return p.svg()
