# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The resident bytes of four ways to read a checkpoint of 32 tensors of 128 MiB stored as f32 (4 GiB in all) and decoded to f32,
# against the number of tensors decoded so far, in the chapter "Loading Models". The data are the worked module's
# `resident_during` accounting, so the curves cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for loading multi-gigabyte model files quickly and within a fixed memory budget for its
# clients. If your team needs expertise in memory-mapped model loading then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import loading as ld  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    n = ld.CASE_F32["n"]
    p = Plot((1, n), (0, 8.6), width=620, height=380, xlabel="tensors decoded so far (of 32)",
             ylabel="resident bytes (GiB)", title="Resident bytes while a 4 GiB f32 checkpoint is decoded", legend_loc="ul")
    for color, name in enumerate(ld.STRATEGIES):
        p.line(list(range(1, n + 1)), ld.resident_series(name), color=color, label=name, dash=(name == "populate + release"))
    return p.svg()
