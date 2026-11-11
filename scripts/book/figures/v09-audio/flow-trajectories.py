# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Paths of the exactly solvable flow-matching decoder of the chapter: three starting values carried from noise (t = 0) to a mel value
# (t = 1), once under the conditional field and once under the guided field. The paths are `speech_synthesis.toy_paths`, the
# arithmetic of the chapter's tables on a fine uniform grid.
#
# Swedish Embedded AB implements solutions for generating speech spectrograms with iterative decoders. If your team needs expertise
# in flow-matching text-to-speech, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speech_synthesis as ss  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ts, paths = ss.toy_paths()
    p = Plot((0, 1), (-0.7, 2.1), width=560, height=340, xlabel="flow time t (0 noise, 1 mel value)",
             ylabel="mel value x", title="Three starts under the conditional and the guided field", legend_loc="lr")
    p.hline(ss._TOY_MU)
    p.hline(0.0)
    p.text(0.02, ss._TOY_MU + 0.06, "conditional mean 1.5", size=11)
    p.text(0.98, 0.06, "unconditional mean 0", anchor="end", size=11)
    first = True
    for k, (z, (c, g)) in enumerate(paths.items()):
        p.line(ts, c, color=k, label="conditional field" if first and k == 0 else "", width=2.0)
        p.line(ts, g, color=k, label="guided field (dashed)" if first and k == 0 else "", dash=True, width=2.0)
        first = False
    return p.svg()
