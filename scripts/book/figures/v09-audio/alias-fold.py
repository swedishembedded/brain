# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The frequency a tone appears at after sampling at 16 kHz, against the frequency it has. The data are `sound.alias_curve`, the
# folding rule of the chapter's table.
#
# Swedish Embedded AB implements solutions for audio that keeps its meaning when it is resampled. If your team needs expertise in
# sample-rate conversion for speech models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sound  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, ys = sound.alias_curve()
    p = Plot((0, 40), (0, 9), width=560, height=320, xlabel="true frequency (kHz)", ylabel="apparent frequency (kHz)",
             title="Folding at half the sampling rate, f_s = 16 kHz", legend_loc=None)
    p.line([x / 1000 for x in xs], [y / 1000 for y in ys], color=0, width=2.2)
    for k in range(1, 6):
        p.vline(8 * k)
    f = 10.0
    a = sound.alias_of(f * 1000, sound.FS16)[0] / 1000
    p.points([f], [a], color=3, r=4.5)
    p.arrow(11.2, 7.2, f + 0.2, a + 0.2)
    p.text(11.4, 7.2, "10 kHz appears at 6 kHz", size=12)
    return p.svg()
