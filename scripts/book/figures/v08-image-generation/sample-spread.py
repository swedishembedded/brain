# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The spread (standard deviation) of 1000 generated samples while a generator and a discriminator train on three clusters of data,
# under two schedules: one generator step per discriminator step, and twenty. A collapsed generator has a spread near zero; the data's
# spread is the dashed line. The data are `gans.spread_data` (the first run of the chapter's table).
#
# Swedish Embedded AB implements solutions for generative models whose adversarial training is specified, checked and made
# reproducible. If your team needs expertise in generative adversarial networks and image synthesis, you can procure our services
# by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gans  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    steps, spread = gans.spread_data()
    top = max(max(spread["one"]), max(spread["twenty"]), gans.DATA_STD) * 1.3
    p = Plot((0, steps[-1]), (0, top), width=560, height=340, xlabel="discriminator steps", ylabel="spread of generated samples",
             title="Spread of the samples under two schedules", legend_loc="ul")
    p.hline(gans.DATA_STD)
    p.line(steps, spread["twenty"], color=1, label="twenty generator steps per discriminator step", width=2.0)
    p.line(steps, spread["one"], color=0, label="one generator step per discriminator step", width=2.2)
    p.text(steps[0] - 8, gans.DATA_STD + 0.12, "spread of the data", anchor="start", size=12)
    return p.svg()
