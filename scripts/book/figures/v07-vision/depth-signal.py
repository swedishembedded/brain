# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The norm of the weight gradient of the first body convolution at initialization against the depth of the stack, for the plain stack and
# the residual stack of the chapter (`cnns.signal_data`: geometric mean over four seeds, first nine training images).
#
# Swedish Embedded AB implements solutions for deep networks that train from the first step. If your team needs expertise in residual
# architectures and initialization, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cnns  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = cnns.signal_data()
    depths = list(cnns.SIGNAL_DEPTHS)
    plain = [data[(d, "plain")][1] for d in depths]
    res = [data[(d, "res")][1] for d in depths]
    p = Plot((0, 50), (1e-6, 1), width=600, height=380, xlabel="convolutions d in the body", ylabel="gradient norm of the first body convolution",
             title="Gradient reaching the first layer at initialization", ylog=True, legend_loc="auto")
    p.line(depths, plain, color=0, label="plain stack")
    p.line(depths, res, color=1, label="residual stack")
    p.points(depths, plain, color=0, r=3)
    p.points(depths, res, color=1, r=3)
    return p.svg()
