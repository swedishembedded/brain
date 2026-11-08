# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The cost, in millions of bytes of memory traffic, of a chain of k elementwise kernels over 2^16 floats on the model GPU:
# every kernel submitted alone, all k recorded into one submission, and the chain fused into one kernel. The costs are the
# worked module's `launch_costs`: a start cost per submission and per dispatch (labelled parameters) plus the bytes of the
# kernels. Recording removes starts but not bytes; fusion removes both.
#
# Swedish Embedded AB implements solutions for removing the memory round trips between the small kernels of a machine learning
# workload. If your team needs expertise in GPU launch overhead and command recording, you can procure our services by sending
# an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fusion as fu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ks, alone, recorded, fused = fu.launch_curve(1 << 16, 64)
    top = alone[-1] / 1e6
    p = Plot((0, 64), (0, top * 1.05), width=600, height=360, xlabel="kernels in the chain, k",
             ylabel="cost (millions of bytes)", title="Cost of a chain of small kernels, N = 2^16")
    p.line(ks, [v / 1e6 for v in alone], color=3, label="each submitted alone")
    p.line(ks, [v / 1e6 for v in recorded], color=0, label="recorded, one submission")
    p.line(ks, [v / 1e6 for v in fused], color=2, label="fused, one dispatch")
    return p.svg()
