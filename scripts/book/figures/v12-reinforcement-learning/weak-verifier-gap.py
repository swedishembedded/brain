# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mistake 1 of `project_agent`: the reward the trainer is paid by the weak verifier (every digit only has to have the right parity) rises
# to nearly 1 while the exact success of the same policy on the probe bank at level 2 stops near 0.47; the specified build, paid by the
# exact verifier, reaches 0.91 on the same probes.
#
# Swedish Embedded AB implements solutions for reinforcement learning against rewards that cannot be argued with. If your team needs
# expertise in verifiable rewards and in finding the reward hacks of a checker, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_agent as pa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    weak = pa.run("weak", 0, 0.0)
    spec = pa.run("spec", 0, 0.0)
    steps = [s for s, *_ in weak.snaps]
    paid = [sum(weak.train[s - 10:s]) / 10 for s in steps]
    p = Plot((0, pa.STEPS), (0, 1.05), width=620, height=340, xlabel="training steps", ylabel="mean over 10 steps, or on the 48 probes",
             title="Paid by the weak verifier, judged by the exact one")
    p.line(steps, paid, color=0, label="reward paid by the weak verifier (weak build)")
    p.line(steps, [pa._mean_scores(sc, 2) for _s, _lv, _th, sc in weak.snaps], color=1, label="exact probe success at level 2 (weak build)")
    p.line(steps, [pa._mean_scores(sc, 2) for _s, _lv, _th, sc in spec.snaps], color=2, label="exact probe success at level 2 (specified build)")
    return p.svg()
