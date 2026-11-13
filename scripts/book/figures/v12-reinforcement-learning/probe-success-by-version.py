# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The version served by the paired gate in the headline run of `project_agent`: its mean success on the probe bank at each of the five
# levels, at the step of each snapshot, with the steps at which the curriculum raised the level.
#
# Swedish Embedded AB implements solutions for agents that improve from verified feedback and for deciding, with a stated test, which
# version may be served. If your team needs expertise in reinforcement learning from verifiable rewards, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_agent as pa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    r = pa.run("spec", 0, 0.0)
    _records, registry, _served = pa.serve(r)
    steps = [0] + [s for s, *_ in r.snaps]
    served_at = []
    for s in steps:
        served_at.append([v for v in registry if v[0] <= s][-1])
    p = Plot((0, pa.STEPS), (0, 1.0), width=620, height=340, xlabel="training steps", ylabel="mean probe success of the served version",
             title="The served agent, level by level")
    for lv in range(1, pa.LEVELS + 1):
        p.line(steps, [pa._mean_scores(v[2], lv) for v in served_at], color=lv - 1, label=f"level {lv}")
    for adv in r.advances:
        p.vline(adv[0])
    return p.svg()
