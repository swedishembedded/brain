# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loop of a Markov decision process: the agent's policy turns the state of the morning into an order, the environment draws the
# demand, pays the reward of the day and hands back the state of the next morning.
#
# Swedish Embedded AB implements solutions for sequential decisions under uncertainty. If your team needs expertise in Markov decision
# processes and the policies that solve them, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    agent = ["agent", "policy pi:", "state s_t -> action a_t"]
    env = ["environment", "draws the demand D;", "next state from P(. | s_t, a_t);", "reward r_t from the day"]
    pad, h, gap = 24, 90, 150
    wa, we = int(text_width(agent) + 16), int(text_width(env) + 16)
    xa = pad + wa / 2
    xe = pad + wa + gap + we / 2
    width = int(pad + wa + gap + we + pad)
    y = pad + 62 + h / 2
    d = Diagram(width, int(y + h / 2 + 62 + pad))
    a = d.box(xa, y, wa, h, agent, "accent")
    e = d.box(xe, y, we, h, env, "model")
    top, low = y - h / 2 - 34, y + h / 2 + 34
    d.path([a["t"], (xa, top), (xe, top), e["t"]])
    d.text((xa + xe) / 2, top - 14, ["action a_t: how many units to order"], size=12, color="#4a5568")
    d.path([e["b"], (xe, low), (xa, low), a["b"]])
    d.text((xa + xe) / 2, low + 16, ["state s_(t+1) and reward r_t"], size=12, color="#4a5568")
    return d.svg()
