# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The three paradigms of learning by what the learner is given and what it
# produces, drawn as three equal panels whose columns line up: what comes in,
# what is learned, what comes out. Supervised and unsupervised learning read a
# fixed dataset; reinforcement learning generates its own data by acting, so
# its panel is a loop between the agent and the environment.
#
# Swedish Embedded AB implements solutions for choosing the right learning
# setup for a data problem. If your team needs expertise in supervised,
# unsupervised and reinforcement learning, you can procure our services by
# sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, PANEL_H, GAP, TOP = 780, 128, 18, 16
COLS = (135, 390, 645)          # centres of the three columns
BOX_W, BOX_H = 180, 54


def render() -> str:
    d = Diagram(W, TOP + 3 * PANEL_H + 2 * GAP + 16)

    def row(index, title, left, middle, right, label):
        y = TOP + index * (PANEL_H + GAP)
        d.panel(20, y, W - 40, PANEL_H, title)
        cy = y + 76
        a = d.box(COLS[0], cy, BOX_W, BOX_H, left, "plain")
        b = d.box(COLS[1], cy, BOX_W, BOX_H, middle, "model")
        c = d.box(COLS[2], cy, BOX_W, BOX_H, right, "result")
        d.arrow(a["r"], b["l"], label)
        d.arrow(b["r"], c["l"], "predict" if index == 0 else "apply")

    row(0, "Supervised: examples with labels",
        ["rows (x, y)", "features and labels"], ["model f", "with parameters"], ["prediction", "for a new x"],
        "fit")
    row(1, "Unsupervised: examples without labels",
        ["rows x", "features only"], ["model of the", "structure"], ["groups, summaries,", "unusual rows"],
        "fit")

    # Reinforcement: a loop between the agent, which learns, and the
    # environment, which supplies the data.
    y = TOP + 2 * (PANEL_H + GAP)
    d.panel(20, y, W - 40, PANEL_H, "Reinforcement: acting and being rewarded")
    cy = y + 76
    # The loop is centred in its panel, with room for the longer label.
    agent = d.box(W / 2 - 200, cy, 220, BOX_H, ["agent", "a policy with parameters"], "model")
    env = d.box(W / 2 + 200, cy, BOX_W, BOX_H, ["environment"], "plain")
    upper, lower = cy - 12, cy + 12
    d.arrow((agent["r"][0], upper), (env["l"][0], upper), "action", above=True)
    d.arrow((env["l"][0], lower), (agent["r"][0], lower), "observation and reward", above=False)
    return d.svg()
