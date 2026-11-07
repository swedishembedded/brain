# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The end-to-end training loop as a block diagram: the stages every step runs
# (batch, forward, loss, backward, clip, AdamW with its schedule, log), the two
# stages every epoch adds (evaluate, checkpoint) and the one that runs once at
# the end (test). The snake follows the order of the chapter's pseudocode.
#
# Swedish Embedded AB implements solutions for building and verifying training
# pipelines for compact neural networks. If your team needs expertise in
# end-to-end training loops, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 770, 482
COLS = (135, 313, 491, 669)
BOX_W, BOX_H = 150, 58
ROW1, ROW2, ROW3 = 86, 206, 414


def render() -> str:
    d = Diagram(W, H)
    d.panel(18, 18, W - 36, 322, "every step (S steps make an epoch)")
    train = d.box(COLS[0], ROW1, BOX_W, BOX_H, ["training set", "X, T"], "result")
    batch = d.box(COLS[1], ROW1, BOX_W, BOX_H, ["shuffle, batch", "B rows per step"], "model")
    fwd = d.box(COLS[2], ROW1, BOX_W, BOX_H, ["forward", "Y = f(theta, x)"], "model")
    loss = d.box(COLS[3], ROW1, BOX_W, BOX_H, ["loss", "mean squared error"], "accent")
    bwd = d.box(COLS[3], ROW2, BOX_W, BOX_H, ["backward", "gradient g"], "model")
    clip = d.box(COLS[2], ROW2, BOX_W, BOX_H, ["clip", "global norm to tau"], "model")
    adam = d.box(COLS[1], ROW2, BOX_W, BOX_H, ["AdamW", "update theta, m, v"], "model")
    log = d.box(COLS[0], ROW2, BOX_W, BOX_H, ["log", "loss, norm, rate"], "plain")
    sched = d.oval(COLS[1], 290, 120, 44, ["schedule", "rate eta(t)"], "plain")
    d.arrow(train["r"], batch["l"])
    d.arrow(batch["r"], fwd["l"])
    d.arrow(fwd["r"], loss["l"])
    d.arrow(loss["b"], bwd["t"])
    d.arrow(bwd["l"], clip["r"])
    d.arrow(clip["l"], adam["r"])
    d.arrow(adam["l"], log["r"])
    d.arrow(sched["t"], adam["b"])
    # the loop closes: after the log, the next step takes the next batch
    d.path([(log["l"][0], ROW2), (30, ROW2), (30, ROW1), (train["l"][0], ROW1)])
    d.panel(18, 352, W - 36, 112, "every epoch")
    ev = d.box(COLS[0], ROW3, BOX_W, 50, ["evaluate", "validation loss"], "model")
    ck = d.box(COLS[1], ROW3, BOX_W, 50, ["checkpoint", "keep theta if lower"], "result")
    test = d.box(COLS[2], ROW3, BOX_W, 50, ["test, once", "after the last epoch"], "accent")
    d.arrow(log["b"], ev["t"])
    d.arrow(ev["r"], ck["l"])
    d.arrow(ck["r"], test["l"])
    return d.svg()
