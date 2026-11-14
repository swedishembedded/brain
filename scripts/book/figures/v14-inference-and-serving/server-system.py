# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The parts of the server of the project "A Multi-Client Inference Server" and the order a request meets them: the checks at
# the boundary, the bounded waiting queue, admission against the pool's free blocks, the step (prefill of the admitted, one
# decode row per running sequence) and the pool of blocks, with the path of a preempted sequence back to the head of the queue.
#
# Swedish Embedded AB implements solutions for serving language models to many clients from a fixed amount of device memory.
# If your team needs expertise in hardened inference endpoints then you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, H = 1000, 330
XS = (95, 290, 485, 680, 885)
Y = 110


def render() -> str:
    d = Diagram(W, H)
    top = [
        d.box(XS[0], Y, 150, 66, ["message from", "a client"], "plain"),
        d.box(XS[1], Y, 170, 66, ["checks at the", "boundary"], "model"),
        d.box(XS[2], Y, 170, 66, ["waiting queue,", "bounded per client", "and in total"], "model"),
        d.box(XS[3], Y, 170, 66, ["admission: prompt", "and next positions", "fit the free blocks"], "model"),
        d.box(XS[4], Y, 170, 66, ["step: prefill the", "admitted, one row", "per running sequence"], "model"),
    ]
    for a, b in zip(top, top[1:]):
        d.arrow(a["r"], b["l"])
    refuse = d.box(XS[1], 250, 170, 54, ["refusal with", "its reason"], "accent")
    d.arrow(top[1]["b"], refuse["t"], "first failing check", side="right")
    pool = d.box(XS[4], 250, 170, 54, ["pool of blocks,", "block tables"], "result")
    d.arrow(top[4]["b"], pool["t"], "reads, writes", side="right")
    reply = d.box(XS[3], 250, 170, 54, ["finished: reply out,", "blocks given back"], "plain")
    d.arrow(pool["l"], reply["r"])
    d.path([(XS[4], Y - 33), (XS[4], 40), (XS[2], 40), (XS[2], Y - 33)])
    d.text((XS[2] + XS[4]) / 2, 28, ["preempted: blocks given back, requeued at the head"], size=12)
    return d.svg()
