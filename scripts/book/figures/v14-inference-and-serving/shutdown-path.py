# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path from a signal to the end of a serving process in the chapter "Running and Observing an Engine": one signal thread
# owns SIGINT and SIGTERM and fires a shared latch; every surface stops accepting, the jobs behind open requests are cancelled,
# the drain is bounded twice, and the process ends without running its library finalizers. A second signal ends it at once.
#
# Swedish Embedded AB implements solutions for observing, controlling and cleanly stopping long-running inference servers for
# its clients. If your team needs expertise in graceful shutdown of model servers then you can procure our services by sending
# an email to info@swedishembedded.com.
from svgdiagram import MUTED, RED, Diagram

W, H = 1000, 380


def render() -> str:
    d = Diagram(W, H)
    sig = d.box(85, 140, 130, 66, ["SIGINT or", "SIGTERM"], "plain")
    thr = d.box(255, 140, 160, 66, ["signal thread:", "fire the latch,", "then log"], "accent")
    latch = d.box(445, 140, 160, 66, ["shutdown latch", "(every clone", "sees it once)"], "result")
    http = d.box(660, 60, 200, 66, ["HTTP surface: stop", "accepting, cancel", "every job, drain"], "model")
    dbus = d.box(660, 220, 200, 66, ["D-Bus surface: stop", "the stats stream,", "then close the bus"], "model")
    done = d.box(890, 140, 170, 66, ["exit without", "running library", "finalizers"], "result")
    second = d.box(255, 290, 160, 66, ["second signal:", "exit at once", "(status 130)"], "accent")
    d.arrow(sig["r"], thr["l"])
    d.arrow(thr["r"], latch["l"])
    d.path([(525, 125), (545, 125), (545, 60), http["l"]])
    d.path([(525, 155), (545, 155), (545, 220), dbus["l"]])
    d.path([http["r"], (785, 60), (785, 125), (805, 125)])
    d.path([dbus["r"], (785, 220), (785, 155), (805, 155)])
    d.arrow(thr["b"], second["t"], color=RED)
    d.text(660, 330, ["bounds in the code: 3 seconds for open HTTP requests,", "5 seconds to join the D-Bus thread, 30 seconds for",
                      "the whole drain, after which the process ends anyway"], size=12, color=MUTED)
    return d.svg()
