# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The gates one HTTP request passes, in the order the chapter "Security of Machine Learning Services" traces them through
# brain's accept loop, router and dispatch: what is checked at each, what refuses there, and with which ceiling. The numbers
# are the worked module's constants, so the picture and the tables cannot disagree.
#
# Swedish Embedded AB implements solutions for inference services that stay up and keep their clients apart under hostile
# requests for its clients. If your team needs expertise in hardening machine learning servers then you can procure our
# services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import security as sec  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402

W, H = 1000, 530
BW, BH = 200, 70
COLS = [130, 375, 620, 865]
ROWS = [100, 260, 420]


def render() -> str:
    d = Diagram(W, H)
    labels = ["refused here without a key", "keyed: the request is someone's", "dispatch: the call is about to cost a device"]
    for y, label in zip(ROWS, labels):
        d.text(COLS[0] + 14, y - BH / 2 - 14, [label], size=12, color=MUTED, anchor="start")
    stages = [
        [(["accept", f"at most {sec.MAX_CONNECTIONS:,} held,", "the rest closed"], "plain"),
         (["TLS handshake", f"within {sec.HEADER_DEADLINE_S} seconds", "(when configured)"], "plain"),
         (["request head", f"within {sec.HEADER_DEADLINE_S} seconds"], "plain"),
         (["key check", "every route, unknown", "ones too: 401"], "accent")],
        [(["edge permit", f"{sec.EDGE_CONCURRENCY} in flight,", "else 503 at once"], "accent"),
         (["body", f"8 MiB (64 MiB uploads): 413,", f"whole within {sec.BODY_DEADLINE_S} seconds"], "accent"),
         (["parse and bounds", "ranges, counts, no host", "path: 400"], "accent"),
         (["media decode", f"{sec.MAX_CONCURRENT_DECODES} at once, pixels from", "the header: 400 or 503"], "accent")],
        [(["dispatch gate", "training 403, caller's", "budget 429"], "result"),
         (["admission", "a lane within the", "deadline, else 429"], "result"),
         (["executor lane", "the model runs"], "model")],
    ]
    boxes = []
    for r, row in enumerate(stages):
        line = []
        for c, (lines, kind) in enumerate(row):
            line.append(d.box(COLS[c], ROWS[r], BW, BH, lines, kind))
        boxes.append(line)
    for line in boxes:
        for a, b in zip(line, line[1:]):
            d.arrow(a["r"], b["l"])
    # Row ends wrap to the start of the next row.
    for r in range(len(boxes) - 1):
        end, start = boxes[r][-1], boxes[r + 1][0]
        mid = (ROWS[r] + ROWS[r + 1]) / 2
        d.path([end["b"], (end["b"][0], mid), (start["t"][0], mid), start["t"]], color=INK)
    d.text(W / 2, 505, ["cheap checks first: a peer is refused before it costs a body, a decode or a device"], size=12,
           color=MUTED)
    return d.svg()
