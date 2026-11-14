# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The block tables of three sequences and the physical pool they point into, at the end of the allocator trace of the
# chapter "Managing the KV Cache": A and its fork D share block 0, D's copied tail is block 4, and C took the two blocks B
# freed (3 on top of the stack, then 2) and then block 5. The state is the worked module's `block_trace_state`, so the
# picture cannot disagree with the table beside it.
#
# Swedish Embedded AB implements solutions for serving many concurrent conversations from the memory of one device for its
# clients. If your team needs expertise in KV cache paging then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paged_kv as pk  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 960, 360
SLOT_W, SLOT_H, POOL_W, POOL_H = 84, 46, 96, 58


def render() -> str:
    _, alloc, tables = pk.block_trace_state()
    d = Diagram(W, H)
    order = [("A", 70), ("D", 370), ("C", 640)]
    pool_y = 290
    pool_x = [72 + i * 117 for i in range(len(alloc.ref))]
    pool = {}
    for i, x in enumerate(pool_x):
        r = alloc.ref[i]
        kind = "result" if r > 1 else ("model" if r == 1 else "plain")
        label = "free" if r == 0 else ("1 reference" if r == 1 else f"{r} references")
        pool[i] = d.box(x, pool_y, POOL_W, POOL_H, [f"block {i}", label], kind)
    d.text(W / 2, pool_y + 52, ["physical pool: 8 blocks of 4 rows (free list, bottom to top: "
                                + ", ".join(str(b) for b in alloc.free) + ")"], size=12)
    for name, x0 in order:
        t = tables[name]
        n = len(t.blocks)
        d.group(x0 - 14, 26, n * (SLOT_W + 10) + 18, 104)
        d.text(x0, 44, [f"{name}: {t.len} rows"], anchor="start", weight="bold")
        for j, b in enumerate(t.blocks):
            cx = x0 + SLOT_W / 2 + j * (SLOT_W + 10)
            slot = d.box(cx, 92, SLOT_W, SLOT_H, [f"logical {j}", f"-> {b}"], "accent")
            d.arrow(slot["b"], pool[b]["t"])
    return d.svg()
