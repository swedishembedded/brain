# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which keys each query reads: the causal mask of Volume V, and the same positions with a sliding window of 4 keys plus 2 sink keys
# that are always kept. Row i is the query at position i, column j the key at position j. Blue cells are keys inside the window,
# yellow cells are sinks beyond it, white cells are keys the query no longer reads, and empty cells are the future. The mask is
# `attention_variants.mask_data`, the function behind the worked example of the chapter.
#
# Swedish Embedded AB implements solutions for serving language models within a memory budget. If your team needs expertise in
# long-context attention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import attention_variants as av  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

T, N_WIN, N_SINK = 12, 4, 2
CELL = 24
W, H = 760, 430
GRID_TOP = 96
LEFT_X, RIGHT_X = 60, 430
KIND = {"w": "model", "s": "result", "x": "plain"}


def grid(d, x0, title, mask, caption):
    d.text(x0 + T * CELL / 2, 30, [title], size=14, weight="bold")
    d.text(x0 + T * CELL / 2, 52, ["key position j"], size=12, color=MUTED)
    d.text(x0 - 26, GRID_TOP + T * CELL / 2, ["query"], size=12, color=MUTED, anchor="end")
    for i in range(T):
        d.text(x0 - 8, GRID_TOP + i * CELL + CELL / 2, [str(i)], size=11, color=MUTED, anchor="end")
        d.text(x0 + i * CELL + CELL / 2, GRID_TOP - 10, [str(i)], size=11, color=MUTED)
        for j in range(T):
            if mask[i][j] != ".":
                d.swatch(x0 + j * CELL + 1, GRID_TOP + i * CELL + 1, CELL - 2, CELL - 2, KIND[mask[i][j]])
    d.text(x0 + T * CELL / 2, GRID_TOP + T * CELL + 28, caption, size=12, color=MUTED)


def render() -> str:
    full = av.window_mask(T, T)
    keep = av.mask_data(T, N_WIN, N_SINK)
    d = Diagram(W, H)
    grid(d, LEFT_X, "Causal: every earlier key", full, [f"the last query reads {T} keys; the cache grows with the text"])
    grid(d, RIGHT_X, f"Window of {N_WIN} and {N_SINK} sinks", keep,
         [f"the last query reads {N_WIN + N_SINK} keys, and so does every later one"])
    return d.svg()
