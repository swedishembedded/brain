#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Chart per-LEVEL progress across generations, from scoreboard.sh's CSV.

One line per level rather than one average, because the average hides the
thing worth knowing: which levels moved. A level stuck at 0.10 with no kills
has a navigation problem and a level at 0.02 with kills has a combat one, and
a mean over the nine says neither.

The line at 1.0 is the bar that matters - above it the level was finished -
and 2.0 is a level taken apart: every monster, every item, every secret.

usage: plot-progress.py out/gen/scoreboard.csv docs/
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, GOOD = "#22222a", "#d8d8e0", "#3aa86a"


def main(csv_path, out_dir):
    rows = list(csv.DictReader(Path(csv_path).open()))
    if not rows:
        sys.exit("no rows in " + csv_path)
    by_map = defaultdict(list)
    skipped = 0
    for r in rows:
        # A row whose numbers are missing is a run that did not report, not a
        # run that scored zero, and drawing it as zero would put a dip in the
        # curve where there is only a gap. Say how many rather than failing:
        # a plot is for looking at, and refusing to draw anything because one
        # level of one generation is missing helps nobody.
        try:
            point = (int(r["generation"]), float(r["progress"]),
                     int(float(r["exits"])))
        except (ValueError, TypeError):
            skipped += 1
            continue
        by_map[int(r["map"])].append(point)
    if skipped:
        print(f"  skipped {skipped} row(s) that reported no result")
    if not by_map:
        sys.exit("no usable rows in " + csv_path)
    for m in by_map:
        by_map[m].sort()

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[3, 1])
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(sorted(by_map)):
        gens = [g for g, _, _ in by_map[m]]
        prog = [p for _, p, _ in by_map[m]]
        ax.plot(gens, prog, marker="o", ms=4, lw=1.6,
                color=cmap(i / max(len(by_map) - 1, 1)), label=f"E1M{m}")
        # A filled marker wherever the level was actually finished.
        for g, p, e in by_map[m]:
            if e > 0:
                ax.plot([g], [p], marker="*", ms=14, color=GOOD, zorder=5)

    ax.axhline(1.0, color=GOOD, ls="--", lw=1.2)
    ax.text(0.01, 1.02, "finished", color=GOOD, transform=ax.get_yaxis_transform())
    ax.axhline(2.0, color=GOOD, ls=":", lw=1.0)
    ax.text(0.01, 2.02, "cleared: every kill, item and secret",
            color=GOOD, transform=ax.get_yaxis_transform())
    ax.set_ylabel("progress")
    ax.set_title("Per-level progress by generation (a star is an exit)", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(ncol=5, fontsize=8, frameon=False)

    # How many of the nine were finished, which is the number the run is for.
    gens = sorted({g for m in by_map for g, _, _ in by_map[m]})
    done = [sum(1 for m in by_map for g, _, e in by_map[m] if g == gg and e > 0)
            for gg in gens]
    bx.bar(gens, done, color=GOOD)
    bx.set_ylim(0, 9)
    bx.set_xlabel("generation")
    bx.set_ylabel("levels finished")
    bx.grid(color=GRID, lw=0.6, axis="y")

    out = Path(out_dir) / "progress.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    # And the same thing as text, for a terminal.
    for m in sorted(by_map):
        track = "  ".join(f"g{g}:{p:.2f}{'*' if e else ''}" for g, p, e in by_map[m])
        print(f"  E1M{m}  {track}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
