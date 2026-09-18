#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Turn a training run's own stdout into the two charts the README shows.

A private helper for this sample, not a standalone tool: it parses exactly the
lines `sample-decision-doom train` prints and nothing else.

    ./plot-run.py out/train-run1.log docs/

Deliberately reads the LOG rather than a metrics file the sample writes. The
log is what a person watching the run already has, so a chart can be made after
the fact from any run anybody kept - including one that crashed.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, GOOD, WARN = "#22222a", "#d8d8e0", "#3aa86a", "#d05a50"


def parse(path):
    """(warm-start losses, per-iteration rollout stats) from a run's stdout."""
    losses, iters = [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"step\s+(\d+)\s+loss\s+([-\d.]+)", line)
        if m:
            losses.append((int(m.group(1)), float(m.group(2))))
        m = re.search(
            r"iter\s+(\d+)\s+return\s+([-+\d.]+)\s+wins\s+(\d+)/(\d+)\s+steps\s+(\d+)"
            r"\s+critic mse\s+([\d.]+)",
            line,
        )
        if m:
            iters.append(
                dict(
                    iter=int(m.group(1)),
                    ret=float(m.group(2)),
                    wins=int(m.group(3)),
                    steps=int(m.group(5)),
                    mse=float(m.group(6)),
                )
            )
    return losses, iters


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    log, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    losses, iters = parse(log)

    if losses:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
        ax.plot([s for s, _ in losses], [v for _, v in losses], color=GOOD, linewidth=1.6)
        style(ax, "Warm start: cloning the scripted player", "optimizer step", "cross-entropy")
        fig.tight_layout()
        fig.savefig(outdir / "warmstart-loss.png")
        plt.close(fig)

    if iters:
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), dpi=140)
        a.plot([d["iter"] for d in iters], [d["ret"] for d in iters], color=GOOD, linewidth=1.8,
               marker="o", markersize=3)
        a.axhline(0, color=GRID, linewidth=1)
        style(a, "Mean return per rollout", "PPO iteration", "return")
        b.plot([d["iter"] for d in iters], [d["mse"] for d in iters], color=WARN, linewidth=1.8,
               marker="o", markersize=3)
        # The critic's error is worth showing next to the return: PPO's
        # advantages are only as good as the baseline subtracted from them, so
        # a return that climbs while this stays high is a return to distrust.
        style(b, "Critic mean squared error", "PPO iteration", "mse")
        fig.tight_layout()
        fig.savefig(outdir / "training.png")
        plt.close(fig)

    print(f"{len(losses)} warm-start points, {len(iters)} iterations -> {outdir}")


if __name__ == "__main__":
    main()
