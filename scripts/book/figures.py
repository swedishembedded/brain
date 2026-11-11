#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Render the book's figures from their sources.

Every figure in docs/book is generated, never drawn by hand, so it can be
reviewed as text and regenerated when the idea it shows changes:

    scripts/book/figures/<volume dir>/<name>.dot  -> Graphviz (structure)
    scripts/book/figures/<volume dir>/<name>.py   -> render() -> SVG (numbers,
                                                     or computed structure
                                                     drawn via dot_svg())

both written to docs/book/<volume dir>/img/<name>.svg. A `.py` figure computes
its data with the arithmetic the chapter teaches, so the picture and the text
cannot disagree.

Each SVG records the SHA-256 of its source. `--check` verifies that every
source has its SVG, that the recorded hash matches (the SVG is not stale),
that no SVG lacks a source, and that no figure overflows: no label wider
than its box, no box across the edge of its panel, nothing off the canvas
(`svgcheck`, which measures text with Helvetica's advance widths). Rendering
fails on the same check. The check does not re-render: Graphviz
layouts depend on the installed fonts, so byte-for-byte comparison across
machines would fail for reasons unrelated to the figure.

    scripts/book/figures.py            # render every figure
    scripts/book/figures.py --check    # verify freshness and pairing
    scripts/book/figures.py --png DIR  # also rasterize every figure to DIR
    scripts/book/figures.py --all      # re-render even figures whose source is unchanged

Only stale figures are re-rendered unless --all is given.

Swedish Embedded AB implements solutions for reproducible engineering
documentation. If your team needs expertise in machine learning education
material, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deps  # noqa: E402
import svgcheck  # noqa: E402
ROOT = HERE.parents[1]
SRC = HERE / "figures"
BOOK = ROOT / "docs" / "book"
STAMP = re.compile(r"<!-- source-sha256: ([0-9a-f]{64}) -->")
DOT_DEFAULTS = ["-Gfontname=Helvetica", "-Nfontname=Helvetica", "-Efontname=Helvetica",
                "-Gfontsize=11", "-Nfontsize=11", "-Efontsize=11", "-Gbgcolor=white", "-Gpad=0.15"]


def sources():
    return sorted(p for p in SRC.rglob("*") if p.suffix in (".dot", ".py") and p.is_file())


def target(src: Path) -> Path:
    rel = src.relative_to(SRC)
    return BOOK / rel.parent / "img" / (src.stem + ".svg")


def digest(src: Path) -> str:
    # A Python figure depends on every book module it imports: the plotting
    # and diagram helpers and the worked modules whose data it draws.
    if src.suffix == ".py":
        return deps.digest(src)
    return hashlib.sha256(src.read_bytes()).hexdigest()


def dot_svg(source: str, where: str = "<dot>") -> str:
    """Graphviz source to a compact SVG. Python figures whose structure is
    computed (a tree built by the chapter's algorithm) call this too."""
    r = subprocess.run(["dot", "-Tsvg", *DOT_DEFAULTS], input=source, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{where}: dot failed: {r.stderr.strip()}")
    svg = r.stdout
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.S)
    svg = re.sub(r"<!--.*?-->\s*", "", svg, flags=re.S)
    svg = re.sub(r"<title>[^<]*</title>\s*", "", svg)
    return "\n".join(line.strip() for line in svg.splitlines() if line.strip()) + "\n"


def render_dot(src: Path) -> str:
    return dot_svg(src.read_text(encoding="utf-8"), str(src))


def render_py(src: Path) -> str:
    sys.path.insert(0, str(HERE))
    import svgplot
    svgplot.Plot.context = src.stem
    spec = importlib.util.spec_from_file_location(f"fig_{src.stem}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render()


def render(src: Path) -> str:
    svg = render_dot(src) if src.suffix == ".dot" else render_py(src)
    return f"<!-- source-sha256: {digest(src)} -->\n{svg}"


def check() -> int:
    errors = []
    expected = set()
    for src in sources():
        out = target(src)
        expected.add(out)
        if not out.is_file():
            errors.append(f"{out.relative_to(ROOT)}: missing (render {src.relative_to(ROOT)})")
            continue
        text = out.read_text(encoding="utf-8")
        m = STAMP.search(text)
        if not m or m.group(1) != digest(src):
            errors.append(f"{out.relative_to(ROOT)}: stale, its source changed")
        errors += [f"{out.relative_to(ROOT)}: {p}" for p in svgcheck.problems(text)]
    for svg in BOOK.rglob("img/*.svg"):
        if svg not in expected:
            errors.append(f"{svg.relative_to(ROOT)}: no source under scripts/book/figures")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("figures: run `make generate/book/figures`", file=sys.stderr)
        return 1
    print(f"figures: {len(expected)} up to date")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    png_dir = None
    if "--png" in sys.argv:
        png_dir = Path(sys.argv[sys.argv.index("--png") + 1])
        png_dir.mkdir(parents=True, exist_ok=True)
    everything = "--all" in sys.argv
    overflow = []
    for src in sources():
        out = target(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        stamp = STAMP.search(out.read_text(encoding="utf-8")) if out.is_file() else None
        # A figure whose recorded source digest still matches is not
        # re-rendered: some recompute whole training runs.
        if everything or not stamp or stamp.group(1) != digest(src):
            svg = render(src)
            out.write_text(svg, encoding="utf-8")
            print(f"{out.relative_to(ROOT)} ({out.stat().st_size // 1024} KiB)")
            overflow += [f"{out.relative_to(ROOT)}: {p}" for p in svgcheck.problems(svg)]
        if png_dir:
            subprocess.run(["rsvg-convert", "-z", "1.5", "-o", str(png_dir / (src.stem + ".png")), str(out)],
                           check=True)
    if overflow:
        # Written anyway, so the PNG can be looked at; but the figure is wrong.
        print("\n".join(overflow), file=sys.stderr)
        print("figures: the figures above overflow (see scripts/book/svgcheck.py)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
