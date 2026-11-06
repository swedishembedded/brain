#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Typeset the book as one PDF, separate from the engineering manual.

Reading order comes from docs/book/book.yaml (the same specification the
published navigation is generated from): front matter, then each volume with
written chapters as a LaTeX part (its introduction first), then the appendices.
Page transforms and the pandoc invocation are the documentation build's own
(docs/pandoc/build-docs.py), so the two PDFs are typeset by one
implementation; the book adds the book class and docs/pandoc/book.tex.

    scripts/book/pdf.py      # -> build/book/brain-book.{md,pdf}

Requires pandoc, xelatex and rsvg-convert (pandoc converts the SVG figures).

Swedish Embedded AB implements solutions for publishing technical material
from a single verified source. If your team needs expertise in machine
learning education and documentation, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bookspec import BOOK, ROOT, load  # noqa: E402

OUT = ROOT / "build" / "book"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]

spec = importlib.util.spec_from_file_location("build_docs", ROOT / "docs" / "pandoc" / "build-docs.py")
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def latex(cmd: str) -> str:
    return f"`{cmd}`{{=latex}}\n\n"


def page(rel: str, unnumbered=False, drop_title=False) -> str:
    """One page of docs/book, with its links and images rewritten for the PDF."""
    path = BOOK / rel
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.S)
    if drop_title:
        text = re.sub(r"^# .*\n", "", text, count=1, flags=re.M)
    elif unnumbered:
        text = re.sub(r"^(# .*)$", r"\1 {-}", text, count=1, flags=re.M)
    return docs.transform(f"docs/book/{rel}", text).strip() + "\n\n"


def source_date() -> str:
    r = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%B %Y", "--", "docs/book"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or docs.FALLBACK_DATE


def build_markdown() -> Path:
    book = load()
    parts = [f"""---
title: "{book.title}"
subtitle: "{book.subtitle}"
author: "{book.author}"
date: "{source_date()}"
toc-title: "Contents"
---

"""]
    # Front-matter sections are unnumbered; numbering resumes with Volume I.
    parts.append(latex("\\frontmatter\\setcounter{secnumdepth}{-2}"))
    parts += [page(rel, unnumbered=True) for rel in book.front]
    parts.append(latex("\\mainmatter\\setcounter{secnumdepth}{3}"))
    for v in book.volumes:
        written = [ch for ch in v.chapters if ch.written]
        if not written:
            continue
        title = f"Volume {ROMAN[v.number - 1]}: {v.title}"
        parts.append(latex(f"\\part{{{title}}}"))
        parts.append(page(v.intro_rel, drop_title=True))
        parts += [page(ch.rel) for ch in written]
    parts.append(latex("\\appendix"))
    parts += [page(rel) for rel in book.back]
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "brain-book.md"
    md.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {md.relative_to(ROOT)} ({md.stat().st_size // 1024} KiB)")
    return md


def main() -> int:
    md = build_markdown()
    _, bad = docs.build_pdf(
        str(md), pdf=str(OUT / "brain-book.pdf"),
        extra=("--toc-depth=1", "--top-level-division=chapter", "-V", "documentclass=book",
               "-V", "classoption=openright", "-V", "mainfont=P052", "-V", "mathfont=Latin Modern Math",
               "-H", str(ROOT / "docs" / "pandoc" / "book.tex")))
    if bad:
        print("WARNING: some content overflows the page; see build/book/brain-book.pandoc.log", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
