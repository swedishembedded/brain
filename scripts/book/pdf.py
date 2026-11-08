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
ABOUT = "about-the-author.md"
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


def tex(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def title_header(book) -> Path:
    """The title page and its reverse (the colophon), from book.yaml."""
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "unknown"
    out = OUT / "title.tex"
    out.write_text(rf"""\makeatletter
\renewcommand{{\maketitle}}{{%
  \begin{{titlepage}}\raggedright
    \vspace*{{0.12\textheight}}
    {{\color{{accent}}\rule{{\linewidth}}{{1.5pt}}}}\par\vspace{{18pt}}
    {{\displayfont\fontsize{{30}}{{36}}\selectfont\bfseries\color{{ink}} {tex(book.title)}\par}}
    \vspace{{14pt}}
    \parbox{{0.86\linewidth}}{{\raggedright\large\itshape\color{{muted}} {tex(book.subtitle)}}}\par
    \vspace{{18pt}}{{\color{{accent}}\rule{{\linewidth}}{{0.6pt}}}}\par
    \vspace{{0.18\textheight}}
    {{\displayfont\Large\color{{ink}} {tex(book.author)}\par}}
    \vfill
    {{\displayfont\small\color{{muted}} Swedish Embedded AB\par}}
  \end{{titlepage}}
  \thispagestyle{{empty}}
  {{\small\color{{muted}}
    \vspace*{{\fill}}
    \noindent\textbf{{\color{{ink}}{tex(book.title)}}}\\[2pt]
    {tex(book.subtitle)}\\[10pt]
    Copyright \textcopyright\ 2026 {tex(book.author)}.\\[4pt]
    This work is licensed under the Creative Commons Attribution 4.0
    International License (CC BY 4.0).\\[10pt]
    Edition of {source_date()}, typeset from source revision \texttt{{{rev}}}.
    Every worked example in this edition was computed, and every figure
    generated, from that revision.\\[10pt]
    Corrections and questions are welcome:
    \href{{mailto:{book.author_email}}}{{{book.author_email}}}.
    Each correction is recorded in the history of the source, so every
    edition can be traced to the one before it.\\[10pt]
    Published by Swedish Embedded AB, Sweden.
    \href{{https://swedishembedded.com}}{{swedishembedded.com}}\par}}
  \cleardoublepage
}}
\makeatother
""", encoding="utf-8")
    return out


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
        parts.append(latex(f"\\part{{{v.title}}}"))
        parts.append(page(v.intro_rel, drop_title=True))
        parts += [page(ch.rel) for ch in written]
    # Reference material becomes lettered appendices; the author page closes
    # the book unnumbered.
    parts.append(latex("\\appendix"))
    parts += [page(rel) for rel in book.back if rel != ABOUT]
    parts.append(latex("\\backmatter"))
    parts += [page(rel, unnumbered=True) for rel in book.back if rel == ABOUT]
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "brain-book.md"
    md.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {md.relative_to(ROOT)} ({md.stat().st_size // 1024} KiB)")
    return md


REPORT = OUT / "overflow.txt"


def overflow_report(log: str) -> list[str]:
    """Every line of the final LaTeX run wider than the text, with its printed
    page number and the start of what overflows, written to REPORT.

    LaTeX logs a page as "[N" when it ships it out, so the first such mark
    after a warning is the page the line is on. A display equation shows as an
    overfull alignment; a table cell or a code span as an overfull paragraph.
    """
    last = log.split("LaTeX run number")[-1]
    found = []
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\) ([^\n]*)\n([^\n]*)", last):
        if float(m.group(1)) <= 2.0:
            continue  # a hairline
        page = re.search(r"\[(\d+)(?=[\]\s])", last[m.end():])
        what = "display or table row" if "detected" in m.group(2) else m.group(3).strip()[:100]
        found.append(f"page {page.group(1) if page else '?'}: {float(m.group(1)):.0f}pt  {what}")
    found = list(dict.fromkeys(found))
    REPORT.write_text("".join(f + "\n" for f in found), encoding="utf-8")
    return found


def main() -> int:
    md = build_markdown()
    title = title_header(load())
    _, bad = docs.build_pdf(
        str(md), pdf=str(OUT / "brain-book.pdf"),
        extra=("--toc-depth=1", "--top-level-division=chapter", "-V", "documentclass=book",
               "-V", "classoption=openright", "-V", "mainfont=P052", "-V", "mathfont=Latin Modern Math",
               "-H", str(ROOT / "docs" / "pandoc" / "book.tex"), "-H", str(title),
               "--lua-filter", str(ROOT / "docs" / "pandoc" / "book-tables.lua")))
    # A glyph the font lacks is silently dropped from the page; refuse it.
    log = (OUT / "brain-book.pandoc.log").read_text(encoding="utf-8", errors="replace")
    overflows = overflow_report(log)
    if overflows:
        print(f"{len(overflows)} line(s) run past the margin; see {REPORT.relative_to(ROOT)}", file=sys.stderr)
        return 1
    missing = sorted(set(re.findall(r"Missing character: There is no (\S+) \((U\+[0-9A-F]+)\)", log)))
    if missing:
        print("missing glyphs (the PDF would drop them): " +
              ", ".join(f"{c} {u}" for c, u in missing), file=sys.stderr)
        return 1
    # A table wider than even a landscape page overflows it; the table filter
    # names each one.
    too_wide = re.findall(r"book-tables: (.+)", log)
    if too_wide:
        print("tables too wide to typeset (split or narrow them):\n  " + "\n  ".join(too_wide),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
