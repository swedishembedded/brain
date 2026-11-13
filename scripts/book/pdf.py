#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Typeset the book as PDFs, separate from the engineering manual: the whole
book as one file, and each volume with written chapters as a file of its own.

Reading order comes from docs/book/book.yaml (the same specification the
published navigation is generated from): front matter, then each volume with
written chapters as a LaTeX part (its introduction first), then the appendices.
A volume's file has the same front matter, the volume's introduction and its
chapters numbered from 1 (as the text refers to them, "Volume V, chapter 2"),
and the glossary and author page; links between chapters are plain text in
every PDF, so a volume on its own has none that dangle.
Page transforms and the pandoc invocation are the documentation build's own
(docs/pandoc/build-docs.py), so the two PDFs are typeset by one
implementation; the book adds the book class and docs/pandoc/book.tex.

    scripts/book/pdf.py               # the whole book and every written volume
    scripts/book/pdf.py --whole       # -> build/book/brain-book.{md,pdf}
    scripts/book/pdf.py --volumes     # every written volume, each on its own
    scripts/book/pdf.py --volume v05  # -> build/book/volumes/brain-book-v05-<dir>.{md,pdf}
    scripts/book/pdf.py -j 3          # how many PDFs typeset at once (default 2)

Requires pandoc, xelatex and rsvg-convert (pandoc converts the SVG figures).

Swedish Embedded AB implements solutions for publishing technical material
from a single verified source. If your team needs expertise in machine
learning education and documentation, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import importlib.util
import re
from concurrent.futures import ProcessPoolExecutor
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


def title_header(book, out: Path, title: str, subtitle: str) -> Path:
    """The title page and its reverse (the colophon), from book.yaml: `title`
    and `subtitle` are the book's own, or a volume's title over the book's."""
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "unknown"
    out.write_text(rf"""\makeatletter
\renewcommand{{\maketitle}}{{%
  \begin{{titlepage}}\raggedright
    \vspace*{{0.12\textheight}}
    {{\color{{accent}}\rule{{\linewidth}}{{1.5pt}}}}\par\vspace{{18pt}}
    {{\displayfont\fontsize{{30}}{{36}}\selectfont\bfseries\color{{ink}} {tex(title)}\par}}
    \vspace{{14pt}}
    \parbox{{0.86\linewidth}}{{\raggedright\large\itshape\color{{muted}} {tex(subtitle)}}}\par
    \vspace{{18pt}}{{\color{{accent}}\rule{{\linewidth}}{{0.6pt}}}}\par
    \vspace{{0.18\textheight}}
    {{\displayfont\Large\color{{ink}} {tex(book.author)}\par}}
    \vfill
    {{\displayfont\small\color{{muted}} Swedish Embedded AB\par}}
  \end{{titlepage}}
  \thispagestyle{{empty}}
  {{\small\color{{muted}}
    \vspace*{{\fill}}
    \noindent\textbf{{\color{{ink}}{tex(title)}}}\\[2pt]
    {tex(subtitle)}\\[10pt]
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


def build_markdown(book, out: Path, title: str, subtitle: str, volume=None) -> Path:
    """The markdown of one PDF: the whole book, or `volume` alone."""
    parts = [f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{book.author}"
date: "{source_date()}"
toc-title: "Contents"
---

"""]
    # Front-matter sections are unnumbered; numbering resumes with Volume I
    # (or with the volume's first chapter).
    parts.append(latex("\\frontmatter\\setcounter{secnumdepth}{-2}"))
    parts += [page(rel, unnumbered=True) for rel in book.front]
    if volume is not None:
        parts.append(page(volume.intro_rel, unnumbered=True))
    parts.append(latex("\\mainmatter\\setcounter{secnumdepth}{3}"))
    for v in [volume] if volume is not None else book.volumes:
        written = [ch for ch in v.chapters if ch.written]
        if not written:
            continue
        if volume is None:
            parts.append(latex(f"\\part{{{v.title}}}"))
            parts.append(page(v.intro_rel, drop_title=True))
        parts += [page(ch.rel) for ch in written]
    # Reference material becomes lettered appendices; the author page closes
    # the book unnumbered. The chronology covers the whole engine, so only the
    # whole book carries it.
    back = [rel for rel in book.back if volume is None or rel != "history/chronology.md"]
    parts.append(latex("\\appendix"))
    parts += [page(rel) for rel in back if rel != ABOUT]
    parts.append(latex("\\backmatter"))
    parts += [page(rel, unnumbered=True) for rel in back if rel == ABOUT]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KiB)")
    return out


REPORT = OUT / "overflow.txt"


def overflow_report(log: str, report: Path = REPORT) -> list[str]:
    """Every line of the final LaTeX run wider than the text, with its printed
    page number and the start of what overflows, written to `report`.

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
    report.write_text("".join(f + "\n" for f in found), encoding="utf-8")
    return found


def typeset(name: str, title: str, subtitle: str, volume=None) -> list[str]:
    """Build one PDF, `<name>.pdf`, and return its problems (none when clean)."""
    book = load()
    base = OUT / name
    md = build_markdown(book, base.with_suffix(".md"), title, subtitle, volume)
    header = title_header(book, base.with_name(base.name + ".title.tex"), title, subtitle)
    docs.build_pdf(
        str(md), pdf=str(base.with_suffix(".pdf")),
        extra=("--toc-depth=1", "--top-level-division=chapter", "-V", "documentclass=book",
               "-V", "classoption=openright", "-V", "mainfont=P052", "-V", "mathfont=Latin Modern Math",
               "-H", str(ROOT / "docs" / "pandoc" / "book.tex"), "-H", str(header),
               "--lua-filter", str(ROOT / "docs" / "pandoc" / "book-tables.lua")))
    log = base.with_suffix(".pandoc.log").read_text(encoding="utf-8", errors="replace")
    problems = []
    report = REPORT if volume is None else base.with_name(base.name + ".overflow.txt")
    overflows = overflow_report(log, report)
    if overflows:
        problems.append(f"{len(overflows)} line(s) run past the margin; see {report.relative_to(ROOT)}")
    # A glyph the font lacks is silently dropped from the page; refuse it.
    missing = sorted(set(re.findall(r"Missing character: There is no (\S+) \((U\+[0-9A-F]+)\)", log)))
    if missing:
        problems.append("missing glyphs (the PDF would drop them): " + ", ".join(f"{c} {u}" for c, u in missing))
    # A table wider than even a landscape page overflows it; the table filter
    # names each one.
    too_wide = re.findall(r"book-tables: (.+)", log)
    if too_wide:
        problems.append("tables too wide to typeset (split or narrow them):\n  " + "\n  ".join(too_wide))
    return [f"{name}: {p}" for p in problems]


def jobs(args: list[str]) -> list[tuple]:
    """(name, title, subtitle, volume) of every PDF the arguments ask for."""
    book = load()
    whole = ("brain-book", book.title, book.subtitle, None)
    written = [v for v in book.volumes if any(ch.written for ch in v.chapters)]
    def of(v):
        n = book.volumes.index(v)
        return (f"volumes/brain-book-{v.dir}", f"Volume {ROMAN[n]}: {v.title}", book.title, v)
    if "--whole" in args:
        return [whole]
    if "--volumes" in args:
        return [of(v) for v in written]
    if "--volume" in args:
        want = args[args.index("--volume") + 1]
        hit = [v for v in written if v.id == want or v.dir == want]
        if not hit:
            sys.exit(f"no written volume {want!r}; written: {', '.join(v.id for v in written)}")
        return [of(hit[0])]
    return [whole] + [of(v) for v in written]


def main() -> int:
    args = sys.argv[1:]
    # A mistyped option must not fall through to building the whole book.
    valued = {"--volume", "-j"}
    known = {"--whole", "--volumes"} | valued
    i = 0
    while i < len(args):
        if args[i] not in known:
            print(f"pdf.py: unknown argument {args[i]}", file=sys.stderr)
            print("usage: pdf.py [--whole | --volumes | --volume ID] [-j N]", file=sys.stderr)
            return 2
        i += 2 if args[i] in valued else 1
    # Each PDF is one single-threaded xelatex chain; a few run side by side.
    workers = int(args[args.index("-j") + 1]) if "-j" in args else 2
    todo = jobs(args)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
        problems = [p for found in pool.map(typeset, *zip(*todo)) for p in found]
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
