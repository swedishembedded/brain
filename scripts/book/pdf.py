#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Typeset the book as PDFs, separate from the engineering manual: the whole
book as one file, and each volume with written chapters as a file of its own.

Which files are built and the pages each holds come from edition.py, shared
with the EPUB build (epub.py): front matter, then each volume with written
chapters as a LaTeX part (its introduction first), then the appendices. A
volume's file has the same front matter, the volume's introduction and its
chapters numbered from 1, and the glossary and author page; links between
chapters are plain text in every PDF, so a volume on its own has none that
dangle.
Page transforms and the pandoc invocation are the documentation build's own
(docs/pandoc/build-docs.py), so the two PDFs are typeset by one
implementation; the book adds the book class and docs/pandoc/book.tex.

    scripts/book/pdf.py               # the whole book and every written volume
    scripts/book/pdf.py --whole       # -> build/book/brain-book.{md,pdf}
    scripts/book/pdf.py --volumes     # every written volume, each on its own, and the reference PDF
    scripts/book/pdf.py --reference   # notation and pseudocode alone -> build/book/volumes/FFPIM-Appendix-...
    scripts/book/pdf.py --volume v05  # -> build/book/volumes/FFPIM-Volume-05-<Title>.{md,pdf}
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bookspec import BOOK, ROOT, load  # noqa: E402
from edition import ABOUT, AUTHOR_PHOTO, OUT, Edition, output, parse, read, select, source_date, source_revision  # noqa: E402

spec = importlib.util.spec_from_file_location("build_docs", ROOT / "docs" / "pandoc" / "build-docs.py")
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def latex(cmd: str) -> str:
    return f"`{cmd}`{{=latex}}\n\n"


def page(rel: str, unnumbered=False, drop_title=False) -> str:
    """One page of docs/book, with its links and images rewritten for the PDF."""
    text = read(rel)
    if drop_title:
        text = re.sub(r"^# .*\n", "", text, count=1, flags=re.M)
    elif unnumbered:
        # A starred chapter does not update the running head, so the page
        # sets it to its own title instead of the previous chapter's.
        m = re.search(r"^# (.*)$", text, flags=re.M)
        if m:
            mark = latex(rf"\markboth{{{tex(m.group(1))}}}{{}}")
            text = f"{text[:m.start()]}# {m.group(1)} {{-}}\n\n{mark}{text[m.end():].lstrip()}"
    return docs.transform(f"docs/book/{rel}", text).strip() + "\n\n"


def tex(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def title_header(book, out: Path, title: str, subtitle: str) -> Path:
    """The title page and its reverse (the colophon), from book.yaml: `title`
    and `subtitle` are the book's own, or a volume's title over the book's."""
    rev = source_revision()
    out.write_text(rf"""\makeatletter
\renewcommand{{\bookauthor}}{{{tex(book.author)}}}
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
  {portrait_page(book)}
}}
\makeatother
""", encoding="utf-8")
    return out


def portrait_page(book) -> str:
    """A page of its own with the author's portrait and name under it, or
    nothing when the photograph is absent."""
    if not AUTHOR_PHOTO.is_file():
        return ""
    return rf"""\thispagestyle{{empty}}
  \vspace*{{\fill}}
  \begin{{center}}
    \includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{{AUTHOR_PHOTO}}}\par\vspace{{14pt}}
    {{\displayfont\Large\color{{ink}} {tex(book.author)}\par}}
  \end{{center}}
  \vspace*{{\fill}}
  \cleardoublepage"""


def about_page() -> str:
    """The author page, with the portrait under its heading."""
    text = page(ABOUT, unnumbered=True)
    if not AUTHOR_PHOTO.is_file():
        return text
    head, sep, body = text.partition("\n\n")
    photo = latex(rf"\begin{{center}}\includegraphics[height=0.42\textheight]{{{AUTHOR_PHOTO}}}\end{{center}}")
    return head + sep + photo + body


def volume_title_page(volume) -> str:
    """The title of a volume on a page of its own, for a volume's PDF; its
    introduction starts on the next page under the same running head."""
    intro = (BOOK / volume.intro_rel).read_text(encoding="utf-8")
    title = tex(re.search(r"^# (.*)$", intro, flags=re.M).group(1))
    return latex(
        rf"\cleardoublepage\thispagestyle{{empty}}\vspace*{{0.28\textheight}}"
        rf"{{\raggedright\displayfont\fontsize{{28}}{{34}}\selectfont\bfseries\color{{ink}} {title}\par}}"
        rf"\vspace{{10pt}}{{\color{{accent}}\rule{{\linewidth}}{{1.2pt}}}}"
        rf"\addcontentsline{{toc}}{{chapter}}{{{title}}}\markboth{{{title}}}{{}}\cleardoublepage")


def build_markdown(book, edition: Edition, out: Path) -> Path:
    """The markdown of one PDF, with the LaTeX that sets its divisions."""
    parts = [f"""---
title: "{edition.title}"
subtitle: "{edition.subtitle}"
author: "{book.author}"
date: "{source_date()}"
toc-title: "Contents"
---

"""]
    # Front-matter sections are unnumbered; numbering resumes with Volume I
    # (or with the volume's first chapter).
    parts.append(latex("\\frontmatter\\setcounter{secnumdepth}{-2}"))
    parts += [page(rel, unnumbered=True) for rel in edition.front]
    if edition.kind == "volume":
        parts.append(volume_title_page(edition.volumes[0]))
        parts.append(page(edition.volumes[0].intro_rel, drop_title=True))
    parts.append(latex("\\mainmatter\\setcounter{secnumdepth}{3}"))
    for v in edition.volumes:
        if edition.kind == "book":
            parts.append(latex(f"\\part{{{v.title}}}"))
            parts.append(page(v.intro_rel, drop_title=True))
        parts += [page(ch.rel) for ch in edition.written(v)]
    # Reference material becomes lettered appendices; the author page closes
    # the book unnumbered.
    parts.append(latex("\\appendix"))
    parts += [page(rel) for rel in edition.back]
    parts.append(latex("\\backmatter"))
    parts.append(about_page())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KiB)", flush=True)
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


def typeset(edition: Edition) -> list[str]:
    """Build one PDF and return its problems (none when clean)."""
    book = load()
    md = build_markdown(book, edition, output(edition, ".md"))
    header = title_header(book, output(edition, ".title.tex"), edition.title, edition.subtitle)
    docs.build_pdf(
        str(md), pdf=str(output(edition, ".pdf")),
        extra=("--toc-depth=1", "--top-level-division=chapter", "-V", "documentclass=book",
               "-V", "classoption=openright", "-V", "mainfont=P052", "-V", "mathfont=Latin Modern Math",
               "-H", str(ROOT / "docs" / "pandoc" / "book.tex"), "-H", str(header),
               "--lua-filter", str(ROOT / "docs" / "pandoc" / "book-figures.lua"),
               "--lua-filter", str(ROOT / "docs" / "pandoc" / "book-tables.lua")))
    log = output(edition, ".pandoc.log").read_text(encoding="utf-8", errors="replace")
    problems = []
    report = REPORT if edition.kind == "book" else output(edition, ".overflow.txt")
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
    return [f"{edition.name}: {p}" for p in problems]


def main() -> int:
    args = sys.argv[1:]
    workers = parse(args, "pdf.py")
    if workers is None:
        return 2
    todo = select(args)
    # Each PDF is one single-threaded xelatex chain; a few run side by side.
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
        problems = [p for found in pool.map(typeset, todo) for p in found]
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
