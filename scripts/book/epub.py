#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Build the book as EPUB 3 files, one beside each PDF, and verify each one.

The editions and their pages are edition.py's, shared with the PDF build
(pdf.py): the whole book, each written volume, and the notation and
pseudocode reference. Each edition becomes one EPUB under build/book/:

- Structure: a volume is a top-level section, its introduction first, and
  each chapter a section under it, numbered within its volume as the text
  refers to it ("Volume V, chapter 2"); front matter, appendices (lettered as
  in the PDF) and the author page are top-level sections of their own. The
  navigation lists volumes and chapters.
- Links between chapters stay links when the target is in the same edition
  and become plain text when it is not, so no edition has a link that
  dangles.
- Math is MathML. A display formula too wide for a phone or a tablet's
  column is set on several lines, broken where a typesetter would break it
  (docs/pandoc/book-math-lines.lua); one that cannot be broken is set
  smaller to fit, and so is a table too wide for the line. Nothing scrolls:
  a paging reading system cannot split a scrolling box across pages.
- Figures are the SVG files themselves (numbered per
  chapter, as the text cites them), tables are HTML tables.

A Kindle edition (`--kindle`) differs only in its figures. Amazon converts
an EPUB to its own format, and its SVG support is partial: a figure drawn at
its own pixel size into a smaller box comes out cropped instead of scaled.
So each SVG figure is rasterized (KINDLE_FIGURE_PX wide, white ground, a
small palette to keep the file small; Amazon's delivery fee is per megabyte)
and sized to the width of the page, as Amazon's guidelines ask. Amazon
converts every uploaded EPUB to its own format; this one uses only what
that conversion handles reliably. Both are built by default.

Each EPUB is then verified (`verify`): the container is well formed, every
page parses as XHTML, every link and image resolves, and every formula,
table and figure of the source is in the EPUB, rendered (a formula pandoc
cannot convert to MathML is left as TeX source, which fails the check). Last,
epubcheck checks it against the EPUB specification.

Memory: pandoc holds a whole edition in memory, about 2.3 GB at its peak for
the whole book in November 2026, so each pandoc runs under a heap limit
(PANDOC_HEAP) and stops with "Heap exhausted" instead of exhausting the
machine; -j N runs N editions at once, each under that limit.

    scripts/book/epub.py               # every edition, each as EPUB and as Kindle EPUB
    scripts/book/epub.py --whole       # -> build/book/brain-book.epub
    scripts/book/epub.py --volumes     # every written volume and the reference
    scripts/book/epub.py --volume v05  # -> build/book/volumes/FFPIM-Volume-05-<Title>.epub
    scripts/book/epub.py --reference   # notation and pseudocode alone
    scripts/book/epub.py --kindle      # the Kindle EPUBs alone (<name>-Kindle.epub)
    scripts/book/epub.py --standard    # the standard EPUBs alone
    scripts/book/epub.py --check FILE  # verify an EPUB that was already built
    scripts/book/epub.py -j 3          # how many EPUBs are built at once (default 2)

Requires pandoc, rsvg-convert (the cover, and the Kindle figures), ImageMagick
(the Kindle figures' palette) and epubcheck.

Swedish Embedded AB implements solutions for publishing technical material
from a single verified source, in print and electronic formats. If your team
needs expertise in documentation and e-book pipelines for engineering
content, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import html
import importlib.util
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bookspec import ROOT, Book, load  # noqa: E402
from edition import (ABOUT, AUTHOR_PHOTO, ROMAN, Edition, output, parse, read, select,  # noqa: E402
                     source_date, source_day, source_revision)

PANDOC = ROOT / "docs" / "pandoc"
CSS = PANDOC / "epub.css"
FILTER = PANDOC / "book-epub.lua"
MATH_LINES = PANDOC / "book-math-lines.lua"
FIGURES = PANDOC / "book-figures.lua"
# The accent and ink colours of the PDF (docs/pandoc/book.tex), for the cover.
ACCENT, INK, MUTED = "#2B6CB0", "#1A202C", "#4A5568"
# The most memory one pandoc may take; well above the whole book's peak.
PANDOC_HEAP = "4g"
# A Kindle edition's figures: 1100 px wide (sharp across a 300 ppi e-reader
# page) in a 64-colour palette, which keeps a figure's text and lines clean
# at about 25 KiB (38 KiB at 1400 px and 256 colours; 32 colours saves no
# more), measured on 30 of the book's figures.
KINDLE_FIGURE_PX = 1100
KINDLE_COLOURS = 64
KINDLE_CACHE = ROOT / "build" / "book" / "kindle-figures"
# A cover the size e-book stores ask for (1.5:1, 1600 px wide).
COVER_W, COVER_H = 1600, 2400

spec = importlib.util.spec_from_file_location("build_docs", PANDOC / "build-docs.py")
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)

LINK = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)\s]+?\.md)(?:#[^)]*)?\)")
FENCE = re.compile(r"^(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})(\s+)(.*)$")


def anchor(rel: str) -> str:
    """The identifier of a page's title: unique in the book, stable across builds."""
    return "page-" + re.sub(r"[^a-z0-9]+", "-", rel.removesuffix(".md").lower()).strip("-")


def links(rel: str, text: str, pages: set[str]) -> str:
    """Links to another page of this edition point at its title; a link to a
    page outside the edition becomes its text. A fragment is dropped: section
    identifiers are not unique across the pages of an edition."""
    def repl(m):
        target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), m.group(2)))
        return f"[{m.group(1)}](#{anchor(target)})" if target in pages else m.group(1)
    return LINK.sub(repl, text)


# A reference to a chapter in the text: "Volume II, Chapter 1", or
# "Chapter 4" for a chapter of the same volume.
REFERENCE = re.compile(r"\b(?:Volume ([IVX]+),\s+)?[Cc]hapter (\d+)\b")
# What a reference inside is already literal or already a link: code, math,
# a markdown link or image.
PROTECTED = re.compile(r"(`+).*?\1|\$\$.*?\$\$|\$[^$\n]+\$|!?\[[^\]]*\]\([^)]*\)", re.S)


def references(rel: str, text: str, pages: set[str], book: Book) -> str:
    """Each chapter the text refers to by number becomes a link to it when it
    is in this edition: a reader of the whole book can follow every
    "Volume I, Chapter 10", and a volume read on its own links the chapters
    it holds. Headings, code and math are left as they are."""
    here = next((v for v in book.volumes if rel.startswith(v.dir + "/")), None)

    def target(m):
        if m.group(1):
            if m.group(1) not in ROMAN:
                return None
            volume = book.volumes[ROMAN.index(m.group(1))]
        elif here is not None:
            volume = here
        else:
            return None
        n = int(m.group(2))
        if not 1 <= n <= len(volume.chapters):
            return None
        ch = volume.chapters[n - 1].rel
        return ch if ch in pages and ch != rel else None

    def plain(part):
        def repl(m):
            t = target(m)
            return f"[{m.group(0)}](#{anchor(t)})" if t else m.group(0)
        return REFERENCE.sub(repl, part)

    out, fence = [], None
    for line in text.split("\n"):
        f = FENCE.match(line)
        if f and (fence is None or f.group(1)[0] == fence[0]):
            fence = None if fence else f.group(1)
        if fence or f or HEADING.match(line):
            out.append(line)
            continue
        parts, last = [], 0
        for m in PROTECTED.finditer(line):
            parts += [plain(line[last:m.start()]), m.group(0)]
            last = m.end()
        out.append("".join(parts) + plain(line[last:]))
    return "\n".join(out)


def headings(text: str, shift: int, title: str, classes: str) -> str:
    """Every heading outside code moved `shift` levels down; the page's title
    (its first heading) replaced by `title` with `classes` attached. A page
    at the top level (not shifted) keeps its sections out of the navigation,
    which lists volumes and chapters only."""
    out, fence, first = [], None, True
    for line in text.split("\n"):
        f = FENCE.match(line)
        if f and (fence is None or f.group(1)[0] == fence[0] and len(f.group(1)) >= len(fence)):
            fence = None if fence else f.group(1)
        elif fence is None and (h := HEADING.match(line)):
            if first:
                line, first = f"{'#' * (len(h.group(1)) + shift)} {title} {{{classes}}}", False
            else:
                line = f"{'#' * min(6, len(h.group(1)) + shift)}{h.group(2)}{h.group(3)}"
                if shift == 0 and len(h.group(1)) == 2:
                    line += " {.unlisted}"
        out.append(line)
    return "\n".join(out)


def page(rel: str, pages: set[str], book: Book, level: int, label: str = "", kind: str = "") -> str:
    """One page of docs/book for the EPUB: its title at `level` (1 for a
    volume, front or back page, 2 for a chapter under its volume), prefixed
    with `label` and carrying the class `kind`, links resolved within the
    edition and image paths made relative to the repository."""
    text = read(rel)
    m = re.search(r"^# (.*)$", text, flags=re.M)
    if not m:
        raise SystemExit(f"docs/book/{rel}: no title heading")
    title = f"{label}{m.group(1)}"
    classes = f"#{anchor(rel)}" + (f" .{kind}" if kind else "")
    text = headings(links(rel, references(rel, text, pages, book), pages), level - 1, title, classes)
    return docs.transform(f"docs/book/{rel}", text).strip() + "\n\n"


def colophon(book: Book, edition: Edition) -> str:
    """The reverse of the title page, as in the PDF."""
    return f"""# Copyright {{.unnumbered .colophon}}

**{edition.title}**\\
{edition.subtitle}

Copyright © 2026 {book.author}. This work is licensed under the Creative
Commons Attribution 4.0 International License (CC BY 4.0).

Edition of {source_date()}, built from source revision `{source_revision()}`.
Every worked example in this edition was computed, and every figure
generated, from that revision.

Corrections and questions are welcome: <{book.author_email}>. Each correction
is recorded in the history of the source, so every edition can be traced to
the one before it.

Published by Swedish Embedded AB, Sweden. <https://swedishembedded.com>

"""


def build_markdown(book: Book, edition: Edition, out: Path) -> Path:
    """The markdown of one EPUB; its metadata goes in through pandoc options."""
    pages = set(edition.pages())
    parts = [colophon(book, edition)]
    parts += [page(rel, pages, book, 1, kind="front") for rel in edition.front]
    for v in edition.volumes:
        parts.append(page(v.intro_rel, pages, book, 1, kind="volume"))
        parts += [page(ch.rel, pages, book, 2, f"{n}. ", "chapter")
                  for n, ch in enumerate(edition.written(v), start=1)]
    parts += [page(rel, pages, book, 1, f"Appendix {chr(ord('A') + i)}. ", "appendix")
              for i, rel in enumerate(edition.back)]
    about = page(ABOUT, pages, book, 1, kind="back")
    if AUTHOR_PHOTO.is_file():
        head, sep, body = about.partition("\n\n")
        photo = f"![{book.author}]({AUTHOR_PHOTO.relative_to(ROOT).as_posix()}){{.portrait}}\n\n"
        about = head + sep + photo + body
    parts.append(about)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


def wrap(text: str, chars: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > chars:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def cover(book: Book, edition: Edition, out: Path) -> Path:
    """A typographic cover in the PDF title page's design, as PNG: e-book
    readers show a raster cover in their library, not every one an SVG."""
    def block(lines, x, y, size, leading, **attrs):
        a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return "".join(f'<text x="{x}" y="{y + i * leading}" font-size="{size}" {a}>{html.escape(t)}</text>'
                       for i, t in enumerate(lines))
    margin = 150
    title = wrap(edition.title, 18)
    subtitle = wrap(edition.subtitle, 50)
    top = 560
    sub_top = top + len(title) * 118 + 60
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{COVER_W}" height="{COVER_H}">
<rect width="100%" height="100%" fill="#FFFFFF"/>
<rect x="{margin}" y="{top - 160}" width="{COVER_W - 2 * margin}" height="6" fill="{ACCENT}"/>
{block(title, margin, top, 100, 118, font_family="Nimbus Sans", font_weight="bold", fill=INK)}
{block(subtitle, margin, sub_top, 50, 66, font_family="P052", font_style="italic", fill=MUTED)}
<rect x="{margin}" y="{sub_top + len(subtitle) * 66 + 20}" width="{COVER_W - 2 * margin}" height="3" fill="{ACCENT}"/>
{block([book.author], margin, COVER_H - 520, 64, 0, font_family="Nimbus Sans", fill=INK)}
{block(["Swedish Embedded AB"], margin, COVER_H - 200, 40, 0, font_family="Nimbus Sans", fill=MUTED)}
</svg>"""
    src = out.with_name(out.name.removesuffix(".png") + ".svg")
    src.write_text(svg, encoding="utf-8")
    subprocess.run(["rsvg-convert", "-f", "png", "-o", str(out), str(src)], check=True)
    return out


def identifier(edition: Edition) -> str:
    """The same identifier for every build of an edition, so a reader's
    library sees a new build as an update of the book it has."""
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'https://swedishembedded.com/book/' + edition.name)}"


SVG_FIGURE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+\.svg)\)")


def raster(svg: Path) -> Path:
    """The Kindle PNG of an SVG figure, made once and kept under the hash of
    the SVG's content."""
    import hashlib
    out = KINDLE_CACHE / (hashlib.sha256(svg.read_bytes()).hexdigest()[:20] + ".png")
    if not out.is_file():
        KINDLE_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.png")
        subprocess.run(["rsvg-convert", "-w", str(KINDLE_FIGURE_PX), "-b", "white", "-f", "png", "-o", str(tmp), str(svg)],
                       check=True)
        # Maximum zlib compression with adaptive filtering: 7% smaller, same pixels.
        subprocess.run(["magick", str(tmp), "-strip", "-colors", str(KINDLE_COLOURS),
                        "-define", "png:compression-level=9", "-define", "png:compression-strategy=1",
                        "-define", "png:compression-filter=5", f"PNG8:{tmp}"], check=True)
        os.replace(tmp, out)
    return out


def kindle_figures(text: str) -> str:
    """The markdown of an edition with every SVG figure replaced by its PNG,
    sized to the width of the page."""
    return SVG_FIGURE.sub(lambda m: f"{m.group(1)}{raster(ROOT / m.group(2)).relative_to(ROOT).as_posix()}){{width=100%}}",
                          text)


def build(edition: Edition, kindle: bool = False) -> list[str]:
    """Build one EPUB (for Kindle when `kindle`), verify it, and return its
    problems (none when clean)."""
    book = load()
    suffix = "-Kindle" if kindle else ""
    md = build_markdown(book, edition, output(edition, f"{suffix}.epub.md"))
    if kindle:
        md.write_text(kindle_figures(md.read_text(encoding="utf-8")), encoding="utf-8")
    epub = output(edition, f"{suffix}.epub")
    image = cover(book, edition, output(edition, ".cover.png"))
    meta = {
        "title": edition.title, "subtitle": edition.subtitle, "creator": [{"role": "author", "text": book.author}],
        "publisher": "Swedish Embedded AB", "lang": "en-GB", "date": source_day(),
        "identifier": [{"scheme": "UUID", "text": identifier(edition) + ("-kindle" if kindle else "")}],
        "rights": f"Copyright © 2026 {book.author}. Licensed under CC BY 4.0.",
        "subject": "Machine learning",
    }
    meta_file = output(edition, f"{suffix}.epub.yaml")
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp = output(edition, f"{suffix}.epub.tmp")
    cmd = ["pandoc", "+RTS", f"-M{PANDOC_HEAP}", "-RTS", str(md), "-f", "markdown", "-t", "epub3", "-o", str(tmp),
           "--metadata-file", str(meta_file), "--epub-cover-image", str(image),
           "--css", str(CSS), "--lua-filter", str(MATH_LINES), "--lua-filter", str(FIGURES), "--lua-filter", str(FILTER), "--mathml",
           "--toc", "--toc-depth=2", "--split-level=2", "--highlight-style=tango",
           "--resource-path", str(ROOT)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return [f"{edition.name}: pandoc failed:\n{r.stderr[-3000:]}"]
    tmp.replace(epub)
    print(f"wrote {epub.relative_to(ROOT)} ({epub.stat().st_size // 1024} KiB)", flush=True)
    problems = [f"pandoc: {w}" for w in r.stderr.splitlines() if w.startswith(("[WARNING]", "book-epub: figure", "book-epub: raw HTML"))]
    counts = re.search(r"book-epub: counts math=(\d+) table=(\d+) image=(\d+)", r.stderr)
    if not counts:
        return [f"{edition.name}: the EPUB filter reported no counts"]
    problems += verify(epub, dict(zip(("Math", "Table", "Image"), map(int, counts.groups()))))
    return [f"{edition.name}: {p}" for p in problems]


# ---- verification -------------------------------------------------------------

NS = {"opf": "http://www.idpf.org/2007/opf", "c": "urn:oasis:names:tc:opendocument:xmlns:container",
      "x": "http://www.w3.org/1999/xhtml", "m": "http://www.w3.org/1998/Math/MathML",
      "svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
XHTML = "{%s}" % NS["x"]


def verify(epub: Path, want: dict[str, int] | None = None) -> list[str]:
    """Everything that must hold for the EPUB to read as the book does: its
    structure and content (`structure`) and conformance (`epubcheck`)."""
    return structure(epub, want) + epubcheck(epub)


def structure(epub: Path, want: dict[str, int] | None = None) -> list[str]:
    """The container, pages, links and content of the EPUB; each problem
    names the file inside the EPUB. `want` is how many formulas, tables and
    images the source has (the filter's counts), when known.

    Pages are parsed one at a time and only their identifiers and links are
    kept, so memory stays at one page however large the book."""
    problems = []
    with zipfile.ZipFile(epub) as z:
        names = z.namelist()
        info = z.infolist()[0]
        if info.filename != "mimetype" or info.compress_type != zipfile.ZIP_STORED \
                or z.read("mimetype") != b"application/epub+zip":
            problems.append("mimetype must be the first entry, stored uncompressed")
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.find(".//c:rootfile", NS).get("full-path")
        opf_dir = posixpath.dirname(opf_path)
        opf = ET.fromstring(z.read(opf_path))
        manifest = {item.get("id"): item for item in opf.find("opf:manifest", NS)}
        hrefs = {posixpath.normpath(posixpath.join(opf_dir, i.get("href"))): i for i in manifest.values()}
        for path in sorted(set(hrefs) - set(names)):
            problems.append(f"{path}: in the manifest but not in the container")
        for path in sorted(set(names) - set(hrefs) - {"mimetype", "META-INF/container.xml",
                                                        "META-INF/com.apple.ibooks.display-options.xml", opf_path}):
            problems.append(f"{path}: in the container but not in the manifest")
        for ref in opf.find("opf:spine", NS):
            if ref.get("idref") not in manifest:
                problems.append(f"spine names {ref.get('idref')}, which the manifest does not have")

        ids, refs = {}, []
        found = {"Math": 0, "Table": 0, "Image": 0}
        for path, item in hrefs.items():
            if item.get("media-type") != "application/xhtml+xml" or path not in names:
                continue
            try:
                root = ET.fromstring(z.read(path))
            except ET.ParseError as e:
                problems.append(f"{path}: not well-formed XHTML: {e}")
                continue
            ids[path] = {el.get("id") for el in root.iter() if el.get("id")}
            here = posixpath.dirname(path)
            for el in root.iter():
                tag = el.tag
                if tag == f"{{{NS['m']}}}math":
                    found["Math"] += 1
                    if not any(True for _ in el.iter() if _ is not el):
                        problems.append(f"{path}: an empty formula")
                elif tag == XHTML + "table":
                    found["Table"] += 1
                elif tag == XHTML + "img":
                    found["Image"] += 1
                    src = posixpath.normpath(posixpath.join(here, el.get("src", "")))
                    if src not in names:
                        problems.append(f"{path}: image {el.get('src')} is not in the EPUB")
                    if not el.get("alt") and "cover" not in path:
                        problems.append(f"{path}: image {el.get('src')} has no alt text")
                elif tag == f"{{{NS['svg']}}}image":
                    # The cover: an image in an SVG wrapper, not a figure of the book.
                    src = posixpath.normpath(posixpath.join(here, el.get(f"{{{NS['xlink']}}}href", "")))
                    if src not in names:
                        problems.append(f"{path}: image {src} is not in the EPUB")
                elif tag == XHTML + "span" and "math" in (el.get("class") or "").split():
                    # pandoc leaves a formula it cannot convert to MathML as
                    # its TeX source in a span of class "math".
                    if el.find("m:math", NS) is None:
                        problems.append(f"{path}: formula left as TeX source: {''.join(el.itertext())[:80]}")
                if tag == XHTML + "a" and el.get("href") and not re.match(r"[a-z]+:", el.get("href")):
                    refs.append((path, el.get("href")))
            del root
        for path, href in refs:
            target, _, frag = href.partition("#")
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), target)) if target else path
            if target not in ids:
                problems.append(f"{path}: link {href} leads outside the EPUB's pages")
            elif frag and frag not in ids[target]:
                problems.append(f"{path}: link {href} names an anchor that does not exist")
        if want is not None:
            for kind in want:
                if found[kind] != want[kind]:
                    problems.append(f"{want[kind]} {kind.lower()} element(s) in the source, {found[kind]} in the EPUB")

        for path, item in hrefs.items():
            if item.get("media-type") != "image/svg+xml" or path not in names:
                continue
            try:
                svg = ET.fromstring(z.read(path))
            except ET.ParseError as e:
                problems.append(f"{path}: not well-formed SVG: {e}")
                continue
            for el in svg.iter():
                if el.tag.endswith("}script") or el.tag == "script":
                    problems.append(f"{path}: SVG carries a script")
                href = el.get(f"{{{NS['xlink']}}}href") or el.get("href")
                if href and not href.startswith(("#", "data:")):
                    problems.append(f"{path}: SVG refers to {href} outside itself")
    return problems


def epubcheck(epub: Path) -> list[str]:
    """Conformance to the EPUB specification, by the W3C's own checker."""
    if shutil.which("epubcheck") is None:
        return ["epubcheck is not installed, so conformance is unchecked (sudo apt install epubcheck)"]
    r = subprocess.run(["epubcheck", "--quiet", str(epub)], capture_output=True, text=True)
    if r.returncode == 0:
        return []
    report = [ln for ln in (r.stdout + r.stderr).splitlines() if re.match(r"(ERROR|FATAL|WARNING)", ln)]
    return [f"epubcheck: {ln}" for ln in (report or [(r.stdout + r.stderr).strip()[-2000:]])]


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--check"]:
        if len(args) != 2:
            print("usage: epub.py --check FILE.epub", file=sys.stderr)
            return 2
        problems = verify(Path(args[1]))
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    workers = parse(args, "epub.py", {"--kindle", "--standard"})
    if workers is None:
        return 2
    editions = select([a for a in args if a not in ("--kindle", "--standard")])
    variants = [False] * ("--kindle" not in args) + [True] * ("--standard" not in args)
    todo = [(e, k) for e in editions for k in variants]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as pool:
        problems = [p for found in pool.map(build, *zip(*todo)) for p in found]
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(todo)} EPUB(s) built, {len(problems)} problem(s)", file=sys.stderr if problems else sys.stdout)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
