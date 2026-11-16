# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Which files the book is published as, and what each one holds.

One answer shared by the PDF builder (pdf.py) and the EPUB builder (epub.py),
so the two formats always carry the same pages in the same order: the whole
book; each volume with written chapters on its own (its chapters numbered from
1, as the text refers to them, "Volume V, chapter 2"); and the notation and
pseudocode appendices as a reference of their own, which the volumes leave out
instead of repeating. Each format adds its own typesetting to these pages.

Swedish Embedded AB implements solutions for publishing technical material
from a single verified source. If your team needs expertise in machine
learning education and documentation, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import re
import subprocess
import sys
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from bookspec import BOOK, ROOT, Book, Volume, load

OUT = ROOT / "build" / "book"
ABOUT = "about-the-author.md"
CHRONOLOGY = "history/chronology.md"
# The author's portrait, shown on the author page (and in the PDF also on a
# page of its own before the contents).
AUTHOR_PHOTO = BOOK / "img" / "author.jpg"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]
# The reference pages every volume relies on. The whole book carries them as
# appendices; the volumes leave them to the reference edition.
SHARED_REFERENCE = ("notation.md", "pseudocode.md")
# Paths in the book that the source date and revision of an edition follow:
# a commit elsewhere in the repository leaves every edition as it was.
SOURCES = ("docs/book", "docs/pandoc", "scripts/book")


@dataclass
class Edition:
    """One published file of the book, without its extension.

    `kind` is "book" (the whole book, each volume a part with its
    introduction), "volume" (one volume alone, under a title page of its own)
    or "reference" (the shared reference pages alone). `back` are the
    appendices in order; the author page closes every edition after them."""
    name: str
    title: str
    subtitle: str
    kind: str
    front: list[str]
    volumes: list[Volume]
    back: list[str]

    def written(self, volume: Volume):
        return [ch for ch in volume.chapters if ch.written]

    def pages(self) -> list[str]:
        """Every page of docs/book in this edition, in reading order."""
        rels = list(self.front)
        for v in self.volumes:
            rels.append(v.intro_rel)
            rels += [ch.rel for ch in self.written(v)]
        return rels + list(self.back) + [ABOUT]


def read(rel: str) -> str:
    """A page of docs/book as the reader sees it: without HTML comments (the
    licence header, worked-example and concept markers)."""
    return re.sub(r"<!--.*?-->\n?", "", (BOOK / rel).read_text(encoding="utf-8"), flags=re.S)


def source_date() -> str:
    r = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%B %Y", "--", "docs/book"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "Unreleased"


def source_day() -> str:
    """The day the book last changed, as an ISO 8601 date (2026-11-16), for
    metadata a reading system parses."""
    r = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "--", "docs/book"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "1970-01-01"


def source_revision() -> str:
    """The last commit that changed what the book is built from, not HEAD."""
    r = subprocess.run(["git", "log", "-1", "--format=%h", "--abbrev=9", "--", *SOURCES],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def version() -> str:
    """brain's major and minor version, "v5.1": the workspace version in
    Cargo.toml, the one place a release (make release/*) sets it and the
    version its tag (v5.1.0) names."""
    raw = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]
    major, minor, _ = raw.split(".", 2)
    return f"v{major}.{minor}"


def words(text: str) -> str:
    """`text` as a file name: ASCII letters and digits only (an accented
    letter loses its accent), each word capitalized, joined by hyphens."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "-".join(w[0].upper() + w[1:] for w in re.findall(r"[A-Za-z0-9]+", ascii_text))


def signature(book: Book) -> str:
    """What every file name ends with: the author and the version of brain
    the edition was built from, e.g. Martin-Schroder-v5.1."""
    return f"{words(book.author)}-{version()}"


def volume_file(book: Book, n: int) -> str:
    """The file name of volume `n` (from 0) without its extension: the
    initials of the book's title, the volume number in two digits so the
    files sort in reading order, the volume's title, and the signature,
    e.g. FFPIM-Volume-02-What-Does-It-Mean-To-Learn-Martin-Schroder-v5.1."""
    acronym = "".join(w[0] for w in book.title.split() if w[0].isupper())
    return f"{acronym}-Volume-{n + 1:02d}-{words(book.volumes[n].title)}-{signature(book)}"


def whole(book: Book) -> Edition:
    """The whole book, named by its full title, e.g.
    From-First-Principles-To-Intelligent-Machines-Martin-Schroder-v5.1."""
    written = [v for v in book.volumes if any(ch.written for ch in v.chapters)]
    return Edition(f"{words(book.title)}-{signature(book)}", book.title, book.subtitle, "book",
                   list(book.front), written, [rel for rel in book.back if rel != ABOUT])


def volume(book: Book, v: Volume) -> Edition:
    n = book.volumes.index(v)
    # The chronology covers the whole engine, so only the whole book carries it.
    back = [rel for rel in book.back if rel not in (ABOUT, CHRONOLOGY) and rel not in SHARED_REFERENCE]
    return Edition(f"volumes/{volume_file(book, n)}", f"Volume {ROMAN[n]}: {v.title}", book.title,
                   "volume", list(book.front), [v], back)


def reference(book: Book) -> Edition:
    acronym = volume_file(book, 0).split("-")[0]
    return Edition(f"volumes/{acronym}-Appendix-Notation-And-Pseudocode-Conventions-{signature(book)}",
                   "Appendix: Notation and Pseudocode Conventions", book.title, "reference", [], [],
                   [rel for rel in book.back if rel in SHARED_REFERENCE])


def select(args: list[str]) -> list[Edition]:
    """The editions the builders' common arguments ask for:

        (none)        the whole book, every written volume and the reference
        --whole       the whole book
        --volumes     every written volume and the reference
        --volume ID   one written volume, by id or directory
        --reference   the reference alone
    """
    book = load()
    written = whole(book).volumes
    if "--whole" in args:
        return [whole(book)]
    if "--reference" in args:
        return [reference(book)]
    if "--volumes" in args:
        return [volume(book, v) for v in written] + [reference(book)]
    if "--volume" in args:
        want = args[args.index("--volume") + 1]
        hit = [v for v in written if v.id == want or v.dir == want]
        if not hit:
            raise SystemExit(f"no written volume {want!r}; written: {', '.join(v.id for v in written)}")
        return [volume(book, hit[0])]
    return [whole(book)] + [volume(book, v) for v in written] + [reference(book)]


SELECTORS = {"--whole", "--volumes", "--reference"}
VALUED = {"--volume", "-j"}


def parse(args: list[str], prog: str, extra: set[str] = frozenset()) -> int | None:
    """Reject an argument neither builder knows (so a mistyped option never
    falls through to building everything); returns the number of parallel
    jobs (-j N, default 2), or None after printing the usage."""
    i = 0
    while i < len(args):
        if args[i] not in SELECTORS | VALUED | extra:
            print(f"{prog}: unknown argument {args[i]}", file=sys.stderr)
            print(f"usage: {prog} [--whole | --volumes | --volume ID | --reference] [-j N]"
                  + "".join(f" [{e}]" for e in sorted(extra)), file=sys.stderr)
            return None
        i += 2 if args[i] in VALUED else 1
    return int(args[args.index("-j") + 1]) if "-j" in args else 2


def output(edition: Edition, suffix: str) -> Path:
    return OUT / f"{edition.name}{suffix}"
