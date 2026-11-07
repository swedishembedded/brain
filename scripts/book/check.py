#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The book gate: docs/book is held to its specification and to its evidence.

Structure (from book.yaml):
  - chapter ids and concept ids are unique; a concept is introduced once;
  - every required concept is introduced earlier in reading order, so no
    chapter leans on an idea the reader has not met;
  - every crate under crates/ is explained by at least one chapter;
  - every markdown file under docs/book is a front/back page, a volume
    introduction or a specified chapter; a volume with a written chapter has
    its introduction.

Content (of every written page):
  - each chapter starts with `# <its title>` and has the recurring sections
    every chapter shares (a project chapter has its own set);
  - a chapter's Key terms define exactly the concepts it introduces, each
    marked with its concept id;
  - every `commit <hash>` citation resolves to a commit in this repository
    and every `knowledge #N` citation to an existing knowledge entry;
  - every `crates/...` path a page names exists;
  - every image a page embeds exists, is an SVG, and fits the size budget;
    every SVG under docs/book is embedded by some page;
  - no U+2014 (the repository forbids it), no machine-specific absolute path.

Generated content is checked by its own generators (worked.py, figures.py,
chronology.py, manifest.py) which `make book/check` runs alongside this.

    scripts/book/check.py

Swedish Embedded AB implements solutions for keeping large technical texts
correct as the systems they describe change. If your team needs expertise in
machine learning engineering and its documentation, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bookspec import BOOK, ROOT, load  # noqa: E402
from glossary import key_terms  # noqa: E402

CHAPTER_SECTIONS = ["The problem", "Before you begin", "Check your understanding", "Exercises",
                    "Summary", "Key terms", "Answers"]
PROJECT_SECTIONS = ["The goal", "Before you begin", "Milestones", "Acceptance tests", "Summary"]

SVG_BUDGET = 120 * 1024
COMMIT = re.compile(r"\bcommit `?([0-9a-f]{7,40})`?", re.I)
KNOWLEDGE = re.compile(r"\bknowledge #(\d+[a-z]?)\b")
CRATE_PATH = re.compile(r"`(crates/[A-Za-z0-9_./-]+)`")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
EM_DASH = chr(0x2014)  # the repository forbids it in prose
MACHINE_PATH = re.compile(r"(?<![\w.])/(?:data|home|Users|mnt)/[\w.-]+/")
KNOWLEDGE_DIR = ROOT / ".agents" / "knowledge"


def structure(book, errors):
    seen_chapters, introduced = set(), {}
    for ch in book.chapters():
        where = f"book.yaml: {ch.volume.id}/{ch.id}"
        if ch.id in seen_chapters:
            errors.append(f"{where}: duplicate chapter id")
        seen_chapters.add(ch.id)
        if not ch.question.endswith("?"):
            errors.append(f"{where}: question must be a question")
        # Requirements are checked against what is known before this
        # chapter plus what it has introduced up to that point.
        for c in ch.requires:
            if c not in introduced:
                errors.append(f"{where}: requires '{c}', which no earlier chapter introduces")
        for c in ch.introduces:
            if c in introduced:
                errors.append(f"{where}: '{c}' already introduced by {introduced[c]}")
            introduced[c] = ch.id

    crates = {p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").is_file()}
    explained = {c for ch in book.chapters() for c in ch.brain}
    for c in sorted(explained - crates):
        errors.append(f"book.yaml: names crate '{c}', which does not exist under crates/")
    for c in sorted(crates - explained):
        errors.append(f"book.yaml: crate '{c}' is explained by no chapter")
    return introduced


def pages(book, errors):
    """Every expected page that exists, as (path, chapter or None)."""
    expected = {}
    for rel in book.front + book.back:
        expected[rel] = None
    for v in book.volumes:
        written = [ch for ch in v.chapters if ch.written]
        if written:
            expected[v.intro_rel] = None
            if not (BOOK / v.intro_rel).is_file():
                errors.append(f"docs/book/{v.intro_rel}: missing (volume {v.id} has written chapters)")
        for ch in v.chapters:
            expected[ch.rel] = ch
    for rel in ["index.md"]:
        expected[rel] = None

    present = {p.relative_to(BOOK).as_posix() for p in BOOK.rglob("*.md")}
    for rel in sorted(present - set(expected)):
        errors.append(f"docs/book/{rel}: not a page of the book (not in book.yaml)")
    for rel in book.front + book.back + ["index.md"]:
        if rel not in present:
            errors.append(f"docs/book/{rel}: missing")
    return [(BOOK / rel, expected[rel]) for rel in sorted(present & set(expected))]


def commit_exists(sha, cache={}):
    """A cited commit must be in HEAD's history, not merely somewhere in the
    object store: a commit only on a side branch (or the pre-rebase twin of a
    mainline commit) is absent from a fresh clone of the main branch."""
    if sha not in cache:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=ROOT,
                           capture_output=True)
        cache[sha] = r.returncode == 0
    return cache[sha]


def knowledge_ids():
    return {m.group(1).lstrip("0") or "0" for p in KNOWLEDGE_DIR.glob("*.md")
            if (m := re.match(r"(\d+[a-z]?)-", p.name))}


BOLD = re.compile(r"\*\*([^*]+?)\*\*")


def slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.casefold()).strip("-")


def content(page_list, errors, owner=None):
    """owner maps each concept id to the chapter that introduces it."""
    owner = owner or {}
    kids = knowledge_ids()
    embedded = set()
    for path, ch in page_list:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if ch is not None:
            first = next((l for l in lines if l.startswith("# ")), "")
            if first != f"# {ch.title}":
                errors.append(f"{rel}: first heading must be '# {ch.title}'")
            headings = [l[3:].strip() for l in lines if l.startswith("## ")]
            required = PROJECT_SECTIONS if ch.id.startswith("project-") else CHAPTER_SECTIONS
            for sec in required:
                if sec not in headings:
                    errors.append(f"{rel}: missing section '## {sec}'")
            defined = {cid for _, _, cid in key_terms(text)}
            for cid in sorted(set(ch.introduces) - defined):
                errors.append(f"{rel}: Key terms lacks concept '{cid}' (introduced here per book.yaml)")
            for cid in sorted(defined - set(ch.introduces)):
                errors.append(f"{rel}: Key terms defines '{cid}', which this chapter does not introduce")
        in_key_terms = False
        for n, line in enumerate(lines, start=1):
            if line.startswith("## "):
                in_key_terms = line.strip() == "## Key terms"
            # A term is defined (in bold) only by the chapter that introduces
            # its concept; anywhere else it is a duplicate definition.
            if ch is not None and not in_key_terms:
                for term in BOLD.findall(line):
                    if term.endswith((".", ":")):
                        continue  # a run-in paragraph label, not a definition
                    cid = slug(term)
                    if cid in owner and owner[cid] != ch.id:
                        errors.append(f"{rel}:{n}: **{term}** is defined here, but '{cid}' is "
                                      f"introduced by chapter '{owner[cid]}'")
            if EM_DASH in line:
                errors.append(f"{rel}:{n}: U+2014 em dash (use a comma, colon or parentheses)")
            if MACHINE_PATH.search(line):
                errors.append(f"{rel}:{n}: machine-specific absolute path")
            for sha in COMMIT.findall(line):
                if not commit_exists(sha):
                    errors.append(f"{rel}:{n}: commit {sha} is not in the history of HEAD")
            for k in KNOWLEDGE.findall(line):
                if k.lstrip("0") not in kids:
                    errors.append(f"{rel}:{n}: knowledge #{k} does not exist")
            for p in CRATE_PATH.findall(line):
                if not (ROOT / p.rstrip("/")).exists():
                    errors.append(f"{rel}:{n}: {p} does not exist")
            for src in IMAGE.findall(line):
                if src.startswith(("http://", "https://")):
                    errors.append(f"{rel}:{n}: remote image {src} (figures are local SVGs)")
                    continue
                img = (path.parent / src).resolve()
                embedded.add(img)
                if not img.is_file():
                    errors.append(f"{rel}:{n}: image {src} does not exist")
                elif img.suffix != ".svg":
                    errors.append(f"{rel}:{n}: image {src} is not an SVG")
    for svg in sorted(BOOK.rglob("*.svg")):
        rel = svg.relative_to(ROOT).as_posix()
        if svg.resolve() not in embedded:
            errors.append(f"{rel}: not embedded by any page")
        if svg.stat().st_size > SVG_BUDGET:
            errors.append(f"{rel}: {svg.stat().st_size // 1024} KiB exceeds the {SVG_BUDGET // 1024} KiB budget")


def main() -> int:
    book = load()
    errors = []
    introduced = structure(book, errors)
    page_list = pages(book, errors)
    content(page_list, errors, introduced)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"book check: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    chapters = list(book.chapters())
    written = sum(ch.written for ch in chapters)
    print(f"book check: OK ({len(book.volumes)} volumes, {written}/{len(chapters)} chapters written, "
          f"{len(introduced)} concepts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
