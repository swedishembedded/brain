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
  - each chapter starts with `# <its title>` and ends with the reference
    apparatus a reader looks up (exercises, key terms, answers); the body above it is the author's own,
    and no heading or run-in label is recycled from chapter to chapter;
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

# The reference apparatus at the end of a chapter, the only headings every chapter shares.
CHAPTER_SECTIONS = ["Exercises", "Key terms", "Answers"]
# A project chapter names its own body sections (build steps, tests, extensions) like any
# other body heading; of the shared apparatus it must carry the Answers to its recall questions.
PROJECT_SECTIONS = ["Answers"]
# Apparatus a chapter may also carry under the same names; every other heading or
# run-in label (`**Label.**`) belongs to one chapter's subject, so the same one
# in more than RECYCLED_LIMIT chapters is a template showing through.
APPARATUS = {"Exercises", "Key terms", "Answers",
             "Summary", "Check your understanding", "References", "Further reading"}
RECYCLED_LIMIT = 3
# Labels that announce a template slot instead of a subject; purpose and
# prerequisites belong in the prose.
TEMPLATE_LABELS = {"the problem", "before you begin", "what this is for", "what it is for", "purpose",
                   "why it matters", "why it matters later", "making it work", "the goal"}
# The same label repeated inside one chapter is the same template at a smaller scale.
REPEAT_LIMIT = 2
HEADING = re.compile(r"^#{2,4} (.+?)\s*$")
# A run-in label: `**Label.**`, or the italic form `*Label: ...*` that opens a line.
RUN_IN = re.compile(r"^\*\*([^*]+?)[.:]\*\*|^\*([A-Z][^*:.]{1,40})[:.]")

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

    # The crates git knows about: the book describes committed code, and a
    # crate another session is still building in the working tree is not yet
    # part of brain.
    tracked = subprocess.run(["git", "ls-files", "--", "crates/*/Cargo.toml"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    crates = {Path(t).parent.name for t in tracked}
    explained = {c for ch in book.chapters() for c in ch.brain}
    for c in sorted(explained - crates):
        errors.append(f"book.yaml: names crate '{c}', which does not exist under crates/")
    for c, reason in sorted(book.omitted.items()):
        if c not in crates:
            errors.append(f"book.yaml: omits crate '{c}', which does not exist under crates/")
        if c in explained:
            errors.append(f"book.yaml: crate '{c}' is both omitted and explained by a chapter")
        if not str(reason).strip():
            errors.append(f"book.yaml: omitted crate '{c}' has no reason")
    for c in sorted(crates - explained - set(book.omitted)):
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
            else:
                # A volume introduction is one page of prose: no section headings,
                # so no set of labels ("why it comes here") can recur volume to volume.
                for n, line in enumerate((BOOK / v.intro_rel).read_text(encoding="utf-8").splitlines(), 1):
                    if line.startswith("## "):
                        errors.append(f"docs/book/{v.intro_rel}:{n}: a volume introduction has no section headings ({line!r})")
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


def recycled(page_list, errors):
    """Headings and run-in labels repeated across chapters: a template, not writing."""
    where: dict[str, set[str]] = {}
    for path, ch in page_list:
        if ch is not None:
            _labels(path, path.relative_to(ROOT).as_posix(), errors, where)
    for label, files in sorted(where.items()):
        if len(files) > RECYCLED_LIMIT:
            errors.append(f"'{label}' heads a section in {len(files)} chapters (limit {RECYCLED_LIMIT}): "
                          "give each its own subject-specific wording")


def stock_sentences(page_list, errors):
    """A sentence of six or more words repeated verbatim in more than RECYCLED_LIMIT chapters
    is a formula, not writing (a stock lead-in to the recall questions was the case that
    prompted this). Only body prose counts: the end apparatus, generated blocks, code and
    tables are left out."""
    where: dict[str, set[str]] = {}
    for path, ch in page_list:
        if ch is None:
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"<!-- worked:.*?<!-- /worked -->", "", text, flags=re.S)
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        body = []
        for line in text.splitlines():
            if line.startswith("## ") and line[3:].strip() in APPARATUS:
                break
            if not line.startswith(("|", "#", "!", "<")):
                body.append(line)
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(body)):
            sentence = sentence.strip()
            if len(sentence.split()) >= 6 and not re.match(r"\d+\.\s", sentence):
                where.setdefault(sentence, set()).add(path.relative_to(ROOT).as_posix())
    for sentence, files in sorted(where.items()):
        if len(files) > RECYCLED_LIMIT:
            errors.append(f"the sentence {sentence!r} appears in {len(files)} chapters (limit "
                          f"{RECYCLED_LIMIT}): write it in each chapter's own words")


def _labels(path, rel, errors, where):
    """Collect one chapter's headings and run-in labels; report its template slots and repeats."""
    local: dict[str, int] = {}
    in_fence = in_apparatus = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            in_apparatus = line[3:].strip() in APPARATUS
        # Inside the end apparatus, labels (such as the groups of answers) are apparatus too.
        m = None if in_fence or in_apparatus else (HEADING.match(line) or RUN_IN.match(line))
        if not m:
            continue
        text = next(g for g in m.groups() if g)
        label = text.strip().lower()
        if re.match(r"(figure|table) \d", label):
            continue  # a numbered caption, not a section label
        if label in TEMPLATE_LABELS:
            errors.append(f"{rel}: '{text.strip()}' is a template slot, not a subject: "
                          "say it in the prose or name what the section is about")
        where.setdefault(label, set()).add(rel)
        local[label] = local.get(label, 0) + 1
    for label, n in sorted(local.items()):
        if n > REPEAT_LIMIT:
            errors.append(f"{rel}: '{label}' heads {n} sections (limit {REPEAT_LIMIT}): "
                          "say what each one is about")


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
    recycled(page_list, errors)
    stock_sentences(page_list, errors)
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
