#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Generate the book's chronology appendix from the mainline history.

For each month: how many commits landed, which crates first appeared (with
the commit that introduced them) and which engineering findings were written
down. A crate that was renamed is dated by its earliest name, found by
following the file renames git records, so a model that began life under
another name keeps its true starting date. Dates are author dates: the
history is rebased to stay linear, which rewrites commit dates but not author
dates. Each crate is linked to the chapters that explain it.

The page covers history up to a pinned revision, recorded in the page
itself, so that unrelated commits do not make it stale: `--check` rebuilds
it at that pin and compares, and regenerating moves the pin to HEAD.

    scripts/book/chronology.py           # write the chronology page (pin = HEAD)
    scripts/book/chronology.py --check   # exit 1 if the page differs at its pin

Swedish Embedded AB implements solutions for turning engineering history into
durable, teachable knowledge. If your team needs expertise in machine
learning systems, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bookspec import BOOK, ROOT, load  # noqa: E402

OUT = BOOK / "history" / "chronology.md"
KNOWLEDGE_DIR = ROOT / ".agents" / "knowledge"
CRATE = re.compile(r"^crates/([^/]+)/")
KNOWLEDGE = re.compile(r"^\.agents/knowledge/(\d+[a-z]?)-[^/]+\.md$")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


PIN = re.compile(r"<!-- history-through: ([0-9a-f]{40}) -->")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def git_log(rev: str):
    raw = subprocess.run(
        ["git", "log", rev, "--reverse", "-M", "--name-status", "--format=\x1e%H\x1f%aI"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for chunk in raw.split("\x1e")[1:]:
        head, _, body = chunk.partition("\n")
        sha, date = head.split("\x1f")
        changes = [line.split("\t") for line in body.strip().splitlines() if line]
        yield sha, date, changes


def knowledge_title(kid: str) -> str | None:
    for p in KNOWLEDGE_DIR.glob(f"{kid}-*.md"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return re.sub(r"^#\s*\d+[a-z]?\.\s*", "", line).strip()
    return None


def build(rev: str) -> str:
    first_seen: dict[str, tuple[str, str]] = {}
    renamed_from: dict[str, str] = {}
    per_month = Counter()
    knowledge_by_month = defaultdict(list)
    for sha, date, changes in git_log(rev):
        month = date[:7]
        per_month[month] += 1
        rename_votes = Counter()
        for status, *paths in changes:
            path = paths[-1]
            if (m := CRATE.match(path)) and m.group(1) not in first_seen:
                first_seen[m.group(1)] = (date, sha)
            # A crate is renamed when its manifest moves; code extracted from
            # one crate into a new one moves files but leaves the manifest.
            if status.startswith("R") and len(paths) == 2 and all(
                    p.endswith("/Cargo.toml") for p in paths):
                a, b = CRATE.match(paths[0]), CRATE.match(paths[1])
                if a and b and a.group(1) != b.group(1):
                    rename_votes[(a.group(1), b.group(1))] += 1
            if status.startswith("A") and (k := KNOWLEDGE.match(path)):
                knowledge_by_month[month].append(k.group(1))
        for (old, new), _ in rename_votes.most_common():
            renamed_from.setdefault(new, old)

    def origin(crate):
        """The earliest (date, sha, name) along the crate's rename chain."""
        best, name, seen = first_seen[crate] + (crate,), crate, set()
        while name in renamed_from and name not in seen:
            seen.add(name)
            name = renamed_from[name]
            if name in first_seen and first_seen[name][0] < best[0]:
                best = first_seen[name] + (name,)
        return best

    book = load()
    explained = defaultdict(list)
    for ch in book.chapters():
        for c in ch.brain:
            explained[c].append(ch)
    current = sorted(p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").is_file())
    births = defaultdict(list)
    for c in current:
        if c in first_seen:
            date, sha, name = origin(c)
            births[date[:7]].append((date, c, sha, name))

    lines = [
        "<!-- SPDX-License-Identifier: CC-BY-4.0 -->",
        "<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->",
        "<!-- GENERATED by scripts/book/chronology.py from git history; do not edit. -->",
        f"<!-- history-through: {rev} -->",
        "",
        "# Chronology of brain",
        "",
        "This appendix is the engineering timeline the book's historical threads are drawn from. "
        "It is generated from the repository's own history: for every month, the number of commits, "
        "the crates that first appeared (dated by their earliest name when they were later renamed, "
        "and linked to the chapters that explain them) and the engineering findings that were written "
        "down. Dates are author dates; the history is kept linear by rebasing, which preserves them.",
        "",
        "The chapters teach in the order a reader needs the ideas. This page shows the order in which "
        "brain actually needed them, which is often very different: the engine began with a "
        "mixture-of-experts language model and a GPU kernel compiler, and only later grew the toy "
        "models a beginner would start with.",
        "",
    ]
    for month in sorted(per_month):
        y, m = month.split("-")
        lines += [f"## {MONTHS[int(m) - 1]} {y}", "", f"{per_month[month]} commits.", ""]
        if births[month]:
            lines += ["| First commit | Crate | Explained in |", "|---|---|---|"]
            for date, c, sha, name in sorted(births[month]):
                was = f" (as `{name}`)" if name != c else ""
                where = ", ".join(f"[{ch.title}](../{ch.rel})" if ch.written else ch.title
                                  for ch in explained[c]) or "not yet placed"
                lines.append(f"| {date[:10]}, commit `{sha[:9]}` | `{c}`{was} | {where} |")
            lines.append("")
        found = [(k, t) for k in knowledge_by_month[month] if (t := knowledge_title(k))]
        if found:
            lines += ["Findings recorded this month:", ""]
            # Finding titles quote the measurements of their day; marked as
            # reviewed historical numbers for the perf-number gate.
            lines += [f"- knowledge #{k.lstrip('0')}: {t} <!-- perf-number: dated historical finding -->"
                      for k, t in found]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        pin = PIN.search(current)
        if not pin or build(pin.group(1)) != current:
            print("docs/book/history/chronology.md is stale - run `make generate/book/chronology`",
                  file=sys.stderr)
            return 1
        print(f"chronology: up to date (through {pin.group(1)[:9]})")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(git("rev-parse", "HEAD").strip()), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
