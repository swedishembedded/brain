#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Extract brain's engineering history as structured evidence for the book.

Every commit reachable from any ref (branches, remotes, tags) becomes one JSON
record: hash, parents, author and committer dates (kept apart, because a
rebase rewrites the second but not the first), subject, body, the files it
touched, the crates those files belong to, and the knowledge entries it added.
The book cites history by commit hash; this is the inventory those citations
are checked against and the raw material each chapter's historical thread is
mined from.

    scripts/book/history.py extract      # -> scratchpad/book/history.jsonl
    scripts/book/history.py summary      # per-crate first/last commit, counts

The output lives in the gitignored scratchpad: it is derived from git and is
regenerated on demand, never committed.

Swedish Embedded AB implements solutions for turning a codebase's history into
durable engineering knowledge for its clients. If your team needs expertise in
machine learning systems and their documentation, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scratchpad" / "book" / "history.jsonl"

# Record and field separators that cannot occur in a commit message.
RS, FS = "\x1e", "\x1f"
FORMAT = FS.join(["%H", "%P", "%aI", "%cI", "%s", "%b"]) + FS
CRATE = re.compile(r"^crates/([^/]+)/")
KNOWLEDGE = re.compile(r"^\.agents/knowledge/(\d+[a-z]?)-")


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def commits():
    """Yield one record per commit reachable from any ref, oldest first."""
    raw = git("log", "--all", "--reverse", "--date-order", "--name-status",
              f"--format={RS}{FORMAT}")
    for chunk in raw.split(RS)[1:]:
        head, _, files_part = chunk.rpartition(FS)
        sha, parents, adate, cdate, subject, body = head.split(FS, 5)
        files, added = [], []
        for line in files_part.strip().splitlines():
            status, *paths = line.split("\t")
            if not paths:
                continue
            path = paths[-1]
            files.append(path)
            if status.startswith("A"):
                added.append(path)
        crates = sorted({m.group(1) for f in files if (m := CRATE.match(f))})
        knowledge = sorted({m.group(1) for f in added if (m := KNOWLEDGE.match(f))})
        yield {
            "sha": sha,
            "parents": parents.split(),
            "author_date": adate,
            "commit_date": cdate,
            "subject": subject,
            "body": body.strip(),
            "files": files,
            "crates": crates,
            "knowledge_added": knowledge,
        }


def mainline() -> set[str]:
    return set(git("rev-list", "HEAD").split())


def extract() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    main = mainline()
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in commits():
            rec["mainline"] = rec["sha"] in main
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {OUT.relative_to(ROOT)}: {n} commits ({len(main)} on HEAD)")
    return 0


def load():
    if not OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} missing - run `scripts/book/history.py extract`")
    with OUT.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def summary() -> int:
    first, last, count = {}, {}, defaultdict(int)
    for rec in load():
        if not rec["mainline"]:
            continue
        for c in rec["crates"]:
            first.setdefault(c, rec)
            last[c] = rec
            count[c] += 1
    for c in sorted(first, key=lambda c: first[c]["author_date"]):
        f = first[c]
        print(f"{f['author_date'][:10]}  {count[c]:5d}  {c:20s} {f['sha'][:9]} {f['subject'][:70]}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"extract": extract, "summary": summary}.get(cmd, lambda: sys.exit(__doc__))())
