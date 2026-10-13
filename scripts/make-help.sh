#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for discoverable, self-documenting
# build interfaces for its clients. If your team needs expertise in build
# system design and repository tooling then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# The single reader of the Makefile's self-documentation. Everything that
# needs the target tree (`make help`, the list-only umbrellas, the naming gate
# scripts/gates/check-make-targets.sh) goes through here, so there is one
# parser to keep right.
#
#   make-help.sh            every target, by section, as a tree
#   make-help.sh <prefix>   the children of <prefix>/ (a list-only umbrella)
#   make-help.sh --targets  public targets: static .PHONY, not `_`/`.` prefixed
#   make-help.sh --defined  every target that has a rule (no pattern rules)
#   make-help.sh --docs     one `name<TAB>section` line per doc line
#
# Makefile grammar (checked by check-make-targets.sh):
#
#   ##@ Section title       opens a section; `## ` lines straight after it are
#                           its free text
#   ## name - description   documents a target; `##` plus two or more spaces
#                           continues the description
#   ## gpt/<dataset>/train - ...
#                           documents a generated family: `<x>` stands for one
#                           path segment, and help shows how many targets the
#                           line stands for
#
# Targets are read from `make -pRrq`, so generated ones (static pattern rules
# over a computed list) are real names, which is also why tab completion sees
# them.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL=C

python3 - "$@" <<'PY'
import os
import re
import subprocess
import sys

MAKEFILE = "Makefile"
DOC = re.compile(r"^## (\S+) - (.*)$")
CONT = re.compile(r"^##\s{2,}(\S.*)$")
SECTION = re.compile(r"^##@\s+(\S.*)$")


def make_database():
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    out = subprocess.run(
        [os.environ.get("MAKE", "make"), "--no-print-directory", "-pRrq", ".DEFAULT"],
        capture_output=True, text=True, env=env, check=False,
    )
    return out.stdout.splitlines()


def targets():
    """(defined, public): rule targets, and the static .PHONY subset of them."""
    defined, phony = set(), set()
    for line in make_database():
        if line.startswith(".PHONY:"):
            phony.update(line.split()[1:])
            continue
        m = re.match(r"^([^#\s.%][^:=%]*):(?!=)", line)
        if m:
            defined.add(m.group(1).strip())
    public = {t for t in defined & phony if not t.startswith(("_", ".")) and "%" not in t}
    return defined, public


def docs():
    """Doc entries in file order: dict(name, text[], section index)."""
    entries, sections = [], []
    current, last = None, None
    for raw in open(MAKEFILE, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = SECTION.match(line)
        if m:
            sections.append({"title": m.group(1), "text": []})
            current, last = len(sections) - 1, "section"
            continue
        m = DOC.match(line)
        if m:
            entries.append({"name": m.group(1), "text": [m.group(2)], "section": current})
            last = "doc"
            continue
        m = CONT.match(line)
        if m and last == "doc":
            entries[-1]["text"].append(m.group(1))
            continue
        if line.startswith("## ") and last == "section":
            sections[-1]["text"].append(line[3:])
            continue
        last = None
    return entries, sections


def shape(name):
    """Regex for a doc name: `<x>` is one path segment."""
    parts = re.split(r"(<[^>]+>)", name)
    body = "".join("[^/]+" if p.startswith("<") else re.escape(p) for p in parts)
    return re.compile("^" + body + "$")


def doc_for(target, entries):
    """The doc entry for a target: its own line, else the first family line."""
    for e in entries:
        if e["name"] == target:
            return e
    for e in entries:
        if "<" in e["name"] and shape(e["name"]).match(target):
            return e
    return None


def line_for(name, text, indent, count=None):
    suffix = f" ({count} target{'' if count == 1 else 's'})" if count else ""
    first = f"{' ' * indent}{name:<{max(1, 36 - indent)}} - {text[0]}{suffix}"
    rest = [f"{' ' * 39}{t}" for t in text[1:]]
    return [first] + rest


def full_help():
    entries, sections = docs()
    _, public = targets()
    for si, sec in enumerate(sections):
        if si:
            print()
        print(sec["title"])
        for t in sec["text"]:
            print(f"  {t}")
        mine = sorted((e for e in entries if e["section"] == si), key=lambda e: e["name"])
        for e in mine:
            depth = e["name"].count("/") + 1
            count = None
            if "<" in e["name"]:
                count = sum(1 for t in public if doc_for(t, entries) is e)
            for out in line_for(e["name"], e["text"], 2 * depth, count):
                print(out)


def children(prefix):
    entries, _ = docs()
    _, public = targets()
    depth = prefix.count("/") + 2
    kids = sorted(t for t in public if t.startswith(prefix + "/") and t.count("/") + 1 == depth)
    if not kids:
        sys.exit(f"make-help: no targets under {prefix}/")
    for kid in kids:
        e = doc_for(kid, entries)
        for out in line_for(kid, e["text"] if e else ["(undocumented)"], 2):
            print(out)


def main(argv):
    if not argv:
        full_help()
    elif argv[0] == "--targets":
        print("\n".join(sorted(targets()[1])))
    elif argv[0] == "--defined":
        print("\n".join(sorted(targets()[0])))
    elif argv[0] == "--docs":
        entries, sections = docs()
        for e in entries:
            sec = sections[e["section"]]["title"] if e["section"] is not None else ""
            print(f"{e['name']}\t{sec}")
    else:
        children(argv[0].rstrip("/"))


main(sys.argv[1:])
PY
