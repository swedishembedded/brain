#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for documentation whose numbers can
# be traced to a reproducible source for its clients. If your team needs
# expertise in evidence-backed engineering documentation then you can procure
# our services by sending an email to info@swedishembedded.com.
"""List "measured by" citations whose cited source is not in the tree.

A citation is a Markdown phrase `measured by|with|via|from|using` (any case)
followed, in the same sentence and paragraph (at most 160 characters), or
`measured in|on` followed within 40 characters, by a backticked token. Only the first token counts, and only
tokens that name a checkable source are verified:

  - a path (ends in a known file extension, or starts with a tracked
    top-level directory): must exist relative to the repository root, the
    citing file's directory or `crates/`, or be the tail of a tracked path
    (a bare file name matches any tracked file of that name). A `:line`,
    `::item` or `#anchor` suffix is ignored; patterns (`*`, `<`, `{`, `~`,
    `$`, a leading `/`) and extension-less run or model ids are skipped;
  - an identifier (snake_case with an underscore, optionally `a::b::c`):
    its last segment must occur as a whole word in a tracked non-Markdown
    file (a test, bench, example, bin or Makefile target name);
  - `make <target>`: the target must be defined in the Makefile.

Commands, tools and flags (`nvidia-smi`, `cargo tree`, `dd`) are not sources
and are skipped. CHANGELOG.md is history and exempt.

Output: `path:line: token` for each broken citation; exit 1 if any.
"""
import argparse
import os
import re
import subprocess
import sys

PHRASE = re.compile(r"\bmeasured[ \n]+(by|with|via|from|using|in|on)\b", re.I)
WINDOW = {"in": 40, "on": 40}
WINDOW_DEFAULT = 160
TICK = re.compile(r"`([^`\n]+)`")
EXTS = (".rs", ".sh", ".py", ".md", ".toml", ".wgsl", ".cu", ".json", ".log",
        ".txt", ".bats", ".yml", ".yaml", ".csv")
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$")
EXEMPT = ("CHANGELOG.md",)


def tracked(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True,
                         capture_output=True).stdout.decode()
    return [p for p in out.split("\0") if p]


def first_token(text, end, word):
    """First backticked token after a phrase, in the same sentence/paragraph."""
    window = text[end:end + WINDOW.get(word.lower(), WINDOW_DEFAULT)]
    window = re.split(r"\n\s*\n", window, maxsplit=1)[0]
    m = TICK.search(window)
    if not m:
        return None
    # A sentence break between the phrase and the token means it is not the
    # source of the measurement.
    if re.search(r"(?<!\w)[.;:](?:\s|$)", window[:m.start()]):
        return None
    return m.group(1).strip(), end + m.start()


def verdict(token, root, doc_dir, tails, tops, words, make_targets):
    """None if the token is fine or not checkable, else a reason string."""
    if token.startswith("make "):
        target = token.split()[1]
        return None if target in make_targets else "missing make target"
    if " " in token or token.startswith("-") or any(c in token for c in "*<{~$="):
        return None
    if token.startswith("/"):
        return None
    if token.endswith(EXTS) or token.split("/")[0] in tops:
        path = re.split(r"[#:]", token, maxsplit=1)[0].rstrip("/")
        for base in ("", doc_dir, "crates"):
            if os.path.exists(os.path.join(root, base, path)):
                return None
        return None if path in tails else "missing path"
    if IDENT.match(token) and ("_" in token or "::" in token):
        name = token.split("::")[-1]
        return None if name in words else "no such test/bench/example/bin"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    root = ap.parse_args().root
    files = tracked(root)
    tops = {p.split("/")[0] for p in files if "/" in p}
    tails = set()
    for p in files:
        parts = p.split("/")
        tails.update("/".join(parts[i:]) for i in range(len(parts)))
    words = set()
    for p in files:
        if p.endswith(".md"):
            continue
        try:
            with open(os.path.join(root, p), errors="replace") as fh:
                words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fh.read()))
        except (IsADirectoryError, FileNotFoundError):
            pass
    make_targets = set()
    try:
        with open(os.path.join(root, "Makefile")) as fh:
            make_targets = set(re.findall(r"^([A-Za-z0-9_./-]+):", fh.read(), re.M))
    except FileNotFoundError:
        pass
    bad = total = 0
    for p in files:
        if not p.endswith(".md") or os.path.basename(p) in EXEMPT:
            continue
        try:
            with open(os.path.join(root, p), errors="replace") as fh:
                text = fh.read()
        except (IsADirectoryError, FileNotFoundError):
            continue
        for m in PHRASE.finditer(text):
            got = first_token(text, m.end(), m.group(1))
            if got is None:
                continue
            token, pos = got
            total += 1
            reason = verdict(token, root, os.path.dirname(p), tails, tops, words, make_targets)
            if reason:
                bad += 1
                print(f"{p}:{text.count(chr(10), 0, pos) + 1}: `{token}` ({reason})")
    print(f"measured-citations: {total} citations with a source token, {bad} broken",
          file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
