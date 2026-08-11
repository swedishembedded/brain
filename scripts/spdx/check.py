#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Pre-commit gate: every SPDX-relevant file has exactly one
"SPDX-License-Identifier: Apache-2.0" line, immediately followed by the
project's copyright line.

File selection (which extensions/basenames count, comment style, where the
header goes relative to a shebang/`#version`/etc.) lives in rules.py.

Usage:
    scripts/spdx/check.py <file> [<file> ...]     # validate (pre-commit passes these)
    scripts/spdx/check.py --fix <file> [<file> ...]  # also insert missing headers

Exit status: 0 if every relevant file is compliant, 1 otherwise (with every
violation printed, not just the first).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import COPYRIGHT_RE, SPDX_ID, classify, find_spdx_lines, is_binary, process


def check_file(path: Path):
    """Return a list of violation strings for `path` (empty if compliant)."""
    rel = path.as_posix().encode()
    style = classify(rel)
    if style is None:
        return []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [f"could not read: {exc}"]
    if is_binary(data):
        return []

    # find_spdx_lines only matches an actual "<comment-prefix> SPDX-License-
    # Identifier: <id>" declaration in this file's comment style — not any
    # line that merely mentions the string (e.g. in a docstring or error
    # message), which is what this very file and rules.py both do.
    spdx_lines = find_spdx_lines(data, style)

    if not spdx_lines:
        return ["missing 'SPDX-License-Identifier' line"]
    if len(spdx_lines) > 1:
        at = ", ".join(str(i + 1) for i, _ in spdx_lines)
        return [f"more than one SPDX-License-Identifier line (lines {at}) — exactly one is allowed"]

    violations = []
    line_no, ident_bytes = spdx_lines[0]
    ident = ident_bytes.decode("utf-8", "replace")
    lines = data.split(b"\n")
    if ident != SPDX_ID:
        violations.append(
            f"line {line_no + 1}: SPDX-License-Identifier must be exactly "
            f"'{SPDX_ID}', found {ident!r}"
        )

    next_line = lines[line_no + 1].decode("utf-8", "replace") if line_no + 1 < len(lines) else ""
    if not COPYRIGHT_RE.search(next_line):
        violations.append(
            f"line {line_no + 2}: expected the copyright line immediately after the "
            f"SPDX-License-Identifier line (got {next_line!r})"
        )
    return violations


def main(argv):
    fix = "--fix" in argv
    paths = [Path(a) for a in argv if a != "--fix"]
    if not paths:
        print("usage: check.py [--fix] <file> [<file> ...]", file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        if not path.is_file():
            continue
        violations = check_file(path)
        if not violations:
            continue
        if fix and violations == ["missing 'SPDX-License-Identifier' line"]:
            rel = path.as_posix().encode()
            data = path.read_bytes()
            new_data = process(rel, data)
            if new_data is not None:
                path.write_bytes(new_data)
                print(f"fixed: {path}")
                continue
        failed = True
        print(f"{path}:")
        for v in violations:
            print(f"  {v}")

    if failed:
        print(
            "\nEvery Rust/C/Python/shell/Makefile/WGSL/... source file must start "
            "with:\n"
            "  // SPDX-License-Identifier: Apache-2.0   (or '#'/'/* */' per language)\n"
            "  // Copyright (c) <year> Martin Schröder <info@swedishembedded.com>\n"
            "Run 'scripts/spdx/check.py --fix <file>...' to add a missing header "
            "automatically (files with a wrong/duplicate SPDX line need a manual look)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
