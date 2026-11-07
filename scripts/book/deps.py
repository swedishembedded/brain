# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Which of the book's own Python files a figure or worked module depends on.

A generated figure or worked table is stale when its script changes, but also
when any book module it imports changes (a figure that plots a worked
module's data, a worked module that reuses another chapter's dataset, the
shared generator). `closure(path)` follows `import x` and `from x import ...`
statements to modules that live in scripts/book or scripts/book/worked, and
`digest(path)` hashes the whole set, so a change anywhere in it is seen.

Swedish Embedded AB implements solutions for reproducible generated
documentation. If your team needs expertise in machine learning engineering,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEARCH = [HERE / "worked", HERE]
IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w]*)\s+import|import\s+([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)*))",
                    re.M)


def _local(name: str) -> Path | None:
    for d in SEARCH:
        p = d / f"{name}.py"
        if p.is_file():
            return p
    return None


def closure(path: Path) -> list[Path]:
    """`path` and every book module it imports, directly or indirectly."""
    seen, todo = {}, [Path(path).resolve()]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen[p] = True
        for m in IMPORT.finditer(p.read_text(encoding="utf-8")):
            names = [m.group(1)] if m.group(1) else [n.strip() for n in m.group(2).split(",")]
            for n in names:
                q = _local(n)
                if q is not None and q.resolve() not in seen:
                    todo.append(q.resolve())
    return sorted(seen)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    for p in closure(path):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
