# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Which files get an SPDX + copyright header, in what comment style, and where.

This is the single source of truth for "is this file SPDX-relevant" — used by
scripts/spdx/check.py (the pre-commit validator). Keep both in sync by only
ever editing the rules here.
"""
import re

SPDX_ID = "Apache-2.0"
COPYRIGHT_TEXT = "Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>"
# Validation is looser than insertion: a file committed in a later year should
# still pass with that year (or a "2026-2028" range), not stay pinned to 2026.
COPYRIGHT_RE = re.compile(
    r"Copyright \(c\) \d{4}(-\d{4})? Martin Schröder <info@swedishembedded\.com>"
)

# A line must BE one of these (modulo surrounding whitespace) to count as an
# actual SPDX declaration — not just any line that mentions the string, which
# this very file and check.py's docstring/messages both do. Matching on the
# substring alone would flag this rule file and the checker as duplicates.
_SPDX_LINE_RES = {
    "slash": re.compile(rb"^\s*//\s*SPDX-License-Identifier:\s*(\S+)\s*$"),
    "hash": re.compile(rb"^\s*#\s*SPDX-License-Identifier:\s*(\S+)\s*$"),
    "cblock": re.compile(rb"^\s*/\*\s*SPDX-License-Identifier:\s*(\S+)\s*\*/\s*$"),
}

# Extensions/basenames that get a header, grouped by comment style.
SLASH_EXTS = {
    "rs", "c", "h", "hpp", "hh", "hxx", "cpp", "cc", "cxx",
    "wgsl", "glsl", "vert", "frag", "comp", "geom", "tesc", "tese",
    "ts", "tsx", "mjs", "cjs", "js", "jsx", "proto",
}
HASH_EXTS = {"py", "sh", "bash", "bats", "mk"}
HASH_BASENAMES = {"Makefile", "makefile", "GNUmakefile"}
CBLOCK_EXTS = {"css", "scss", "less"}

# Directory components that, if present anywhere in the path, exempt the
# file even if its extension would otherwise match (vendored/generated code).
EXCLUDED_DIR_PARTS = {"vendor", "third_party", "node_modules", "generated", ".git"}

_CODING_RE = re.compile(rb"^#.*coding[:=]\s*[-\w.]+")


def classify(path_bytes: bytes):
    """Return 'slash' | 'hash' | 'cblock' | None (skip) for a repo path."""
    path = path_bytes.decode("utf-8", "surrogateescape")
    parts = path.split("/")
    if any(part in EXCLUDED_DIR_PARTS for part in parts):
        return None
    base = parts[-1]
    if base in HASH_BASENAMES:
        return "hash"
    if "." not in base:
        return None
    ext = base.rsplit(".", 1)[-1].lower()
    if ext in SLASH_EXTS:
        return "slash"
    if ext in HASH_EXTS:
        return "hash"
    if ext in CBLOCK_EXTS:
        return "cblock"
    return None


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def find_spdx_lines(data: bytes, style: str):
    """Return [(line_index, identifier_bytes), ...] for lines that are an
    actual SPDX-License-Identifier declaration in `style`'s comment syntax
    (see _SPDX_LINE_RES) — not just any line containing the substring."""
    pattern = _SPDX_LINE_RES[style]
    out = []
    for i, line in enumerate(data.split(b"\n")):
        m = pattern.match(line)
        if m:
            out.append((i, m.group(1)))
    return out


def already_has_spdx(data: bytes, style: str) -> bool:
    return bool(find_spdx_lines(data, style))


def header_lines(style: str):
    if style == "slash":
        return [
            f"// SPDX-License-Identifier: {SPDX_ID}".encode(),
            f"// {COPYRIGHT_TEXT}".encode(),
        ]
    if style == "hash":
        return [
            f"# SPDX-License-Identifier: {SPDX_ID}".encode(),
            f"# {COPYRIGHT_TEXT}".encode(),
        ]
    if style == "cblock":
        return [
            f"/* SPDX-License-Identifier: {SPDX_ID} */".encode(),
            f"/* {COPYRIGHT_TEXT} */".encode(),
        ]
    raise ValueError(f"unknown style {style!r}")


def _line_end(data: bytes, start: int) -> int:
    idx = data.find(b"\n", start)
    return (idx + 1) if idx != -1 else len(data)


def find_insert_offset(data: bytes) -> int:
    """Byte offset the header block must be inserted after.

    Keeps a leading shebang (optionally + a Python coding declaration) or a
    GLSL `#version` pragma as the physical first line(s) of the file, since
    both are positionally significant.
    """
    e1 = _line_end(data, 0)
    line1 = data[0:e1]
    if line1.startswith(b"#!") and not line1.startswith(b"#!["):
        # Real shebangs are "#!/..." or "#! ...". "#![...]" is a Rust inner
        # attribute (e.g. #![no_std]), not a shebang - must not be treated
        # as one, or the header would land after it instead of before.
        offset = e1
        e2 = _line_end(data, offset)
        line2 = data[offset:e2]
        if _CODING_RE.match(line2):
            offset = e2
        return offset
    if line1.startswith(b"#version"):
        return e1
    return 0


def insert_header(data: bytes, style: str) -> bytes:
    """Return `data` with the header inserted, or unchanged if already present."""
    if already_has_spdx(data, style):
        return data
    hdr = b"\n".join(header_lines(style)) + b"\n"
    off = find_insert_offset(data)
    rest = data[off:]
    if rest and not rest.startswith(b"\n"):
        hdr += b"\n"
    return data[:off] + hdr + rest


def process(path_bytes: bytes, data: bytes):
    """Top-level entry point: returns new data, or None if unchanged/skipped."""
    style = classify(path_bytes)
    if style is None:
        return None
    if is_binary(data):
        return None
    if already_has_spdx(data, style):
        return None
    new_data = insert_header(data, style)
    if new_data == data:
        return None
    return new_data
