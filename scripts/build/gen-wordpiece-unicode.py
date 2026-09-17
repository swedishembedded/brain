#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Regenerate `crates/data/src/wordpiece_unicode.rs` from Python's own UCD.

`BertNormalizer`'s accent-stripping stage is "NFD, then drop every nonspacing
mark". Rust has no canonical decomposition in std and this workspace has no
`unicode-normalization` in its lock, so the *composed* half of that operation
ships as a generated table.

What is tabulated is the COMPOSITION of both steps - `NFD(c)` with category-Mn
characters already removed - not raw NFD. That is what collapses 4,028 entries
into three flat arrays with 2,087 output scalars total, and it is also the only
thing the tokenizer ever asks for. Hangul syllables (11,172 of them) are
excluded: their decomposition is arithmetic, so the Rust side computes it.

Canonical reordering is deliberately not modelled. Reordering only permutes
characters with a nonzero combining class among themselves, and every such
character in a canonical decomposition of a cased/accented letter is a
nonspacing mark that this table has already deleted - so the surviving sequence
is order-stable. `verify` proves that claim against the real `unicodedata` over
the entire code space rather than asserting it.

usage:
  scripts/build/gen-wordpiece-unicode.py            # write the table
  scripts/build/gen-wordpiece-unicode.py --verify   # check the checked-in copy
"""
import argparse
import os
import sys
import unicodedata

HANGUL_BASE, HANGUL_COUNT = 0xAC00, 11172
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "data", "src", "wordpiece_unicode.rs")

HEADER = """// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Canonical decomposition minus nonspacing marks - GENERATED, do not edit.
//!
//! Regenerate with `scripts/build/gen-wordpiece-unicode.py` (and verify a
//! checked-in copy with `--verify`, which `make wordpiece-table/check` runs).
//!
//! This is the table half of [`crate::wordpiece`]'s accent-stripping stage:
//! for every scalar whose `NFD(c)`-with-Mn-removed differs from `c` itself, the
//! surviving scalars. An entry with an empty run is a character that vanishes
//! entirely (it is a nonspacing mark, or decomposes only into them).
//!
//! Hangul syllables are NOT here: their decomposition is arithmetic and
//! [`crate::wordpiece`] computes it directly, which keeps 11,172 rows out of
//! this file.
//!
//! Flat parallel arrays rather than `&[(u32, &[u32])]`: one `&[u32]` literal
//! per row would be one static per row.

/// Scalars with a non-identity decomposition, ascending - the binary-search key.
pub(crate) const KEYS: &[u32] = &[
"""


def build():
    keys, offs, vals = [], [], []
    for cp in range(0x110000):
        if HANGUL_BASE <= cp < HANGUL_BASE + HANGUL_COUNT:
            continue
        c = chr(cp)
        out = "".join(ch for ch in unicodedata.normalize("NFD", c) if unicodedata.category(ch) != "Mn")
        if out == c:
            continue
        keys.append(cp)
        offs.append((len(vals) << 8) | len(out))
        vals.extend(ord(x) for x in out)
    return keys, offs, vals


def render(keys, offs, vals):
    text = HEADER
    for i in range(0, len(keys), 12):
        text += "    " + ", ".join(f"0x{v:x}" for v in keys[i:i + 12]) + ",\n"
    text += "];\n\n"
    text += "/// Packed `(offset << 8) | len` into [`VALS`], parallel to [`KEYS`].\n"
    text += "pub(crate) const OFFS: &[u32] = &[\n"
    for i in range(0, len(offs), 12):
        text += "    " + ", ".join(f"0x{v:x}" for v in offs[i:i + 12]) + ",\n"
    text += "];\n\n"
    text += "/// The surviving scalars, concatenated.\n"
    text += "pub(crate) const VALS: &[u32] = &[\n"
    for i in range(0, len(vals), 12):
        text += "    " + ", ".join(f"0x{v:x}" for v in vals[i:i + 12]) + ",\n"
    text += "];\n"
    return text


def decompose(cp, keys, offs, vals):
    """The Rust side's algorithm, re-implemented here to verify the table."""
    if HANGUL_BASE <= cp < HANGUL_BASE + HANGUL_COUNT:
        s = cp - HANGUL_BASE
        l, v, t = 0x1100 + s // 588, 0x1161 + (s % 588) // 28, s % 28
        return [l, v] + ([0x11A7 + t] if t else [])
    try:
        i = keys.index(cp)
    except ValueError:
        return [cp]
    off, ln = offs[i] >> 8, offs[i] & 0xFF
    return vals[off:off + ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    keys, offs, vals = build()

    # Prove the table (plus the arithmetic Hangul path) reproduces the real
    # operation over the WHOLE code space, including canonical reordering.
    bad = 0
    for cp in range(0x110000):
        want = [ord(ch) for ch in unicodedata.normalize("NFD", chr(cp)) if unicodedata.category(ch) != "Mn"]
        got = decompose(cp, keys, offs, vals)
        if want != got:
            bad += 1
            if bad <= 5:
                print(f"MISMATCH U+{cp:04X}: want {want} got {got}", file=sys.stderr)
    if bad:
        sys.exit(f"{bad} scalars disagree with unicodedata")

    text = render(keys, offs, vals)
    path = os.path.normpath(OUT)
    if args.verify:
        cur = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if cur != text:
            sys.exit(f"{path} is stale - re-run scripts/build/gen-wordpiece-unicode.py")
        print(f"ok: {path} matches ({len(keys)} entries, {len(vals)} scalars)")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}: {len(keys)} entries, {len(vals)} scalars, unicodedata {unicodedata.unidata_version}")


if __name__ == "__main__":
    main()
