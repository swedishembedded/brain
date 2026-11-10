#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the reference tokenizer's ids for `crates/mirage`'s tokenizer golden.

Runs the reference implementation's `RefTokenizer` (the checkpoint publisher's
Python source, imported from a checkout you point at, never vendored here) over
a fixed set of multilingual strings against the tokenizer blob inside a
released `mirage.cact`, and writes `crates/mirage/tests/golden/tokenizer.json`:
the ids it encodes to and the text it decodes them back to. brain's Rust
tokenizer is gated on reproducing both exactly.

Usage:
  python3 tools/goldens/mirage_tokenizer_dump_reference.py \
      --reference-src /path/to/checkout-containing-the-cactus-package \
      --cact /path/to/Cactus-Compute/mirage/mirage.cact \
      --out crates/mirage/tests/golden/tokenizer.json
"""

import argparse
import hashlib
import json
import sys

CASES = [
    # en
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "A photo of two people riding horses on a beach at sunset.",
    # de
    "Größe der Straße, Übung macht den Meister.",
    "Die Rechnung über 1.234,56 Euro ist fällig.",
    # fr
    "Où est la bibliothèque municipale ? Élève à l'école.",
    "Le café est très chaud, mais délicieux.",
    # es
    "¿Dónde está la biblioteca? ¡Qué sorpresa!",
    "El niño comió piñones en la montaña.",
    # it
    "Perché la città è così bella di sera?",
    "Gli spaghetti alla carbonara sono deliziosi.",
    # nl
    "Het is een gezellige avond, ga je mee naar de bioscoop?",
    "De fiets staat achter het huis.",
    # pl
    "Zażółć gęślą jaźń, dzień dobry wszystkim.",
    "Gdzie jest najbliższy przystanek autobusowy?",
    # tags, spacing, bytes, numbers
    "<|en|>Hello<|de|>Hallo",
    "  leading and  double  spaces ",
    "line one\nline two\ttabbed",
    "STOP sign 50 km/h, exit 12B",
    "日本語 😀 mixed 中文",
    "<tool_call>{\"name\": \"f\"}</tool_call>",
    "naïve café résumé coöperate",
    "x",
    "",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-src", required=True)
    ap.add_argument("--cact", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.reference_src)
    from cactus.model.export import RefTokenizer  # noqa: E402

    tok = RefTokenizer.from_cact(args.cact)
    with open(args.cact, "rb") as f:
        cact_sha256 = hashlib.sha256(f.read()).hexdigest()
    cases = []
    for text in CASES:
        ids = tok.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids)})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cact_sha256": cact_sha256, "pieces": len(tok.pieces), "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {len(cases)} cases, {len(tok.pieces)} pieces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
