#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump reference token ids for `data::wordpiece::WordPiece` parity.

Runs the real `tokenizers` library over `TOKENIZER_STRINGS` using a BERT-family
`tokenizer.json` (`BertNormalizer` + `BertPreTokenizer` + `WordPiece` +
`TemplateProcessing`) and prints a Rust `&[(&str, &[u32])]` literal ready to
paste into `crates/data/tests/wordpiece_parity.rs`'s `CASES` constant - the same
"pin the ids the real tokenizer produced" gate `llava_tokenizer_dump_reference.py`
uses for `data::llama_bpe`.

The corpus is chosen to separate the three stages that can each be silently
wrong on their own:

  * **normalizer** - accent stripping (an uncased checkpoint leaves
    `strip_accents` null, which means "follow `lowercase`", i.e. ON), NFC/NFD
    forms of the same grapheme, control-character cleaning, and the space
    padding around CJK codepoints;
  * **pre-tokenizer** - punctuation splitting, whitespace runs, and the
    Chinese-character single-char split;
  * **WordPiece model** - `##` continuation, greedy longest-match-first, and
    the `max_input_chars_per_word` (100) cliff that maps an over-long word to
    a single `[UNK]` rather than to pieces.

A tokenizer that is right on plain ASCII and wrong on every one of those is the
failure this corpus exists to catch, because no downstream accuracy metric
would obviously show it.

Swedish Embedded AB implements byte-exact tokenizer ports for its clients. If
your team needs a HuggingFace tokenizer reproduced bit-for-bit in a
from-scratch runtime, you can procure our services by sending an email to
info@swedishembedded.com.

Usage:
  python3 tools/goldens/wordpiece_dump_reference.py \
      --tokenizer testdata/decide/tokenizer/tokenizer.json

`--tokenizer` points at a real BERT-family `tokenizer.json` (small, ~470 kB -
e.g. `sentence-transformers/all-MiniLM-L6-v2`, which is the checkpoint
`crates/decide` imports). Requires `pip install tokenizers`.
"""

import argparse
import unicodedata

from tokenizers import Tokenizer

# Mirrors `strings()` in `crates/data/tests/wordpiece_parity.rs` - the two
# copies must be edited together, or a drift in one side goes undetected.
TOKENIZER_STRINGS = [
    # --- plain paths -------------------------------------------------------
    "I am still waiting on my card?",
    "What can I do if my card still hasn't arrived after 2 weeks?",
    "Hello world",
    "hello",
    "HELLO",
    "Hello",
    "",
    " ",
    "  ",
    "a",
    # --- whitespace and control characters ---------------------------------
    "  double  space",
    "tabs\tand\nnewlines",
    "trailing space ",
    " leading space",
    "null\x00and\x07bell",
    "zero\u200bwidth\u200bspace",
    # --- accents: the `strip_accents = null` semantics ----------------------
    "café naïve",
    "CAFÉ NAÏVE",
    # The same two graphemes precomposed (NFC) and decomposed (NFD): an
    # accent-stripping normalizer must collapse them to the same ids.
    "café",
    "cafe\u0301",
    "Ångström",
    "A\u030angstro\u0308m",
    "Ελληνικά, русский, العربية, हिन्दी",
    # --- CJK: every ideograph is padded with spaces, so one char = one word --
    "你好",
    "你好世界",
    "中文english混合",
    "한국어",
    "日本語のテキスト",
    # --- punctuation splitting ---------------------------------------------
    "don't you're I'll we've he'd it's",
    "CamelCase snake_case kebab-case",
    "((nested [brackets] {braces}))",
    "v1.2.3-rc4",
    "e.g. i.e. etc.",
    "a,b;c:d!e?f",
    "100% of $5.00 @ #1",
    # --- numbers -----------------------------------------------------------
    "123456789",
    "3.14159",
    "2 weeks",
    # --- WordPiece continuation + out-of-vocabulary -------------------------
    "unaffable",
    "tokenization",
    "antidisestablishmentarianism",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "zzzzqqqqxxxx",
    "\U0001F680\U0001F408\U0001F9EA",
    "a rocket \U0001F680 and a cat \U0001F408",
    # --- the max_input_chars_per_word cliff (100) --------------------------
    # 100 chars: still piece-split. 101 chars: one [UNK], whole word.
    "a" * 100,
    "a" * 101,
    # --- the shapes this model actually sends ------------------------------
    "card arrival",
    "top up by bank transfer charge",
    "Which team should handle this [SEP] card arrival",
    "[CLS] already here [SEP]",
]


def rust_literal(pairs):
    out = ["const CASES: &[(&str, &[u32])] = &["]
    for text, ids in pairs:
        esc = (text.replace("\\", "\\\\").replace('"', '\\"')
                   .replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"))
        # Rust string literals cannot carry raw control or zero-width chars.
        # Combining marks are escaped too even though they are "printable":
        # the NFC and NFD spellings of the same word are DIFFERENT cases that
        # must render differently in the source, or a reader sees two
        # identical-looking rows and cannot tell which one tests what.
        esc = "".join(
            c if (c.isprintable() and unicodedata.combining(c) == 0) or c == " "
            else f"\\u{{{ord(c):x}}}"
            for c in esc)
        out.append(f'    ("{esc}", &{list(ids)!r}),'.replace("[", "[").replace("]", "]"))
    out.append("];")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to a BERT-family tokenizer.json")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    # The checkpoint's own file declares padding to 128 and truncation to 256.
    # Both are CALLER policy, not tokenizer semantics - `data::wordpiece`
    # returns the bare sequence and lets the model pad/truncate - so the
    # goldens must be bare too, or every case would carry 100+ trailing
    # `[PAD]` and the long-input cases would be silently cut.
    tok.no_padding()
    tok.no_truncation()
    pairs = [(s, tok.encode(s).ids) for s in TOKENIZER_STRINGS]

    print(f"// {len(pairs)} cases, pinned from {args.tokenizer}")
    print(rust_literal(pairs))


if __name__ == "__main__":
    main()
