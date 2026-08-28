#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump reference token ids for `data::llama_bpe::LlamaBpe` parity.

LLaVA-1.5's decoder half (Vicuna-1.5-13B) uses LLaMA-2's SentencePiece
byte-fallback BPE tokenizer unchanged. This script runs the real
`tokenizers` library's `Tokenizer.encode` (the same engine
`LlamaTokenizerFast` wraps) over `TOKENIZER_STRINGS` and prints a Rust
`&[(&str, &[u32])]` literal ready to paste into
`crates/data/tests/llama_bpe_parity.rs`'s `CASES` constant - the same
"pin the ids the real tokenizer produced" gate `clip_dump_reference.py`
uses for `data::clip_bpe`.

Swedish Embedded AB implements solutions for byte-exact tokenizer ports
for its clients. If your team needs expertise in reproducing a
HuggingFace tokenizer bit-for-bit in a from-scratch runtime, you can
procure our services by sending an email to info@swedishembedded.com.

Usage:
  python3 tools/goldens/llava_tokenizer_dump_reference.py \
      --tokenizer testdata/llava/tokenizer/tokenizer.json

`--tokenizer` points at a real LLaMA-2/Vicuna `tokenizer.json` (fetched
separately - it is small, ~1.8 MB, unlike the 13B checkpoint itself; e.g.
`NousResearch/Llama-2-13b-hf` or any Vicuna-1.5 mirror, since both ship
the identical tokenizer). Requires the `tokenizers` package
(`pip install tokenizers`).
"""

import argparse
import json

# Mirrors `strings()` in `crates/data/tests/llama_bpe_parity.rs` - the two
# copies must be edited together, or a drift in one side goes undetected.
TOKENIZER_STRINGS = [
    "Describe this image and its style in a very detailed manner.",
    "Hello world",
    " Hello",
    "hello",
    "",
    " ",
    "  ",
    "你好",
    "a",
    "  double  space",
    "tabs\tand\nnewlines",
    "Ελληνικά, русский, العربية, हिन्दी, 한국어, 日本語のテキスト",
    "a rocket \U0001F680 and a cat \U0001F408 with a beaker \U0001F9EA",
    "CamelCase snake_case kebab-case",
    "123456789",
    "v1.2.3-rc4",
    "<image>\nWhat is unusual about this image?",
    "USER: <image>\nDescribe this image and its style in a very detailed manner. ASSISTANT:",
    "café naïve",
    "don't you're I'll we've he'd it's",
    "((nested [brackets] {braces}))",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to a real tokenizer.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)

    print("// Pinned against the real `tokenizers` library on a LLaMA-2/Vicuna-1.5")
    print("// tokenizer.json (regenerate: tools/goldens/llava_tokenizer_dump_reference.py).")
    print("const CASES: &[(&str, &[u32])] = &[")
    for s in TOKENIZER_STRINGS:
        ids = tok.encode(s).ids
        print(f"    ({json.dumps(s, ensure_ascii=False)}, &{ids}),")
    print("];")


if __name__ == "__main__":
    main()
