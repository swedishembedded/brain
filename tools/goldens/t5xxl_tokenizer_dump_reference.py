# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity harnesses for neural network
# inference engines for its clients. If your team needs expertise in validating
# a from-scratch tokenizer or model port against its upstream reference, you can
# procure our services by sending an email to info@swedishembedded.com.
"""Dump the golden token ids for `crates/data/tests/t5xxl_precompiled_parity.rs`.

The reference is HuggingFace `tokenizers`' own `Tokenizer.from_file`, i.e. the
exact library `T5TokenizerFast`/`AutoTokenizer` wrap, reading the exact
`tokenizer.json` brain reads. That file's normalizer is a three-step
`Sequence` - SentencePiece's `Precompiled` charsmap, then `Strip{right}`, then
`" {2,}" -> "▁"` - and the corpus below is chosen to make each step, and
the interaction between them, observable in the ids.

This prints a Rust table on stdout; paste it into the test. The table is
checked in rather than fetched because `/testdata` is gitignored, so a golden
written there would not survive a clone, and a parity test nobody can run is
not a gate.

Usage:
    python3 tools/goldens/t5xxl_tokenizer_dump_reference.py \\
        $BRAIN_FLUX1_DIR/tokenizer_2/tokenizer.json
"""

import sys

from tokenizers import Tokenizer

# Mirrors `CASES` in crates/data/tests/t5xxl_precompiled_parity.rs. The two
# copies must be edited together: a golden that carries its own inputs cannot
# notice that the two sides drifted apart.
CASES = [
    # Plain ASCII, including the prompt the FLUX.1 pipeline is driven with.
    "a photo of a person standing in a park",
    "a photo of a person standing in a park, highly detailed, 8k, cinematic lighting",
    "hello world",
    "",
    # Multiple spaces: the `" {2,}" -> "▁"` step, and its interaction with
    # the Metaspace pre-tokenizer that then prepends/splits on that same
    # character. A SINGLE space must survive as a space.
    "double  spaces   here",
    "one space only",
    "a  b   c    d",
    # Trailing/leading whitespace: only the RIGHT is stripped, and only after
    # the charsmap has folded tabs/newlines/NBSP down to spaces.
    "  leading and trailing  ",
    "tabs\tand\nnewlines",
    "trailing whitespace   ",
    "nbsp and　ideographic space",
    # Ligatures and compatibility forms: the charsmap's whole reason to exist.
    "ligatures ﬁ and ﬂ and ﬃ",
    "fullwidth ＡＢＣ １２３",
    "roman numeral ⅠⅡ and ①",
    "½ ¼ ⅓ fractions",
    "superscript ²³ and ⁵",
    "circled ⒶⒷ parenthesized ⑴",
    "CJK compat ㍻ ㌀",
    # Accents, composed and decomposed, plus a Turkish dotted capital.
    "accents café naïve résumé",
    "composed é vs decomposed é",
    "İstanbul Türkiye",
    # Grapheme clusters where the 6-byte threshold in `normalize_string`
    # decides the answer: halfwidth katakana + dakuten is exactly 6 bytes and
    # therefore takes the PER-CHARACTER path, while the 5-byte `ẛ̣`
    # takes the whole-grapheme path.
    "ｶﾞ halfwidth katakana",
    "ガ vs ガ katakana",
    "ẛ̣ dot above below",
    # Non-Latin scripts and astral-plane input.
    "Ελληνικά, русский, 中文",
    "emoji \U0001f600 and family \U0001f468‍\U0001f469‍\U0001f466",
    # Everything at once.
    "mixed ﬁne café  Ａ  Ⅰ end",
]


def rust_escape(s: str) -> str:
    """Escape `s` as a Rust string literal body, ASCII-only.

    Every non-ASCII character becomes `\\u{...}` rather than being written
    literally: half of this corpus is pairs that LOOK identical in an editor
    (composed vs decomposed `é`, precomposed `ガ` vs `カ`+U+3099, a ZWJ
    hiding between two emoji), and a test whose inputs a reader cannot tell
    apart is not checking what it claims to.
    """
    out = []
    for c in s:
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif c == "\r":
            out.append("\\r")
        elif " " <= c <= "~":
            out.append(c)
        else:
            out.append(f"\\u{{{ord(c):x}}}")
    return "".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    tok = Tokenizer.from_file(sys.argv[1])

    print("/// Mirrors `CASES` in tools/goldens/t5xxl_tokenizer_dump_reference.py.")
    print("#[rustfmt::skip]")
    print("const CASES: &[(&str, &[u32])] = &[")
    for text in CASES:
        ids = tok.encode(text).ids
        norm = tok.normalizer.normalize_str(text)
        print(f'    // -> "{rust_escape(norm)}"')
        print(f'    ("{rust_escape(text)}", &{ids!r}),')
    print("];")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
