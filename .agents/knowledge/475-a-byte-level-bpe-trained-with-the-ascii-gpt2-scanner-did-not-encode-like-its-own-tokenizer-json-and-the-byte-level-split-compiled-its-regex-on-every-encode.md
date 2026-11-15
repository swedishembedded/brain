<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 475. A byte-level BPE trained with the ASCII GPT-2 scanner did not encode like its own tokenizer.json, and the ByteLevel split compiled its regex on every encode

**Lesson:** A tokenizer that is trained in one place and served from a `tokenizer.json` must split text with the one
pre-tokenizer the file declares, not with a look-alike. And a pattern a hot path applies is compiled once per process,
never inside the call.

## What happened

`data::bytebpe::ByteBpe` trained and encoded with `bpe::pre_tokenize`, the hand-written GPT-2 scanner, which treats
`\p{L}` and `\p{N}` as ASCII only (stated in its doc: it serves ASCII corpora). Exporting the vocabulary as a
`tokenizer.json` with the GPT-2 `ByteLevel` pre-tokenizer and loading it with `QwenBpe` gave different ids for any
non-ASCII word: `café,` is one punctuation-and-letter run to the scanner (`é` is not a letter there) and `caf` `é` `,`
to the Unicode pattern, so the scanner had learned merges such as `é,` that the file can never apply. A model trained
on ids from one and served with the other would see text it was never trained on.

Separately, `hf_pretok::PreTokenizer::ByteLevel { use_regex: true }` compiled the GPT-2 `fancy_regex` pattern inside
`apply`, once per `encode` call.

## What changed

- `ByteBpe` trains and encodes with `hf_pretok::gpt2_pieces`, the `ByteLevel` split exactly as a `tokenizer.json`
  declaring it applies it; `ByteBpe::to_hf_tokenizer_json` exports the vocabulary with declared special tokens, and a
  test holds `QwenBpe` over the exported file to the trainer's own ids on mixed-script text.
- The GPT-2 pattern is compiled once per process (`OnceLock`).
- The trainer keeps pair counts incrementally (a merge recounts only the words holding its pair) and is held to the
  textbook trainer's merge order by a test; it never learns a piece that spells a reserved special token or a piece it
  already has (a `tokenizer.json` vocabulary is keyed by string, so two ids for one string cannot be written).
