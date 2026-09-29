<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 178. A tokenizer guessed from one regex feature tokenizes a different model

`QwenBpe` did not read a `tokenizer.json`'s `pre_tokenizer`. It
reconstructed one. A hand-written scanner implemented Qwen's split pattern,
and the file was consulted for exactly one fact: the `K` in `\p{N}{1,K}`,
found by searching the JSON text. A bare `ByteLevel` switched to a second
scanner. Every other declared step was dropped without a word:

- **deepseek-llm, deepseek-math and deepseek-coder-v1.5**: a five-way
  `Split` sequence plus `Digits` (separate CJK and Latin runs, unbounded
  digit runs).
- **deepseek-coder v1**: a four-way `Split` sequence.
- **R1-Distill-Qwen**: an `NFC` normalizer.
- **R1-Distill-Llama**: `model.ignore_merges`.

All of these tokenized as Qwen2 would. The ids were valid vocabulary entries,
so nothing failed; the model simply saw text split in ways it was never
trained on.

`data::hf_pretok` now reads the pipeline the file declares:

- Normalizers: `NFC`, `Sequence`.
- Pre-tokenizers: `Split` with all five behaviours and `invert`, `Digits`,
  `ByteLevel` with `use_regex`/`add_prefix_space`, `Sequence`.
- Anything else is refused by name at load.

The split semantics were taken from the pip `tokenizers` release by probing
it directly, not from the upstream source (whose `main` is a rewrite). A
GGUF names its pre-tokenizer instead of declaring it; `for_gguf` maps each
name to llama.cpp's split sequence and refuses unknown ones. Previously
every name got the Qwen2 split, `deepseek-v3` with a three-digit cap.

The same parity run exposed a decode rule. The `ByteLevel` decoder maps a
token back through the byte map when every one of its chars is a byte-map
char, and emits it as raw UTF-8 otherwise, whether or not it is an added
token. coder v1 registers `ü` as an added token, so `tokenizers` decodes
"über" to `"\u{FFFD}ber"`, and brain now does too.

`crates/data/tests/deepseek_tokenizer_parity.rs` pins all 13 DeepSeek text
tokenizers, encode and decode, to the library's output on a corpus chosen to
separate the four shapes.
