<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 92. A "known, documented gap" is a bug with good manners - the documentation buys time, it does not reduce the debt

`data::unigram` shipped for a year rejecting SentencePiece's `Precompiled`
normalizer by name, and did it *well*: the module doc explained the omission,
`parse_normalizer` returned an error naming the type rather than silently
skipping the step, and a test asserted that exact rejection. Every one of
those was the right call at the time - the alternative, quietly dropping a
normalizer, changes ids for a whole class of inputs with nothing downstream
able to tell. The gap was scoped out honestly because umT5, the only Unigram
checkpoint in the workspace, does not use a charsmap.

Then FLUX.1-dev's real `tokenizer_2` arrived (lesson #91) and the gap was
simply a build break: T5-XXL's normalizer is a three-step `Sequence` and
**all three** steps were unimplemented - `Precompiled`, `Strip`, and a
`" {2,}"` `Replace` whose content is `U+2581`, not the space the existing
`CollapseSpaces` rule hardcoded. The documented gap named ONE of the three.
The other two were invisible because nothing had ever read a file containing
them, and the honest error message for the first one masked the fact that
fixing it alone would not have been enough.

Two things worth carrying forward.

**Port the reference, do not reconstruct it.** The charsmap is a Darts
double-array trie whose four accessors are pure bit math
(`(u >> 10) << ((u & (1 << 9)) >> 6)` and friends), and whose `normalize_string`
tries a whole grapheme cluster only when it is under **6 bytes** and otherwise
falls back per character. That threshold is not an optimization - it changes
the output: `U+FF76 U+FF9E` is exactly 6 bytes, so it decomposes to
`U+30AB U+3099` instead of folding to `U+30AC`, while the 5-byte
`U+1E9B U+0323` folds whole to `U+1E61`. HuggingFace's own `spm_precompiled`
carries a comment daring the reader to "simplify" it and then check XNLI.
Both cases are now in `crates/data/tests/t5xxl_precompiled_parity.rs`
precisely because they are the only lines in an English prompt where a
plausible-looking simplification diverges.

**A synthetic fixture can be validated against the reference too.** The
checkpoint-free unit tests build a tiny double-array by hand (`build_charsmap`),
which is only trustworthy if the reference agrees it is well-formed - so those
same bytes were fed to HuggingFace's `tokenizers.normalizers.Precompiled` and
matched on 11/11 strings including both sides of the 6-byte threshold. That
turns "my reader agrees with my writer" (which proves nothing) into "both
agree with the reference", and it costs one throwaway script. Do this whenever
a test fabricates data in a format someone else defines.
