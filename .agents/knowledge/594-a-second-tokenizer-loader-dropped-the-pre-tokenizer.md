<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 594. A second tokenizer loader dropped the pre-tokenizer and trained 7.9% of a corpus on non-reference ids

**Lesson:** One tokenizer has one loader. A hand-rolled twin of `QwenBpe::from_dir` in the TTS prompt code built its
`tokenizer.json` without Qwen2's pre-tokenizer split, so whitespace runs and punctuation merged across word
boundaries; because training data and serving both used that twin, the error was invisible to every self-consistency
check and only a comparison against the reference (`transformers` Qwen2TokenizerFast) could see it.

## What happened
`qwen3tts::prompt::load_tokenizer` assembled the split-file layout (`vocab.json` + `merges.txt` +
`tokenizer_config.json`) itself and declared no `pre_tokenizer`. Against the reference, on the Sten-TTS v2 corpus
(assistant-framed transcripts, the framing the shards store):

- shards vs reference: 15306 of 194769 train records and 185 of 2151 validation records differed (15491 of 196920,
  7.87%; 8.09M reference tokens in total); every one was identical to the old serving encoding, so the shards were
  built by that loader and training and serving agreed with each other, not with the reference.
- 300 evaluation sentences (fleurs, code_switching, numbers): 0 differ; they contain no whitespace runs or `--`.
- after the fix: serving matches the reference on all 197220 texts.

Trigger text: double spaces (spoken-number transcripts like `fyra  två  fem`), `--`, runs of quotes or spaces.
Plain single-spaced Swedish prose is unaffected, which is why listening tests did not catch it.

## Why
Without the split, byte-level BPE merges (`Ġ Ġ` -> `ĠĠ`) apply across word boundaries; the reference splits first, so
`a  b` is `a`, ` `, ` b`.

## What to do
`load_tokenizer` is now `QwenBpe::from_dir`. `tests/it/tokenizer.rs` pins a self-contained split-file checkpoint
(red before the fix: `[97,256,98]` vs `[97,32,32,98]`) and two real-vocabulary transcripts with reference ids.
The deployed model was trained on the old ids for the affected 7.9% of records and is served with the corrected ids
after this change: a checkpoint trained on the old loader sees reference ids at inference for those inputs. A shard
rebuild (`text_ids` re-encoded from the stored `text`) is needed to retrain on reference ids; the shards' tokenizer
digest in `producers` does not change when the loader does, so do not rely on it to detect this.
