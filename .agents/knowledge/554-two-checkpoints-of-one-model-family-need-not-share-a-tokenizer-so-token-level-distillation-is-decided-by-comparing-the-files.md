<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 554. Two checkpoints of one model family need not share a tokenizer, so token-level distillation is decided by comparing the files

**Lesson:** a family name says nothing about the token ids. Before a teacher's per-token log-probabilities train a
student, compare the two `tokenizer.json` files on everything that decides an encoding: model settings, vocabulary,
merges, normalizer, pre-tokenizer and added tokens. A teacher's log-probability of id 1234 is a signal for the student's
id 1234 only when both tokenizers are identical. Otherwise it scores another piece of text, and the gradient is wrong
while every number stays finite.

## What happened

Qwen3-4B and Qwen3.8-27B look like one family with one tokenizer, and a vocabulary larger only by padding. A
token-level distillation run from the 27B into the 4B would have trained on the student's sampled ids with the
teacher's log-probabilities of the same ids, with nothing to flag it.

## Why

The two tokenizers differ in four of the six parts:

- vocab: 151,643 pieces vs 248,044;
- merges: 151,387 vs 247,587, and the files spell them differently (`["a", "b"]` vs `"a b"`);
- pre-tokenizer: the split pattern treats combining marks (`\p{M}`) as letters in the 27B's, not in the 4B's;
- added tokens: 26 vs 33, at other ids (151,643.. vs 248,044..).

The embedding row counts are 151,936 vs 248,320. A check of `vocab_size` alone would have caught this pair, but not a
pair that pads to the same size with a different pre-tokenizer.

## What to do

`data::tokenizer_identity::TokenizerIdentity::check` compares the parts above and is the only way to get the value.
`rl::objective::onpolicy_distill::DistillEstimator::admit` refuses token-level distillation (`teacher_logprobs`)
unless the value says the tokenizers are identical, with the typed `DistillRefusal`. For a pair that differs, the
sequence-level estimator is used instead: each model scores the completion's text in its own tokenization and chat
template (`teacher_logprob`). It is higher variance.

## Evidence (2026-11-22)

`crates/data/tests/it/tokenizer_identity_real.rs`, `qwen3_4b_and_qwen3_8_27b_do_not_share_a_tokenizer`, run
on the model store's Qwen3-4B and Qwen3.8-27B-FP8: it names exactly `vocab, merges, pre_tokenizer, added_tokens` and
the sizes above. Each checkpoint compares identical to itself.
