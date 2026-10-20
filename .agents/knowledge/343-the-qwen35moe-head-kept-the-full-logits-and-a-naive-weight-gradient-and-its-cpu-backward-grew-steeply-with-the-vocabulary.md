<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 343. The qwen35moe head kept the full logits and a naive weight gradient; its CPU backward grew steeply with the vocabulary

Symptom: a toy qwen35moe fine-tune through the generic full-parameter and preference trainers took about 265 s for its first
step on the CPU backend (vocabulary 151936, 64-token row), against 1 to 2 s per step on the default backend.

Measured (CPU backend, one 32-token row, tiny config): backward 0.06 s at vocabulary 29, 0.15 s at 2048, 8.8 s at 20000, while
forward and the optimizer step stayed flat. A tenfold vocabulary cost about 57 times the backward.

Cause: unlike qwen3 and qwen35, qwen35moe built the full `n x vocab` logits and ran the head and embedding gradients through
the naive `matmul_dw` kernel instead of the shared chunked head (`model::head_ce`, register-tiled `matmul_dw_reg`).

Fix: a training build that owns the head now builds a `HeadCe` (forward, backward and `logits_all` chunked; the weighted loss,
logit scale and per-token log-probabilities go through the head like qwen35's). The toy fine-tune tests of
`sdk/tests/chat_finetune_qwen35moe.rs` run on the CPU backend in seconds, and conformance C8 now runs for qwen35moe (the
unchunked head holds the whole row, the chunked one 64 rows, with equal loss, log-probabilities and gradients).
