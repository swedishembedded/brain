<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 343. The qwen35moe head backward grows steeply with the vocabulary on the CPU backend

Symptom: a toy qwen35moe fine-tune through the generic full-parameter and preference trainers took about 265 s for the
first step on the CPU backend (vocabulary 151936, 64-token row) and 1 to 2 s per step elsewhere; on a card both sdk tests
finish in about 20 s.

Measured (CPU backend, one 32-token row, tiny config, `Gpu::new_cpu`): backward 0.06 s at vocabulary 29, 0.15 s at 2048,
8.8 s at 20000 (forward and the optimizer step stay at 40 to 50 ms and 19 ms). A tenfold vocabulary costs about 57 times the
backward, so the time is not the arithmetic of the head.

Probable cause (read from the source, not isolated by a profile; not fixed): `qwen35moe` computes its head with the full `n x vocab` logits and the naive `matmul_dw` kernel for the
head and embedding gradients, where `qwen3` and `qwen35` use the shared chunked head (`model::head_ce`, register-tiled
`matmul_dw_reg`). The CPU backend's per-dispatch cost makes the naive kernel's access pattern dominant at a real vocabulary.

Consequence: the two qwen35moe training tests of `sdk/tests/chat_finetune_qwen35moe.rs` are `#[ignore]` (run on a card, `--ignored`);
the tokenizer fixture needs the real 151936 vocabulary, so a smaller toy vocabulary is not an option there.

What to do: move qwen35moe onto `HeadCe` as the other two decoders have (its own head tests in `qwen35moe/tests` cover the
logits), then drop the `#[ignore]`s.
