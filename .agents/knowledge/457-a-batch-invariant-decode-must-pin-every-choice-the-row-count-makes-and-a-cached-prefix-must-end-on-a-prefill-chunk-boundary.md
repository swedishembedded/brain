<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 457. A batch-invariant decode must pin every choice the row count makes, and a cached prefix must end on a prefill chunk boundary

**Lesson:** A decode row's result depends on its co-batch through every selection keyed by the step's row count, not
only through the one kernel that visibly switches; and it depends on the prefix cache through the row count of the
prefill chunk that computed each cached row. Pinning the kernels alone leaves the second source, which on Qwen3 changed
the very first sampled token of a repeated request.

## What happened

Making Qwen3 serving batch-invariant (`--exact-decode`) on a Tesla P40, the row-count-dependent choices of one decode
step were:

- the int8 linears' measured GEMV-or-tile choice per power-of-two row bucket (`tune_i8`), the two rounding differently;
- the library tensor-core GEMM from `library_min_rows` rows (absent on Pascal, present on every newer card);
- the int8 attention's split count, sized from the batch (`int8_decode_splits(cfg, bsz, cap)`), which moves a long
  sequence's 128-key chunks between splits and so changes the online-softmax combine;
- the f32-KV attention's form: the fused split kernel at one row, the native batched kernel from two, the triad
  elsewhere;
- on a device with int8 tensor cores, the native int8 GEMV's own switch to its tensor-core block from five rows.

A synthetic engine shaped to reach them (32 KV heads, a 1100-token probe, 16 rows) failed the bit-identity gate in
the default serving and passed once each was pinned; removing the split pin or the int8 variant pin alone made it fail
again (mutation checks). On the real Qwen3-0.6B the default serving's probe left its first run at token 0 at every
weight tier once its prompt was served from the cache: the cold prefill computed the last cached rows inside a
512-row chunk, the warm one computed the suffix as a chunk of a few rows, on another GEMM.

## What to do

- Declare row-count invariance on the candidates (`select::KernelVariant::row_count_invariant`), let an exact caller
  choose among those once per weight shape (`row_invariant_candidates`, `AutoTuner::resolve_row_invariant`), and cut
  a step into groups the chosen kernel keeps its arithmetic over (`exact_decode_group_rows`).
- Choose every other row-dependent variant as for one row and size the dispatch for the real rows
  (`paged_scores_variant_chosen_at`, the single-sequence split count).
- Reuse cached prompt blocks only up to a boundary of the cold prompt's prefill chunks, and publish only blocks of
  whole chunks: then a cached prefix holds what a cold prefill computes and the rest of the prompt runs as the cold
  prompt's remaining chunks.
- Refuse, by name, what cannot be pinned (a per-row adapter bank, latent sparse attention, a tier whose products are
  chosen inside the ops facade).

## Evidence

Gates: `crates/qwen3/tests/it/engine_exact_decode.rs` (synthetic at f32, bf16 and int8 weights with int8 KV and at f32
KV, the real Qwen3-0.6B when the model store holds it); `select::tests::the_row_invariant_*`. Numbers and the
throughput cost: `.agents/roadmap/serving-runtime.md`, "Batch-invariant decode".
