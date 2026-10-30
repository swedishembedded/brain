<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 384. The Gated DeltaNet prefill runs 3x faster on bf16 operands with an fp32 state, and the training backward is not bound by its MMA instructions

Question: knowledge 382 left the GDN prep and recurrence kernels as the largest prefill cost (3.8 ms a layer at 8K tokens, tensor pipe 18 % busy
in the recurrence, issue slots 48 % busy in the prep). Both emulate fp32 with three TF32 MMAs per product. What does a bf16-operand / fp32-state
form cost in accuracy, how fast is it, and does the training path take it too?

Result (gpu1 = GH200, `gdn_chunk_fwd_train_tc -- --ignored production_shape`: 48 heads, 8192 tokens, chunk 64, token-major serving forward):

| | exact (TF32 x 3) | bf16 operands, fp32 accumulation and state |
|---|---|---|
| prep kernel (ncu) | 1.66 ms, 594 M warp instructions | 0.85 ms, 215 M |
| recurrence (ncu) | 2.97 ms, 792 M | 0.77 ms, 159 M |
| a layer's forward (wall) | 3.83 ms (3.50 with the new solve) | 1.20 ms (1.33 before the prep's shared memory was trimmed to fit three blocks an SM) |

What the bf16 pair is (`gdn_chunk_prep_bf16.cu`, `gdn_chunk_loop_bf16.cu`, helpers in `gdn_chunk_common.cuh`):

1. **Round once, where the matrix is made.** The prep writes `w`, `intra`, `query * scale * exp_g_cs` and `key * exp(g_last - g_cs)` as bf16, so the
   recurrence multiplies them with no convert or split arithmetic and loads half the bytes (its four value-column blocks of a head re-read the same
   tiles through L2). `beta` and `exp_g_cs` are folded into the columns of the small `T` operand instead of scaling the 64 x 128 tiles; `u` stays fp32.
2. **The state is fp32 in registers.** The warp that owns a 16 x 16 piece of the 128 x 32 state slice keeps it in registers; the bf16 copy in shared
   memory is only the B operand of the next chunk's two products. The rounding of a product is 2^-9 of its operands and does not compound.
3. **The triangular solve stays fp32** (the one step whose conditioning depends on near-parallel keys).
4. **A first register-prefetch version spilled** (128 registers, 1 GB of local-memory loads) and ran slower than the version without prefetch (2.9 vs
   2.6 ms): the fp32 w/query/key tiles of the next chunk are 72 registers. With bf16 inputs the same prefetch is 32 + 24 registers, and unrolling the
   128-deep k loop only by 2 keeps the fragments from filling the file; the kernel is then 0.77 ms with no spills.
5. **The row-by-row forward substitution was 40 % of the prep's instructions.** 62 barrier-separated steps, one multiply-add each for a 64-thread
   column. Panels of eight rows (all contributions of the finished panels in one pass of fused multiply-adds with the k sum split over four thread
   groups, then the in-panel substitution in registers) need two barriers a panel: the exact prep got 0.33 ms a layer faster (3.83 -> 3.50 ms). It is
   not bit-identical to the old order (the sum order changed); every gdn test holds its TF32-split bound.

Accuracy: the bf16 form's output against the exact form's is 3.1e-3 .. 4.0e-3 relative rms (1.6 - 2.1 steps of 2^-9) on full, even-short and odd-short
chunks (`the_bf16_prefill_form_matches_the_exact_form_within_a_few_bf16_steps`, bound 4 steps). `qwen35_fp8_parity` on the real FP8 27B (8 prompts x
1800 tokens, 32 steps, reference always the exact GDN): exact GDN 98.83 % greedy agreement / mean KL 1.549e-3, bf16 GDN 99.22 % / 1.192e-3, every
flip at a near-tie margin (4 prompts x 1500: 100 % / 9.7e-4 against 96.88 % / 1.12e-3 with four flips at margins 0.02 - 0.05): the FP8 linears dominate
the difference from the bf16-decoded reference.

Selection (`gpu-core/src/gdn_prefill_tune.rs`, slot `gdn_prefill`, candidates `exact` and `bf16`): on placing a model with 128-wide GDN heads the device
runs a synthetic 48-head 2048-token prefill through both forms on a throwaway handle (so its buffers are not charged to the model's), admits the bf16
form within 4 bf16 steps of the exact output and picks it when it is more than 3 % faster; no device name is consulted, a device below compute
capability 8.0 or without either kernel keeps the exact form, `--kernel gdn_prefill=exact|bf16` pins it. Two traps paid for: the tuner left
`set_kernel_timing(true)` on, which put per-launch events into every decode step (single stream 53 -> 21 tok/s); and `Gpu` charges every
allocation to the handle for its lifetime, so measuring on the model's own handle moved `charged_bytes` of a one-layer shard from 26 to 100+ MiB.

Served (FP8 27B, int8 KV, same client, exact -> bf16): c = 1 / 4 / 16 / 64 output tok/s 52.6 / 213.9 / 390.1 / 535.8 -> 53.1 / 210.0 / 400.5 / 573.8,
31K TTFT 4.87 -> 4.49 s (final build, same client: 53.0 / 207.9 / 394.6 / 568.2 tok/s and 4.61 s; the node is shared, runs differ by a few per cent). The GDN pair is now 254 ms of the 3.67 s of kernels in a 31K prefill (nsys): the FP8 GEMM is 2.13 s (0.78 PFLOPS), the
wgmma attention 0.51 s, the elementwise kernels around the GDN layers (`mul` 140 ms, `quant_epilogue` 142, `quant_e4m3_128` 113, `add_rms_quant` 83,
`rmsnorm_rows` 78, `gdn_conv_silu_tok` 64, `sigmoid` 37) another 0.7 s, and the wall clock of the request is 0.8 s over the kernel sum.

Training (W8K NF4 probe, 64 layers, 8192 tokens, seed 7, gpu1, warm steps): the checkpointed forward runs the no-save form (`gdn_chunk_fwd_tokens`), so
it takes the bf16 pair too; 937 -> 981 tokens/s (a second pair of runs on the final build: 946 -> 988), forward 2.11 -> 1.92 s, losses 13.4643 / 13.4072 / 13.3342 / 13.2478 against 13.4644 / 13.4073 /
13.3341 / 13.2479 (equal to the fourth digit; the gradients come from the exact train-form recompute). The train-form forward (`gdn_chunk_loop_train_tc`
128 ms, `gdn_chunk_prep_tok_train_tc` 71 ms of a 5.4 s device step) and the backward (`gdn_chunk_bwd_post_tc` 247 ms, `gdn_bwd_sweep_tc` 189 ms) stay exact:
the backward kernels with their operands rounded to bf16 (a drop-in `warp_mma_acc`) kept every gradient within 1e-3 .. 4e-3 of the exact one (max-norm
relative, `d_key` and `d_beta` worst) but ran only 9 % (post, 1.29 -> 1.17 ms) and 5 % (sweep, 0.98 -> 0.93 ms) faster: they are bound by the slab
staging, the barriers and the shared-memory traffic around the MMAs, not by the MMA instructions, so the 6x cut in tensor work does not reach the clock.
The backward needs the restructuring the forward got (bf16 tiles staged once, `ldmatrix` fragments, registers for the accumulators), not a bf16 flag;
the gain is bounded by their 8 % share of the step.

Not done: the attention single-split at 1K / 4K (0.070 / 0.201 ms against 0.088 / 0.219 ms with four splits): the effective split count is decided
on the device because a recorded tape is replayed at changing contexts, and the host would need the round's context in `PrefillShape`; the saving is
about 0.3 ms of a 130 ms mixed iteration.
