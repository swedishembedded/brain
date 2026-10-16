<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 314. A tensor-core chunk forward takes the Gated DeltaNet forward from about 3 s to under 1 s of device time, and the 27B step is then bound by the host

Question: what does a fused tensor-core forward for the chunked Gated DeltaNet (work package
T1 of the throughput plan) buy on the 64-layer, 8192-token training step, and does it keep the
numbers? Every figure is **VERIFIED-BY-RUN** unless it says **DERIVED**. Workload: `qwen35_train_probe
--tokens 8192 --layers 64 --rank 16 --seed 7 --gdn-segment 2048 --mlp-rows 2048 --bf16-residuals on`
(random weights, the real 27B shape and 248,320-token head), card idle (`other_contexts_start=0`,
`other_contexts_end=0`) under the lease `gpu --exclusive` wrapper. "Before" is the binary of
`465448ae` (origin/main), "after" the tree of `22aa0d99`.

## What changed

Three native kernels in `kernels-cuda`, selected per call by the provider and only when the caller
sets `GdnFastIds::tensor_core` (qwen35 does; qwen35moe and horizon keep the bit-identical path):

- `gdn_chunk_prep_tc` replaces the fourteen whole-tensor dispatches of steps 1-9 and the
  intra-chunk scores (two row scales, cumulative decay, decay mask, Gram product, strict-lower
  mask, unit-triangular solve, three products, the score product and its mask) by one launch per
  (head, chunk): the four matrix products run as three TF32 `mma.sync.m16n8k8` each (a TF32 head
  and remainder per operand, fp32 accumulate), the solve stays fp32 in shared memory (never the
  Neumann series, see the note at step 7 of `model::gdn`), the cumulative decay is the same serial
  fp32 sum, so `g_cs` is bit-identical. The training form also writes what the backward reads.
- `gdn_chunk_prep_tok_tc` / `_tok_train_tc` read the mixer's token-major tensors: the query/key L2
  normalisation, the 3:1 head repeat and the five chunk-major permutes happen in the block's loads
  (`gdn_mixer_stream_fwd` drops two `l2norm_scale`, two `kv_expand` and five `gdn_layout_permute`
  per layer pass), and the chunk-major query, key, value and beta the recurrence and the backward
  read are written from the same tiles.
- `gdn_chunk_loop_tc` / `_train_tc` run the across-chunk recurrence on tensor cores, four blocks of
  32 value columns per head (192 blocks, all resident), fp32 state and accumulation.

Fallback is the existing code path, selected by the same query: a device below compute capability
8.0, the CPU and Vulkan backends (no capability), `BRAIN_NO_PROVIDER=cuda`, a head width other than
128, a chunk over 64 rows, or a backend that declines the launch all run the dispatch sequence of
before, unchanged. A probe with the CUDA provider disabled reproduces the loss of the baseline
(13.4597 at step 3 of the 4-layer probe, 4,684 tokens/s, no tensor-core kernel in the profile).

## Numerical evidence

| check | result |
|---|---|
| `gpu-core/tests/gdn_chunk_fwd_tc.rs`, kernel against an f64 transcription: 64-row, 5-row and 1-row chunks, nearly parallel keys (cosine about 0.97), inference and training outputs, and the token-major form including the chunk-major tensors it writes | largest error relative to the tensor's magnitude 1.4e-6 (bounds 2e-5, 5e-4 for the near-parallel case) |
| `model/tests/gdn_chunk_fwd_train_tc.rs`: forward, state history, decay history and all six `gdn_chunk_bwd` gradients against the portable path at 128-wide heads, 64/32/8-row chunks | within 1.3e-6 (bound 1e-4); the launch census shows 13 fewer steps (the prefix is one launch) |
| `model/tests/gdn_mixer_tc.rs`: the whole layer (conv, norms, gating) forward and every input and weight gradient, training and inference form, 128 and 320 tokens | within 3.6e-6 of the portable layer |
| the 64-layer probe, three steps | losses 13.4607 / 13.4012 / 13.3244, equal to the fourth digit to the baseline's |
| existing gates | `gdn_chunk_fwd(_fast)`, `gdn_chunk_fwd_train_fast` (still bit-identical with the flag off), `gdn_chunk_bwd`, `gdn_mixer_{equivalence,segmented,stream,decode_batched}`, qwen35 `train_tier_trajectory`, `lora_{gdn_segments,nf4_base,recompute,bf16_residuals}` green. The tiny gradcheck models use 16-wide heads, so they never reach the new kernels; `qwen35_lora_on_a_bf16_base_analytic_grads_match_finite_differences` fails on blocks.0.mlp.up (bf16 GEMM noise against finite differences) with and without this change |

## Device time (64 layers, NF4, one timed step; before / after)

| component (`scripts/perf/group-kernels.py`) | before ms (dispatches) | after ms (dispatches) |
|---|---:|---:|
| GDN chain (native GDN kernels included) | 5,088 (21,696) | 3,280 (12,480) |
| library GEMM | 2,656 (21,896) | 2,651 (21,896) |
| element-wise glue | 1,635 (12,874) | 1,299 (9,994) |
| attention element-wise | 5 (96) | 5 (96) |
| RMSNorm | 178 (1,250) | 179 (1,250) |
| head, optimizer, misc | 1,201 (19,339) | 1,195 (18,763) |
| **device total** | **10,762 (77,151)** | **8,609 (64,479)** |

(The grouping script billed the `native:brain_gdn_*` kernels to misc until this change; its GDN
row now matches the plan's 5.13 s. The bf16 base: GDN chain 5,081 to 3,281 ms, device 10,454 to
8,309 ms, 74,191 to 61,519 dispatches.)

By kernel kind (ms, calls before / after): `gdn_chunk_loop_f32` + `_train_f32` 1,255 (576) to
`gdn_chunk_loop_tc` + `_train_tc` 600 (576); the prefix and layout glue it replaces
(`l2norm_scale` 345, `kv_expand` 189, `scale_row` 314 to 31, `gdn_ut_fwd` 96, mask and decay mask 97,
`gdn_layout_permute` 515 to 225, forward share of `bmm_tiled` about 423 of 1,015) to
`gdn_chunk_prep_tok_tc` 191 + `_tok_train_tc` 107. Summing the forward-side kinds gives about
3.0 s before and about 0.9 s after (DERIVED from those rows; the plan's budget is 0.8 s). The
backward half of the chain (sweep 541 ms, backward `bmm_tiled` and glue) is untouched: that is T2.

## Tokens per second, and why the wall figure understates the change

| base | before | after | device s before / after |
|---|---:|---:|---|
| NF4 probe | 385 (21.3 s/step) | 440 (18.6 s/step), two runs 439-443 | 10.76 / 8.61 |
| bf16 probe | 407 | 463 | 10.45 / 8.31 |

These were taken on a node whose load average was 10-50 (other lanes' builds and jobs): the host
thread was the bound in every run (`cpu_s` within 0.5 s of `step_s`; about 280 us per launch where
knowledge 312 measured 35 us), so the wall rate moves with the launch count (77,151 to 64,479), not
with device time, and none of these rates is comparable to the 611-635 of knowledge 311/312, taken
on a quiet host. With the host at that earlier cost the step would be device time plus the
2.7 s gap of 311: about 11.3 s, or 725 tokens/s (DERIVED, not run). A re-run on a quiet node is the
measurement to quote for the plan's 1,000 tokens/s row; the GDN chain at 3.28 s against the plan's 1.50 s (T1 and T2 together) and the device total at 8.6 s against its 5.56 s budget. The largest remaining kinds are the
library GEMMs (2.66 s, the card's practical rate), then the sweep (0.54 s) and the backward
`bmm_tiled` (0.59 s), which belong to T2 (the fused backward), and the backward-side layout glue (`gdn_layout_permute` 225 ms, `l2norm_scale_dx`).

## Process note

Lanes that share one cargo target directory (`lane-brain`) while building the same crates from
different worktrees appear to overwrite each other's compiled artifacts (the unit hash did not
differ between worktrees, and cargo's freshness test compares modification times): a build here
linked an older `brain-model` or `brain-gpu-core` than the source on disk, which showed as errors
for fields and functions that exist in the source and could as easily have been a wrong binary.
Touching the sources cured it each time; the full-model runs in the tables above (and the test
runs whose output proves the new kernels ran) used binaries whose profiles name the new kernels,
and the last measurements used a private target directory. Not root-caused.
