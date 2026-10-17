<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 316. The Gated DeltaNet backward as two tensor-core launches, the mixer's layout and convolution glue folded into kernels, and staging loops that waited for one load at a time

Question: after the tensor-core forward of knowledge 314 (GDN chain 3.28 s of an 8.61 s device
step), what does a tensor-core chunk backward plus the layout and convolution glue around it
buy on the 64-layer, 8192-token NF4 step, and where do the kernels spend their cycles? Workload
`qwen35_train_probe --tokens 8192 --layers 64 --base nf4 --steps 3 --rank 16 --seed 7
--gdn-segment 2048 --mlp-rows 2048 --bf16-residuals on` (random weights, real 27B shape).
Figures are **VERIFIED-BY-RUN** unless marked **DERIVED**.

## What changed (commits in `t-gdn2`)

- **Chunk backward, two launches** (`gdn_bwd_sweep_tc`, `gdn_chunk_bwd_post_tc`). The reverse
  sweep threads the state gradient D backwards: `d_v_new = intra^T d_out + decayed_k D` and
  `D <- D exp(g_last) + q_scaled^T d_out - w^T d_v_new`. Every term acts on D per value column,
  so a head splits over four 32-column blocks (192 blocks instead of the fp32 sweep's 48), exactly
  like the forward recurrence. The one product that contracts over the columns, `d_decayed_k =
  v_new D^T`, does not feed the recurrence: the sweep writes the history of D (`[bhc, 128, 128]`,
  allocated only for the tensor-core tier by `GdnBwdScratchBufs::new_for`) and the second kernel
  forms it. The second kernel is one block per (head, chunk) and does everything else the portable
  path spreads over about sixty dispatches: recomputed `v_new`, the output and `w`/`u` products,
  the closed-form adjoint of the unit-triangular inverse (`T^T d_t_mat T^T`), the decay and
  cumulative-sum gradients, the row-scale gradients. Same opt-in and fallback rules as 314.
- **Layout fold.** Both kernels read `d_out` token-major and write `d_value`, `d_raw_g`, `d_beta`
  token-major (`heads_tok` parameter); `gdn_qk_bwd_tok` sums each key head's value heads and
  applies the l2norm adjoint in one pass (two permutes, two `kv_expand_bwd`, two
  `l2norm_scale_dx` before).
- **Convolution glue.** `gdn_conv_silu_tok` (forward: causal depthwise conv, SiLU and the qkv split
  on the token-major rows, history of a continued round read in), `gdn_conv_hist_tok` (next
  round's history) and `gdn_conv_silu_bwd_tok` (backward for a frozen conv weight, including the
  history rows' gradient) replace the transpose, generic convolution, SiLU, transpose and three
  splits (forward) and two concatenations, two transposes, SiLU and convolution backward. A trained
  conv weight, a declined launch or a kernel size other than four rebuilds the channel-major
  copies and runs the old chain (`ConvActs::{ChannelMajor, TokenMajor}`).
- **Kernel tuning** (all four tensor-core GDN kernels). TF32 operand split by mask instead of two
  `cvt.rna.tf32` per value; staging loops unrolled; the post kernel streams its 128 x 128 operands
  through double-buffered `cp.async` slabs and keeps `d_query` / `d_key` in registers; the prefix's
  triangular solve in outer-product form.

## Numerical evidence

| check | result |
|---|---|
| `model/tests/gdn_chunk_fwd_train_tc.rs`, forward and all six backward gradients of the tensor-core tier against the portable fp32 path, chunks 64/32/8/5/1 | largest error relative to magnitude 3.1e-6 (bound 2e-5); the backward is asserted to be exactly two steps (red with the provider disabled) |
| same file, token-major backward against the chunk-major one with the permutes applied on the host, 1 and 2 sequences, 3:1 head repeat | at most 6e-8 (the order of a shared-memory float atomic sum); query/key leg against an f64 transcription 6e-8 |
| `gpu-core/tests/gdn_qk_bwd_native.rs`, `gdn_qk_bwd_tok` against the f64 chain it replaces | 9.7e-8 |
| `model/tests/gdn_mixer_tc.rs`, segmented layer (history crossing every boundary), frozen conv weight, fused conv forward and backward against the portable chain | at most 2e-6 relative |
| `gdn_chunk_fwd_tc` (f64 reference, forward kernels after the mask split) | 2.9e-6 (was 1.4e-6; bounds 2e-5 / 5e-4 near-parallel keys unchanged) |
| existing | `gdn_chunk_bwd` finite differences, `gdn_mixer_{segmented,stream,equivalence}`, qwen35 `lora_{gdn_segments,recompute,bf16_residuals,nf4_base}`, `train_tier_trajectory` green |

The mask split (`hi = bits & ~0x1FFF`, `lo = (x - hi) & ~0x1FFF`) truncates where `cvt.rna` rounds, so
the three-pass TF32 product carries about 19 bits instead of 20: errors double and stay far inside
the bounds. The triangular solve in outer-product form sums each element's terms in the same order
as the recurrence it replaces, so `T` is bit-for-bit unchanged.

## Device time, 64 layers, NF4, step 2 (idle card, `other_contexts_start=0`)

| component | 314 tree (`adf422e8`) | + chunk backward | + layout fold |
|---|---:|---:|---:|
| GDN chain | 3,285 ms (12,480) | 2,591 ms (7,872) | 2,220 ms (6,528) |
| library GEMM | 2,694 ms (21,896) | 2,680 ms (21,896) | 2,662 ms (21,896) |
| element-wise glue | 1,309 ms (9,994) | 1,094 ms (7,114) | 1,093 ms (7,114) |
| attention element-wise | 5 | 5 | 5 |
| RMSNorm | 180 | 179 | 179 |
| head, optimizer, misc | 1,208 (18,763) | 1,098 (17,803) | 1,065 (17,419) |
| **device total** | **8,681 ms (64,479)** | **7,647 ms (56,031)** | **7,224 ms (54,303)** |

Losses at step 3 equal to the fourth digit (13.4607 / 13.4012 / 13.3244-13.3245). Node load average at
the start of the three runs: 14, 25 and 24 (other lanes' builds), so the wall rate was host-bound
(342-405 tokens/s base step 1-2, 447 and 472 after) and is not quoted as a result; device time is the
primary number. The backward `bmm_tiled` (593 ms), `gdn_bwd_sweep_f32` (537 ms), `gdn_layout_permute`
(225 to 95 ms), `l2norm_scale_dx` (254 ms), `kv_expand_bwd` and the row-scale / row-dot / splice
dispatches of the old backward are gone from the profile; the sweep is 197 ms and the post kernel
401 ms.

## What the card could not measure, and what was read instead

The last three commits (convolution kernels, mask split and post-kernel restructure, unrolled
staging and the solve) landed while a 27B decode server held the card through the lease for hours
(97% utilisation, 76 GB resident): an exclusive run was refused, the 64-layer probe does not fit
beside it, and per-kernel wall times on a time-sliced card are not interpretable (a 3-float kernel
read 1.4 ms). The evidence for those commits is **block cycles read inside the kernels** with
`clock64` (device-side, repeatable to 3%), plus a correctness run of every test above on the shared
card:

| kernel | cycles per block before | after | how |
|---|---:|---:|---|
| `gdn_chunk_bwd_post_tc` (one (head, chunk)) | 311K | 156K | mask split (-22%), value rows staged instead of read per fragment (-40K), `d_query`/`d_key` in registers, slab streaming, unrolled elementwise pass |
| `gdn_chunk_loop_tc` (per 2 chunks) | 97K | 61K | unrolled staging loops: load phases were 70% of the kernel |
| `gdn_bwd_sweep_tc` (per 2 chunks) | 100K | 68K | the same |
| `gdn_chunk_prep_tok_train_tc` (one (head, chunk)) | 117K | 80K | the row-by-row triangular solve was 63K cycles of dependent shared-memory loads; the outer-product form takes 29K |

A back-to-back A/B of the two binaries on the shared card (8 layers, 4096 tokens, NF4, three
alternating repetitions, GPU 91-98% busy with the neighbour): GDN chain 1,257 / 1,323 / 1,453 ms
before against 450 / 490 / 294 ms after, dispatches 4,311 to 3,279. These are contention-inflated
and only their ratio (3-4x for the chain) is meaningful.

**DERIVED** idle-card step for the final tree: scaling the step-2 profile above by the cycle ratios
(post 401 to about 200 ms, `loop_tc` 399 and 201 to about 250 and 127, sweep 197 to about 134,
prefix 191 and 107 to about 130 and 73) and replacing the 610 ms of channel-major convolution glue
by about 60 ms, the GDN chain is about 1.1 s and the device step about 6.0-6.2 s. The plan's budget
for T1 + T2 is 1.5 s of chain. An idle-card run of both binaries is queued in
`runs/t-gdn2/final.sh` and should replace this paragraph.

## What remains to 1,000 tokens/s

Device time is now dominated by the library GEMMs (2.66 s, the card's practical rate), 1.1 s of
element-wise glue outside the mixer (`splice_add`, `mul`, `silu_mul`, `add2`, `concat_split`,
`cast_f32_bf16`, the SwiGLU backward pair: package T4), 1.07 s of head, optimizer, NF4 dequant
(307 ms) and other "misc", and the GDN chain. The 8.19 s step needs the host gap to stay under about 2 s
at roughly 44,000 dispatches. The recurrence kernel is still staging-bound (`cp.async` prefetch of
the next chunk's tiles needs smem that costs the second resident block per SM); the permute of the
recurrence output (95 ms) can be folded into its stores.
