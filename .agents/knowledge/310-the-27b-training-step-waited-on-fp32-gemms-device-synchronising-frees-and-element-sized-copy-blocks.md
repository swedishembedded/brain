<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 310. The 27B training step waited on fp32 GEMMs, on frees that stop the device, and on copy kernels with one block per float

Question: why does one Qwen3.8-27B LoRA/QLoRA step run at 20 tokens/s on a card that
sustains 800 TFLOP/s of bf16, and which of the causes in
`.agents/roadmap/throughput-27b.md` move the number when removed? Every figure below
is labelled **VERIFIED-BY-RUN** (measured in the run named) or **DERIVED** (arithmetic
from those runs and from shapes).

## Conditions of every run

The GH200 was shared for most of the session: other compute contexts (one to three
other training and serving processes, one of them at 40-98% SM) were present in nearly every
run, which is why the same binary read 82 tokens/s and 1,030 tokens/s in two runs
minutes apart. **The figures marked "idle" are the exception**: the card had no other
compute context at process start (`nvidia-smi pmon` empty before, `other_contexts_start=0`
in the probe line) and the run held the gpu lock; a context arrived during the third
step of some of them, so only steps 1-2 are quoted. Everywhere else, read the ratio of
two runs taken together, not the absolute rate.

Workload W4: `qwen35_train_probe --layers 4 --tokens 4096 --rank 16 --bf16-residuals on
--mlp-rows 512 --gdn-segment 2048`, random weights, real 248,320 vocabulary, seed 7
for the new binary (the old binary has no `--seed`). W8K: `--layers 64 --tokens 8192
--base nf4 --mlp-rows 2048 --gdn-segment 2048 --bf16-residuals on --rank 16 --seed 7`.

## What was removed, in order of effect

1. **Frees that wait for the whole device (host side).** Sampling the probe's main
   thread with a debugger while it ran showed 18 of 25 samples inside `cuMemFree_v2`
   (called from `DeviceMem::drop`), the rest in `cuStreamSynchronize`. A training build
   opened its device handle with the backend's block cache off, so every released
   activation was a synchronising driver call; thousands per step. Keeping the freed
   blocks (`hold_freed_blocks`, which `qwen3` already used for decode) took W4 bf16 from
   650-730 to 1,130-1,160 tokens/s (VERIFIED-BY-RUN, same card state, same session). The
   cost is memory: the garbage between two drains stays reserved (W8K NF4 peak 37.7 to
   51.8 GiB), so the build holds blocks only for the 4-bit base or a model of at most 32
   layers, not for a bf16 27B.
2. **No tensor cores.** Every GEMM ran as generated fp32 code (about 33 TFLOP/s ceiling
   without contraction). bf16 and NF4 weight GEMMs, the head, the LoRA products and the
   attention matrix products now run through `libcublas` (`cublasGemmEx`,
   `cublasGemmStridedBatchedEx`, bf16 operands, fp32 accumulation). Component rates, the
   minimum over 8 runs of the whole step including the activation cast (VERIFIED-BY-RUN,
   `crates/gpu-core/tests/cuda_library_bench.rs`, shared card): 8192x17408x5120 643
   TFLOP/s; 8192x5120x17408 439; 8192x12288x5120 819; 8192x10240x5120 343; the
   2048-row MLP chunk 759; the head tile 1024x62080x5120 294; the n = 1024 k/v
   projection only 54 (4% of the FLOPs).
3. **Attention.** The fp32 scalar flash trio is replaced, at 16 GQA layers, by blockwise
   attention around bf16 GEMMs with causal block skipping. T = 8192, 24/4 heads of 256:
   forward 6.7 ms and backward 33 ms per layer (VERIFIED-BY-RUN, same benchmark), against
   about 750 ms per layer fwd+bwd before (knowledge 240, DERIVED to 8K): per step about
   0.75 s instead of about 15 s (DERIVED).
4. **Copy kernels with one block per element.** The native `concat_split` /
   `chan_place` / `concat2` take a block per run of one plane. Called for a channel
   window of a row-major matrix (`H = W = 1`: the MLP chunk `take_rows`, the GQA query
   split, the GDN `mixed_act` split) a plane is one float, so a 512 x 5120 window
   launched 2.6 M blocks; 262 calls cost 24% of the step in the profile. Planes under 32
   floats are now copied flat (bit-identical to the reference, `plumb_native.rs` with
   the new shapes).
5. **Head windows.** The vocabulary windows of the chunked head had arbitrary row
   counts (38,605 and 78,707 columns), so the library GEMM declined them on alignment
   and the two largest products of the step stayed on the fp32 kernel. Windows are now
   whole groups of 64 rows.
6. **Launches.** The GDN training forward is one native launch (recurrence plus both
   histories, bit-identical to the host walk at chunk lengths 64..1). The backward
   computes all `d_state`-free quantities for every chunk at once and sweeps with six
   dispatches per chunk instead of twenty-nine. Linears reading one activation share
   one bf16 cast. W4 dispatches per step: 10,466 (fp32 tier after Q1) to 4,798;
   W8K: 121,247 to 114,207 after the last two items (VERIFIED-BY-RUN, probe census).
7. **Reductions given to one thread.** Four kernels reduced a whole tensor or row
   with a single thread and were on the default path: `gradnorm_sq` (one thread per
   adapter tensor, 118 M elements in all: 5.5 ms a call at W4), `ce_stats` (one thread per
   logit row of 248,320 entries: 77 ms a call, four calls per 4096 tokens), the
   decay-scale gradient of the GDN state update (one thread per head, once per chunk) and
   the per-element `rmsnorm`/`rmsnorm_dx` (0.73 and 1.3 ms a call at 8K rows). The
   cooperative `gradnorm_part` was already written and needed only to be registered by
   the model; `ce_stats` and the decay gradient now have block-per-row/-head native
   kernels (fp32 summation bound against the reference, `ce_stats_native.rs`,
   `gdn_dscale_native.rs`); the RMSNorm twins were registered for training too. On the
   idle card these took the W8K step from 149-158 s (shared) through 17.5 s to 14.7 s.
8. **Q1.** The backward's layer rebuild no longer replays the MLP when the MLP is
   chunked, and a chunk rebuild skips the unread `down` product: the chunked backward does
   no more GEMM FLOPs than the whole-sequence one (unit test on the cost counters).

## Measurements

W4 (4 layers, 4096 tokens), tokens/s of steps 1-3, old and new binary run back to back
under the same lock with two other compute contexts present (VERIFIED-BY-RUN, final binary):

| base | old (c8a33d4e era) | new (this series) | ratio |
|---|---|---|---:|
| bf16 | 153, 141, 137 | 2160, 3150, 3124 | 21x |
| nf4 | 144, 141, 141 | 2062, 2875, 2823 | 20x |

Earlier pairs of the same comparison, taken while the card was busier: bf16 82-86 against
897-1036, nf4 168-292 against 1138-1432. With one context finishing mid-run the new bf16
binary read 7,200-7,300 (0.56 s a step). A 4-layer step carries the head, the loss and the
optimiser once, so these rates are not the 64-layer rates; the W8K table is.

W8K, 64 layers, 8192 tokens, NF4 (VERIFIED-BY-RUN):

| state | s/step | tokens/s | dispatches | peak GiB | card |
|---|---:|---:|---:|---:|---|
| earlier record, old code (knowledge 234) | about 410 | 20 | about 870,000 (DERIVED) | | |
| library GEMMs, attention, batched GDN backward | 238-243 | 34 | 121,247 | 37.7 | shared, 2 contexts |
| with freed blocks kept | 149-158 | 53 | 114,207 | 51.8 | shared, 2 contexts |
| + cooperative grad-norm, block-per-row CE and decay-gradient kernels | 17.5 | **469** | 114,207 | 50.0 | **idle at start**, gpu lock |
| + cooperative RMSNorm, forward and backward | 14.7-14.9 | **550-560** | 114,207 | 51.7 | **idle at start**, gpu lock |

Loss agrees between the tiers to the 4th digit step by step (13.4252, 13.4198, 13.4143 on
the tensor-core and the fp32 tier at W4 bf16; the CUDA fp32 tier matches the CPU backend at
cosine 0.99999 on the tiny model, `lora_cuda_matches_cpu.rs`), the 30-step tier trajectory
gate holds the loss curve within 2% and the adapters at cosine 0.99
(`train_tier_trajectory.rs`), and the 64-layer run's loss falls 13.46, 13.40, 13.32.

Device time per kernel kind at W8K, idle card, one timed step (VERIFIED-BY-RUN, before the
RMSNorm change; sum of kernel durations 12.0 s of a 17.2 s step):

| kind | ms | calls | note |
|---|---:|---:|---|
| library GEMMs | 2,640 | 21,896 | 1.25 PFLOP: about 470 TFLOP/s effective |
| `bmm_tiled` (GDN, fp32) | 1,671 | 29,760 | the reverse sweep's four small products per chunk dominate the calls |
| GDN recurrence, recompute forward | 824 | 384 | |
| GDN recurrence, training forward | 431 | 192 | |
| `rmsnorm` + `rmsnorm_dx` (one thread per row) | 1,117 | 1,250 | replaced by the cooperative kernels afterwards |
| `gdn_layout_permute` | 516 | 4,608 | |
| `l2norm_scale` + `_dx` | 596 | 1,536 | |
| NF4 decode to bf16 | 305 | 2,960 | |
| everything else | about 3,800 | | 40 kinds under 3% each |

The 5 s between that sum and the step is the host: 114,207 dispatches at about 45 us each,
issued one at a time (K8's parameter ABI and graphs are the lever) plus about 3,000 drains.

## What the numbers do and do not say

- **1,000 tokens/s is not reached**: the best idle figure is 550-560 on the 64-layer NF4
  step. What stands between that and 1,000 is, in the device table: the GDN chain (`bmm_tiled`,
  both recurrence launches, layout permutes, `l2norm`: about 3.7 s of the 12 s), the library
  GEMMs themselves (2.6 s, at the card's practical rate), and the launch overhead (about 5 s).
  A persistent GDN backward kernel and per-layer graphs address the first and the last.
- The bf16 27B and the real-checkpoint CLI path were not run end to end in this session
  (the card was shared and the load takes minutes); bf16 at 64 layers, 8K tokens needs about
  54 GB of base, which is why the freed-block cache is off for it.

## Not done

K4 (fused head gradient), Q2, Q6, K7, K8 (arena, parameter ABI, graphs); a persistent
GDN backward kernel; the shared-input A concatenation of K3; the real-checkpoint CLI
runs; an exclusive-card measurement. The GDN sweep (six dispatches per chunk, 128 chunks
per layer at 8K) is about 30% of the remaining launches.
