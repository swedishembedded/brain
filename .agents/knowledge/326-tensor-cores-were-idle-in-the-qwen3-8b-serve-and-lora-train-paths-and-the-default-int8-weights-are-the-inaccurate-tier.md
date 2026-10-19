<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 326. Tensor cores were idle in the Qwen3-8B serve and LoRA-train paths, and the default int8 weights are the inaccurate tier

Question: how far below the GH200's tensor-core capability did the 8B (`crates/qwen3`) fp32 serving and LoRA
training run, what moves them, and what does it cost in accuracy? Every figure is **VERIFIED-BY-RUN** on one
GH200 card (cores pinned to the card's NUMA node), real Qwen3-8B weights where stated, logs under
`runs/b8/` (not in git). Hardware reference: `brain roofline gpu` measures about 50-58 TFLOP/s fp32 here;
`make gh200/probe` measured 410 TFLOP/s TF32 and 833 bf16 for an 8192^3 GEMM.

## What was found

- Serving prefill ran its fp32-weight linears on CUDA-core fp32 kernels and its attention on the portable WGSL
  `paged_flash_prefill`: one 512-row chunk (8B shapes, int8 KV, 3584 tokens cached) took **520.4 ms, 984 tok/s**
  (`sp-8b-kv8-fp32.log`: `qwen_bench serve-prefill 512 5 3584 kv8 --model 8b --gemm fp32`). The attention was 180 ms of it
  (35 percent, at 12 percent of the compute roof) and the library-free GEMMs about 330 ms (63 percent).
- LoRA training at block 3532 (fp32 base, rank 16, real weights) took **12.85 s/step, 275 tok/s**
  (`lora-8b-fp32.log`: `qwen_bench lora-train --t 3532 --steps 3 --rank 16 --weights Qwen3-8B --gemm fp32
  --profile`): the base linears and the attention fwd/bwd on fp32 CUDA cores (the attention trio at 7-9 percent of
  the compute roof), and the LoRA adapter products and the base dX of the qwen3 trainer on naive kernels (the
  qwen35 trainer already used the library for them).
- `GemmPrecision::Auto` (now the default of `Engine`, `FitOpts` and the SDK `ChatFineTune`) takes tensor cores on
  a capable device: TF32 library GEMMs for fp32 weights (no copy of the weights, binary32 accumulation), bf16 for
  bf16/NF4 weights, fp16 `mma` flash attention (paged prefill at head_dim 128 for fp32/bf16/int8 KV; the training
  forward at 128 and 256; the training backward at 128), fp32 library products for the LoRA adapters.
  `Fp32Reference` (`--qwen-gemm-fp32`, SDK `gemm_precision`) pins the old kernels and bars the native attention.

| 8B, one card | fp32 reference | default tier | log |
|---|---|---|---|
| prefill chunk of 512 rows, ms (tok/s) | 520.4 (984) | 71.3 (7185) | `sp-8b-kv8-{fp32,auto}.log` |
| same, bf16 weights (`--qwen-weights-bf16`) | | 61.0 (8393) | `sp-8b-kv8-bf16w.log` |
| LoRA step at block 3532, s (tok/s) | 12.85 (275) | 1.50 (2349) | `lora-8b-fp32.log`, `lora-final-fp32base-3532.log` |
| training forward, s | 4.16 | 0.43 | same |
| LoRA step, bf16 base, block 8192 (not run in the fp32 tier) | | 2.55 (3214) | `lora-final-bf16base-8192.log` |
| batched decode step, 32 rows, ctx 1024, int8 KV, ms | 220-310 (GEMV) | 74 (library GEMM from 8 rows) | `dec2-8b-32-fp32-fp32.log`, `sweep-8b-r32-min*.log` |
| same, 8 rows | 77 | 57 | `sweep-8b-r8-min*.log` |

The training step went 12.85 -> 7.61 s with TF32 GEMMs plus the native attention forward (`lora-8b-auto.log`),
-> 3.37 s with the native backward and the library LoRA products (`lora-8b-auto2.log`) and -> 1.50 s on the final
tree (`lora-final-fp32base-3532.log`). Memory is unchanged by the tier
(peak 41.9 GB sampled by `nvidia-smi` over the first two runs, `gpu-run8b.csv`; L0 measured 41.4 GB for the fp32 trainer; the bf16-base run at block 8192 was not isolated per process).

## Accuracy against the exact fp32 tier (real Qwen3-8B)

Serving engine, last-token distribution of 16 prompts of 256 tokens plus 24 greedy tokens for 6 of them
(`serve-acc-8b.log`: `qwen_bench serve-accuracy --weights Qwen3-8B --text <repo prose> --t 256 --prompts 16 --gen 24`,
int8 KV in every tier, n = 16 prompts, one text source, no repeat seeds; the fp32 tier repeated bit-identically,
KL 0):

| tier | max abs log-prob difference | KL(ref to tier) mean / max | top-1 agree | greedy tokens equal |
|---|---|---|---|---|
| default (TF32 GEMMs, fp16 attention) | 0.89 | 1.6e-4 / 6.4e-4 | 16/16 | 144/144 |
| bf16 weights, tensor cores | 0.62 | 1.9e-4 / 9.3e-4 | 16/16 | 144/144 |
| bf16 weights, fp32 kernels | 0.27 | 1.7e-4 / 8.0e-4 | 16/16 | 132/144 |
| int8 weights (the serve default for 6B+) | 9.6 | 7.5e-2 / 5.1e-1 | 12/16 | 49/144 |

Training forward path (TF32 GEMMs, fp32 attention) over 2048 positions of real text (`acc-8b.log`:
`qwen_bench gemm-accuracy --t 512 --windows 4`): max |dlogit| 0.22, relative RMS 6.8e-4, top-1 agreement
2048/2048, KL mean 3.6e-7, mean next-token NLL 3.5974 in both tiers (n = 2044). The LoRA step's loss on the
real weights is 11.653 in the fp32 and the default tier alike (`lora-8b-*.log`). The gradients are gated on tiny
and head_dim-128 models (`crates/qwen3/tests/lora_gemm_precision.rs`: loss to 2e-3, every adapter gradient at
cosine above 0.999 against `Fp32Reference`) and the native attention kernels against the portable ones and an f64
oracle (`crates/gpu-core/tests/cuda_flash_{prefill_hd128,attn_fwd,attn_bwd}.rs`; the backward reaches cosine
1.000000 and a worst error of 1.7 percent of the gradient's RMS, also for a gradient a million times below unit scale).

## Why it matters

- The registered CE4 serving flags (`--qwen-weights-fp32`) were the exact tier, and the tensor-core default now
  deviates from it by KL 1.6e-4 per token on real prompts with identical greedy tokens in this sample: a prereg
  amendment can state that contract. The tier that actually moves the logits is **int8 weights**, which is what
  `brain serve` picks by default for any checkpoint of 6B parameters or more: 12 of 16 top-1 agreements here.
  Do not let an unflagged 8B server decide a registered result.
- A profile that ranks kernels by device time alone hid the GEMMs: the library steps are not timestamped, so the
  per-kind table showed the portable attention at 94 percent of the timed kernel time while it was 35 percent of
  the pass. Rank with the whole-pass number (#21), and re-profile after each tier change: once the GEMMs moved, the
  attention became the top row, and once it moved the LoRA products and the per-layer elementwise kernels did.
- Training tapes are recorded at construction, so a GEMM tier chosen with a setter afterwards silently changed
  nothing (the first TF32 training run was identical to fp32 to the digit). The tier is a constructor argument
  (`Qwen::new_shard_gemm`, `new_lora_shard_dt_gemm`).
- A gradient is far below fp16's normal range (mean-reduced loss: dO of order 1e-5): an fp16 backward needs a
  power-of-two scale of the whole call, chosen from the largest |dO|, or the small rows underflow.
- `qwen_bench serve` refused an 8B shape through a hard-coded 22 GB budget that no 90 GiB card needed; it is now
  `--budget-gb`.
- Decode: at 32 rows the GEMV kernels re-read the weight per row group (15 percent of the memory roof); the
  library GEMM from 8 rows cut the 8B step from 220-310 ms to 74 ms (a 2-row threshold gave 559 ms in one noisy run,
  so the threshold is 8). What remains at 32 rows is the int8 attention triad (92 ms at a 1024-token context, growing
  with it) and per-dispatch cost: the next lever for an agentic request, whose 300 decode tokens (9.4 s at one
  stream) now cost more than its 4k-token prefill (0.6 s).

## Quiet-card end-to-end, 8B through `brain serve` (addendum)

One card with no other tenant (12:50-13:05), `--qwen-ctx 8192 --qwen-max-batch 32`, fp32 weights unless named, 8 prompts of about
3.7k tokens for prefill and 300 greedy tokens per stream for decode (`e2e8b-decode.log`, rows stamped 12:5x; commands in `run8b5.sh`,
`bench.py`, `benchdec.py`; single run each, so a 5 percent difference is not resolvable):

| tier | prefill tok/s | decode tok/s at 1 / 8 / 32 streams | server memory |
|---|---|---|---|
| exact fp32 (`--qwen-gemm-fp32`, the CE4 registered flags) | 1100 | 33 / 134 / 186 | 34.6 GB |
| default: TF32 GEMMs, fp16 attention, native batched decode attention | 8315 | 36 / 460 / 1607 | 34.7 GB |
| bf16 weights (`--qwen-weights-bf16`) | 6596 | 46 / 569 / 1904 | 20.3 GB |
| int8 weights | 4311 | 45 / 434 / 832 | 13.9 GB |

- The default tier is 7.6x faster at prefill and 8.6x at 32 streams of decode than the exact tier; the bf16 tier decodes about
  18 percent faster than fp32 at 8 and 32 streams on half the memory and prefills a fifth slower, and int8 is slower than both at batch
  32 and, per the accuracy table, far less accurate. (Rows stamped 04:3x-04:4x in the same log are from a shared card and are not
  comparable.) The one-stream rate barely moves: a single stream is bound by the weight stream and by per-step work, not by the GEMM.
- The residual-add + norm fusion, switched off with `--qwen-no-fuse-norm` (`fp32-nofuse` row): 1432 / 405 tok/s at 32 / 8 streams
  against 1607 / 460, and the 32-row decode pass in `qwen_bench serve` 17.55 ms against 15.50 ms (`ab-8b-r32-*.log`; 8 rows 17.19 against
  15.05): about 11 percent.
- CUDA-graph replay switched off (`BRAIN_CUDA_GRAPHS=0`, `fp32-nographs` row): 1600 against 1607 tok/s: no measurable effect at
  these rates, so the library-GEMM capture support added for it has not been shown to matter (whether the serving tape replays at all was
  not counted; `launch_stats().graph_replays` would say).
- LoRA step on the final tree, 8B real weights, fp32 base, block 3532: 1.094 s/step, 3230 tok/s (`lora-final-fp32base-3532.log`), and bf16 base at
  block 8192: 2.53 s/step, 3242 tok/s. The log line printed "Qwen3-0.6B" because the label came from the `--model` default rather than from
  the checkpoint; the run itself loaded Qwen3-8B (164-172 s model build, 34 GB process footprint, the checkpoint's own 36-layer shape),
  and the label now prints the trained shape (`checkpoint (36 layers, d_model 4096, 8.19 B parameters)`). The earlier 1.50 s/step figure in
  the table above was measured before the later launch-count changes.
