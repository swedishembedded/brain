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
| LoRA step at block 3532, s (tok/s) | 12.85 (275) | 3.37 (1047) | `lora-8b-{fp32,auto2}.log` |
| training forward, s | 4.16 | 0.31 | same |

The training step went 12.85 -> 7.61 s with TF32 GEMMs plus the native attention forward (`lora-8b-auto.log`)
and -> 3.37 s once the native backward and the library LoRA products landed. Memory is unchanged by the tier
(peak 41.9 GB sampled by `nvidia-smi` over both runs, `gpu-run8b.csv`; L0 measured 41.4 GB for the fp32 trainer).

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
