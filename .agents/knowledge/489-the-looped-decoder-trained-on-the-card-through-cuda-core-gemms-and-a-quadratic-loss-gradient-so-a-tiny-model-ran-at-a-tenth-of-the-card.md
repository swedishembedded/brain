<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 489. The looped decoder trained on the card through CUDA-core GEMMs and a quadratic loss gradient, so a tiny model ran at a tenth of the card

**Lesson:** A model that dispatches the portable kernels by index bypasses every selection the shared layers make
(library tensor-core GEMMs, flash attention, the two-stage loss). Route a training graph's products through
`model::ops::Ops`, its attention through `block::FlashGqaIds`, and its loss through the statistics kernels, and
profile each step before guessing: the cost sat in three kernels nobody had looked at.

## What happened

`brain-ouro` built its training graph from `MATMUL`/`MATMUL_DX`/`MATMUL_DW` steps, the materialized `[H,T,T]`
attention chain and `ce_grad`, whose per-row cost is quadratic in the vocabulary. On a GH200 the default size
(5.3M parameters, rows of 1024, batch 8) trained at 4680 tokens/s; nsys put 40% of the step in `ce_grad`, 15% in
the attention backward's score gradient and 11% in `matmul_dw`, with no tensor-core kernel in the trace.

## What changed

- The products go through `Ops` (`matmul`, `matmul_dx`, `matmul_dw`): the library TF32 GEMMs under the default
  `GemmPrecision::Auto`, the exact fp32 kernels when a gradient check pins `Fp32Reference`
  (`Ouro::new_train_with` takes the precision; `full_train` passes the run's).
- Attention is the flash trio from 256 tokens up (`FlashGqaIds::for_training`), the native fp16 kernels where the
  device runs them and the precision is not pinned; the loss gradient is `ce_stats` + `ce_grad_stats` (linear in
  the vocabulary).
- `Ops::matmul_dw` takes the library TF32 product too (it was the portable kernel unconditionally), and RMSNorm's
  gain gradient over a training batch runs as chunked partial sums and their reduction (`rmsnorm_dw_part`,
  `rmsnorm_dw_sum`): one thread per channel walked all 8192 rows.
- Measured on gpu3, step rate after the first step, same seed: 4680 -> 17300 tokens/s (Ops, flash, two-stage
  loss) -> 34400 (TF32 weight gradient, two-stage gain gradient); rows of 512: 43800. The step losses are equal
  to four decimals at every stage (8.3660, 8.1619, 7.8158). In the remaining step 85% of the kernel time is the
  portable flash trio: at head_dim 64 no native kernel existed.
