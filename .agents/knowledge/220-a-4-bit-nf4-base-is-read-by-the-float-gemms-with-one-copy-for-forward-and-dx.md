<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 220. A 4-bit NF4 base is read by the float GEMMs, one copy for forward and dX

`Weight::NF4` (QLoRA) is not the W4A8 tier the older `matmul_q4_gemv_nf4`
kernel serves. Training needs the weight read against fp32 activations in both
`x*W^T` and `dY*W`, so `kernels::template::nf4_weight_variant` rewrites the `w`
binding of `matmul`, `matmul_reg3`, `matmul_dx` and `matmul_dx_reg` (the same
mechanism as the int8 base, finding 197): eight codes per word, each load decoded
as `NF4_CODEBOOK[code] * scale[wi >> 5]`. One packed copy serves both
directions; no activation is quantised.

Things that bite:

- `DType::NF4.promote` used to demand `int8_dot` (it was W4A8). The float-read
  tier needs no capability, so it now promotes to itself everywhere, including
  the CPU JIT; otherwise `Weight::upload(.., NF4)` silently became F32 and every
  test passed on the wrong tier.
- The tiled variants (`matmul_reg3`, `matmul_dx_reg`) carry workgroup barriers
  the CPU JIT cannot run, and the CPU backend's fast paths match kernels by
  name, so a `#w=nf4` tiled variant has none. `Ops` chooses the tile only when
  `caps.workgroup_reductions` holds; the CPU keeps the naive kernels, which is
  also what host oracle tests exercise. The tiled kernels are verified under
  `--ignored` on a GPU.
- The codebook has one source, `kernels::template::NF4_CODEBOOK`
  (`model::lut4::NF4_LUT` is an alias), so the host quantizer and the generated
  WGSL decode helper cannot drift.
- `Ops::matmul_dx` now takes the tiled kernel for bf16 as well as NF4; the
  kernels are optional (`model::ops::frozen_base_kernels()`, union in
  `training_kernel_list()`) so models that register only `kernel_list()` keep
  working with the naive bf16 dX.
