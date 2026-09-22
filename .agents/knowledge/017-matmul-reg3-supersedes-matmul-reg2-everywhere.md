<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 17. `matmul_reg3` supersedes `matmul_reg2` - everywhere

They are the same kernel: identical `Params`, identical `@workgroup_size(256)`,
identical dispatch arithmetic, bit-identical output. `reg3` is `reg2` with the
shared-memory bank conflicts removed. Swept across twelve shapes from
`[1,4096,4096]` to `[8192,320,320]`, `reg3` won **every one** by 1.08x-1.30x,
with `max|Δ| = 0` at each. There is no shape where preferring `reg2` is correct.

`crates/unet` registered `matmul_reg2` *beside* the `matmul_reg3` it already
carried through `vae::blocks`, and sent every `nn.Linear` to the slower of the
two it was holding. Dropping its own slot and pointing `pick_gemm` at
`vae::blocks::MATMUL_REG3_SLOT` took the SDXL UNet forward from 1349 ms to
1261 ms with one fewer pipeline compiled.

This is the **fourth** instance of #8 in one workstream, and the second where
the fast kernel was already registered in the same process. When two kernels
differ only by an optimisation, the slow one should not be selectable - the
place to encode that is the shared slot the block set exports, so a caller
cannot re-derive the wrong answer. Eleven other crates still register `reg2`.
