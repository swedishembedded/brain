<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 601. A device capability flag is a promise to every caller of the selector, not to the one path being enabled

**Lesson:** `numeric.int8_dot` on a backend does not mean "this backend has an int8 GEMM". It means every
kernel `select::candidates` can then return for every int8-family dtype (I8, Q4, Q4K, Q8K, NF4, F4E2M1) runs
there, and every model that reads the flag (activation quantization widths, `Dtype::promote`, engine `w8_on`
gates) may change what it builds. Turning it on to reach one kernel is a change to all of them. To hold a packed
tier on a device without the packed dot, keep the flag off and read the codes through the tile decode
(`Ops::hold_packed_by_float_read`, `Ops::matmul`'s float-read fallback), or route a native kernel through
`CpuIsaProvider`; and decide the tier a checkpoint is held at per tensor from its headers, pricing it with
`Weight::bytes_for` (#068), never from the first tensor.

## Evidence (2026-11-26, review of an external K-quant series, brain d11d26063 + 5)

The series set `int8_dot: true` on the CPU backend after adding native routes for `matmul_i8_dyn`, its `#QPG=1`
sibling and `matmul_kq_dyn`. On CPU, `brain-model --test it` went from 1 failure (pre-existing
`glu_act::default_swiglu_is_silu_mul_and_its_backward_to_the_bit`) to 13: `Dtype::Q4` selected
`matmul_q4_dyn_reg`, which nothing on CPU runs ("was not JIT-compiled"); qwen35 started quantizing a 112-wide
activation ("int8 K must be a multiple of 32"). The routes themselves were wrong where the tests skipped on CPU:
`matmul_kq_dyn` read its bindings as `[xq, wq, wsm, wd, xgs, sx, out]` against the kernel's
`[xq, wq, sx, wsm, wd, xgs, out]` (relative L2 3.1e5, then heap corruption), and the group-16 route indexed
activation scales per 16 elements where they are kept per 32 (NaN). The `kquant` test filter the series was
verified with skips every kernel test on CPU, so it reported 9 passed. Qwen3 meanwhile priced Q4K from the head
tensor's format alone and had no loader for it: a Q4_K GGUF that served at fp32 before panicked at engine build.

The fix kept `int8_dot` off, removed the routes, and served the K-quants through `gguf::upload_weight` /
`upload_fused_weight` with the tile-decode fallback; `qwen3/tests/it/engine_gguf_kquant.rs` holds a mixed
Q4_K/Q5_K/Q6_K/Q8_0 model as stored on the CPU backend, with the planner's bytes equal to the engine's.
