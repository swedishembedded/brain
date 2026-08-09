// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet's per-token decay gate: g = -exp(A_log) * softplus(a+dt_bias)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Gated DeltaNet's (Qwen3.5-35B-A3B linear-attention layer) raw log-decay
// gate, `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`'s
// `Qwen3_5MoeGatedDeltaNet.forward`:
//   g[row,h] = -exp(A_log[h]) * softplus(a_proj[row,h] + dt_bias[h])
// `a_proj` is `in_proj_a(hidden)` (a plain matmul, dispatched separately —
// this kernel only fuses the elementwise tail); `A_log`/`dt_bias` are
// per-value-head `[num_v_heads]` parameters, broadcast over rows.
//
// `softplus(x) = log(1+exp(x))` is computed in the STABLE shifted form
// `max(x,0) + log(1+exp(-abs(x)))`, not the naive one: for large positive
// `x`, `exp(x)` overflows f32 long before `softplus(x) ~= x` would, so the
// naive form returns `inf` where the true value is a plain finite number.
// The stable form's `exp(-abs(x))` is always in `(0,1]`, never overflows, and
// is exact in the `x -> -inf` limit too (`softplus(x) -> 0`).
//
// `beta` (`sigmoid(in_proj_b(hidden))`) needs no kernel of its own — it is
// exactly `sigmoid.wgsl` applied to `in_proj_b`'s matmul output.

struct Params {
    rows: u32,
    num_v_heads: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a_proj:  array<f32>;
@group(0) @binding(2) var<storage, read>       a_log:   array<f32>;
@group(0) @binding(3) var<storage, read>       dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> g:       array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.num_v_heads;
    if (idx >= total) { return; }
    let h = idx % p.num_v_heads;
    let x = a_proj[idx] + dt_bias[h];
    let softplus = max(x, 0.0) + log(1.0 + exp(-abs(x)));
    g[idx] = -exp(a_log[h]) * softplus;
}
