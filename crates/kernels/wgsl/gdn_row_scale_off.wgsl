// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row scalar scale with independent flat offsets into x and s (scale_row.wgsl, generalised for a chunk slice)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
//   out[i] = alpha * x[x_off + i] * s[s_off + i / m],   i in [0, total)
//
// Same broadcast shape as `scale_row.wgsl` (`y[i] = s[i/m]*x[i]`), plus an
// independent flat offset into `x` and into `s`, and an `alpha` scalar.
// `scale_row.wgsl` itself is reused UNMODIFIED wherever Gated DeltaNet's
// row-scale needs no offset — `k_beta * exp(g_cs)` (`k_cumdecay`), and
// `value`/`key` scaled by `beta` into `v_beta`/`k_beta` — because those run
// once over EVERY chunk at once (whole-tensor, offset 0), matching
// `scale_row.wgsl`'s existing contract exactly (`crates/model/src/gdn.rs`
// checked this before adding a new kernel). This sibling exists only for the
// sequential per-chunk loop's two
// row-scales that read one chunk's slice out of a larger buffer, which
// `scale_row.wgsl`'s zero-offset callers (`crates/model/src/moe.rs`'s
// shared-expert gate, notably) must not be made to pay for or reason about:
//
// * `attn_inter`'s input, `q_c * exp(g_cs_c)`: `x = query` at the chunk's
//   offset, `s = exp_g_cs` at the SAME chunk's offset, `alpha` folds in the
//   `1/sqrt(Dk)` attention scale so `query` itself never needs a separate
//   in-place scale pass.
// * the state update's `decayed_k = key_c * decay_scale`: `x = key` at the
//   chunk's offset, `s = gdn_decay_scale.wgsl`'s fresh dense output
//   (`s_off = 0`), `alpha = 1`.
//
// `out` is always a dedicated destination scratch in every current caller
// (no offset needed there).

struct Params { total: u32, m: u32, x_off: u32, s_off: u32, alpha: f32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       s:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let row = idx / p.m;
    out[idx] = p.alpha * x[p.x_off + idx] * s[p.s_off + row];
}
