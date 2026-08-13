// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm affine gradients, STAGE 2 of 2 - fold the partials and ACCUMULATE
// @how   one thread per output element, serial fold over P partials
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GroupNorm affine gradients, STAGE 2 of 2 — fold the partials and ACCUMULATE.
//
// One invocation per channel. Reads the `P` partials `gn_dgb_part` wrote and
// adds them into the caller's gradient buffer at the same disjoint offsets
// `gn_dgamma`/`gn_dbeta` used:
//
//   dgb[c]     += sum_t part[(c*P + t)*2 + 0]     (dgamma)
//   dgb[C + c] += sum_t part[(c*P + t)*2 + 1]     (dbeta)
//
// `+=`, not `=`: a parameter gradient accumulates over the whole step and is
// zeroed exactly once by the model's `zero_grads` — the rule the reverse walk
// opens with. Writing here instead of accumulating would drop every
// contribution but the last, and a gradcheck on a single block would not see it.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> dgb:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.C) { return; }

    // Ascending partial order, so a run reproduces.
    var sg = 0.0;
    var sb = 0.0;
    for (var t: u32 = 0u; t < p.P; t = t + 1u) {
        sg = sg + part[(c * p.P + t) * 2u + 0u];
        sb = sb + part[(c * p.P + t) * 2u + 1u];
    }
    dgb[c] = dgb[c] + sg;
    dgb[p.C + c] = dgb[p.C + c] + sb;
}
