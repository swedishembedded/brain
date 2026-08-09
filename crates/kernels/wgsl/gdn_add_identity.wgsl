// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Add the identity to Gated DeltaNet's UT-transform matrix
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// t_mat[row,i,i] += 1.0, for every row in [0,bhc) and i in [0,c_len) — the
// final step of `torch_chunk_gated_delta_rule`'s UT-transform, applied after
// `gdn_ut_step.wgsl`'s forward-substitution loop has finished every row:
// `attn = attn + eye(c_len)`. Diagonal entries are always exactly 0 going in
// (never touched by `gdn_ut_step.wgsl`, whose column index never reaches the
// diagonal), so `+=` and `=` are equivalent here; `+=` is used to match the
// reference literally. `t_mat` is `[bhc, c_len, c_len]` row-major.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> t_mat: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.bhc * p.c_len) { return; }
    let row = idx / p.c_len;
    let i = idx % p.c_len;
    let cc = p.c_len * p.c_len;
    let off = row * cc + i * p.c_len + i;
    t_mat[off] = t_mat[off] + 1.0;
}
