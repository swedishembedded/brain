// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet's per-chunk causal decay mask
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
//   decay_mask[row,i,j] = exp(g_cs[row,i] - g_cs[row,j])   if j <= i
//                       = 0                                 otherwise
// (`torch_chunk_gated_delta_rule`'s
// `decay_mask = (g_cs[...,:,None] - g_cs[...,None,:]).tril().exp() *
// tril(ones)`, computed here directly without materialising the pre-exp
// difference). `row` ranges over `bhc = B*H*n_chunks`; `g_cs` is
// `[bhc, c_len]`, `decay_mask` is `[bhc, c_len, c_len]`, both row-major. One
// invocation per `(row,i,j)`.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       g_cs: array<f32>;
@group(0) @binding(2) var<storage, read_write> decay_mask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let cc = p.c_len * p.c_len;
    if (idx >= p.bhc * cc) { return; }
    let row = idx / cc;
    let rest = idx % cc;
    let i = rest / p.c_len;
    let j = rest % p.c_len;
    if (j <= i) {
        let base = row * p.c_len;
        decay_mask[idx] = exp(g_cs[base + i] - g_cs[base + j]);
    } else {
        decay_mask[idx] = 0.0;
    }
}
