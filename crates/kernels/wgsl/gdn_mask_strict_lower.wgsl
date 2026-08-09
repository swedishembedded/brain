// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Finish Gated DeltaNet's attn0: strictly-lower-triangular mask times the decay mask
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
//   attn0[idx] = raw[idx] * decay_mask[idx]   if j < i
//              = 0                             otherwise
// Second half of `torch_chunk_gated_delta_rule`'s
//   attn = -(k_beta @ key^T) * decay_mask
//   attn = attn.masked_fill(triu(ones, diagonal=0), 0)
// `raw` is `-(k_beta @ key^T)` (from `bmm.wgsl` with `alpha = -1`, `trans_b =
// 1`); this kernel applies the decay mask AND additionally zeroes `j >= i`
// (`gdn_decay_mask.wgsl` alone only zeroes `j > i` — the diagonal `j == i`
// entries it leaves at `exp(0) = 1` must also go to 0 here). Flat layout
// matches `gdn_decay_mask.wgsl`: `[bhc, c_len, c_len]` row-major.
//
// The alternative decomposition (a single fused `attn0`-producing kernel
// doing the batched dot product itself) was not taken: `bmm.wgsl` already
// computes the masked dot product's unmasked half exactly, so this kernel
// only needs to be the small elementwise finish, reusing the general
// primitive rather than a bespoke batched-dot-product kernel.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       raw: array<f32>;
@group(0) @binding(2) var<storage, read>       decay_mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> attn0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let cc = p.c_len * p.c_len;
    if (idx >= p.bhc * cc) { return; }
    let rest = idx % cc;
    let i = rest / p.c_len;
    let j = rest % p.c_len;
    if (j < i) {
        attn0[idx] = raw[idx] * decay_mask[idx];
    } else {
        attn0[idx] = 0.0;
    }
}
