// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of Gated DeltaNet's attn0 mask-and-multiply
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward of `gdn_mask_strict_lower.wgsl`'s
//   attn0[idx] = raw[idx] * decay_mask[idx]   if j < i
//              = 0                             otherwise
// Both operands of the multiply get a gradient (zero outside the mask, since
// `attn0` itself is exactly 0 there so neither operand's value could have
// affected the loss through this op):
//   d_raw_attn0[idx]   = d_attn0[idx] * decay_mask[idx]   if j < i else 0
//   d_decay_mask[idx] += d_attn0[idx] * raw_attn0[idx]    if j < i else 0
//
// `d_raw_attn0` is a plain overwrite into its own dedicated buffer (`attn0`'s
// only forward consumer is this mask-multiply, so `raw_attn0` has exactly one
// producer of its gradient). `d_decay_mask` ACCUMULATES: `decay_mask` has a
// SECOND producer (`gdn_chunk_fwd`'s `intra_scores = raw_intra * decay_mask`
// precompute, this module's `mul.wgsl`-backward contribution, committed
// earlier via `splice_add.wgsl`) — this kernel's `+=` is the second and final
// contribution. Flat layout matches `gdn_mask_strict_lower.wgsl`:
// `[bhc, c_len, c_len]` row-major.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_attn0:     array<f32>;
@group(0) @binding(2) var<storage, read>       raw_attn0:   array<f32>;
@group(0) @binding(3) var<storage, read>       decay_mask:  array<f32>;
@group(0) @binding(4) var<storage, read_write> d_raw_attn0: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_decay_mask: array<f32>;

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
        let da = d_attn0[idx];
        d_raw_attn0[idx] = da * decay_mask[idx];
        d_decay_mask[idx] = d_decay_mask[idx] + da * raw_attn0[idx];
    } else {
        d_raw_attn0[idx] = 0.0;
    }
}
