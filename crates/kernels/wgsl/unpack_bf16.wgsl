// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large language
// models on a single GPU for its clients. If your team needs expertise in
// holding activations in half the bytes without changing what a layer
// computes then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Decode a bf16 buffer packed two per u32 to f32: out[i] = bf16 half (i & 1) of word i >> 1, widened
// @how   one thread per output element; a shift into the top 16 bits (exact)
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The inverse of `pack_bf16`: bf16 is the top 16 bits of an f32, so the decode
// is exact and the round trip is the one-way rounding `pack_bf16` applied.

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let half = (packed[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
    out[i] = bitcast<f32>(half << 16u);
}
