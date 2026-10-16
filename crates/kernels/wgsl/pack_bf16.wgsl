// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large language
// models on a single GPU for its clients. If your team needs expertise in
// holding activations in half the bytes without changing what a layer
// computes then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Round an f32 row buffer to bf16 and pack two per u32: out[j] = bf16(x[2j]) | bf16(x[2j+1]) << 16
// @how   one thread per output word; round-to-nearest-even on the 16 dropped bits, NaN stays NaN
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The device twin of `model::half::pack_bf16`, bit for bit: element 2j in the
// low half of word j, element 2j+1 in the high half, an odd total leaving the
// last high half zero. It exists so a training pass can keep a layer-boundary
// residual stream in bf16 (half the bytes) without a host round trip; the
// matching decode is `unpack_bf16`, and the bf16 weight tier's kernels read the
// same packing.

struct Params { total: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn bf16_bits(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        // NaN: keep the top payload bits and force the quiet bit so it can
        // never truncate into an infinity.
        return (b >> 16u) | 0x0040u;
    }
    return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let j = gid.y * (nwg.x * 64u) + gid.x;
    if (j >= (p.total + 1u) / 2u) { return; }
    let lo = bf16_bits(x[2u * j]);
    var hi = 0u;
    if (2u * j + 1u < p.total) { hi = bf16_bits(x[2u * j + 1u]); }
    out[j] = lo | (hi << 16u);
}
