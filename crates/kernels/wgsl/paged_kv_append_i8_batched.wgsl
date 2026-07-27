// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Append a batch of new tokens' K (or V) into an INT8 paged pool: per (sequence b,
// kv-head) compute a symmetric scale = max|x|/127, quantize the head's head_dim
// values to int8, and pack 4 per u32 into the pool at the sequence's (block,
// offset); the scale is stored per (token slot, kv-head). head_dim must be a
// multiple of 4 (so a packed u32 stays within one head → one scale).
//   src   : [batch, kv_stride] f32   (kv_stride = n_kv * head_dim)
//   pool  : int8 packed as u32 words  ([num_blocks*block_size*kv_stride / 4])
//   scales: [num_blocks*block_size*n_kv] f32
struct Params { batch: u32, kv_stride: u32, block_size: u32, head_dim: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:     array<f32>;
@group(0) @binding(2) var<storage, read>       blocks:  array<u32>;
@group(0) @binding(3) var<storage, read>       offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> pool:    array<u32>;
@group(0) @binding(5) var<storage, read_write> scales:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let n_kv = p.kv_stride / p.head_dim;
    if (idx >= p.batch * n_kv) { return; }
    let b = idx / n_kv;
    let head = idx % n_kv;
    let hd = p.head_dim;
    let kbase = b * p.kv_stride + head * hd;
    var mx = 0.0;
    for (var i: u32 = 0u; i < hd; i = i + 1u) {
        mx = max(mx, abs(src[kbase + i]));
    }
    var scale = mx / 127.0;
    if (scale == 0.0) { scale = 1.0; }
    let slot = blocks[b] * p.block_size + offsets[b];
    scales[slot * n_kv + head] = scale;
    let base_i8 = slot * p.kv_stride + head * hd;
    for (var i: u32 = 0u; i < hd; i = i + 4u) {
        var packed = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let qv = clamp(i32(round(src[kbase + i + j] / scale)), -127, 127);
            packed = packed | ((bitcast<u32>(qv) & 0xffu) << (8u * j));
        }
        pool[(base_i8 + i) / 4u] = packed;
    }
}
