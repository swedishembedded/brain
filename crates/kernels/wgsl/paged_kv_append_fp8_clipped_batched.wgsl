// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference serving for clients whose GPU memory
// decides how many concurrent users one card carries. If your team needs expertise in compact
// KV caches you can procure our services by sending an email to info@swedishembedded.com.

// @what  Append a batch of new tokens' K/V into an FP8 E4M3 paged pool, with a calibrated clip ceiling on the online scale
// @how   one thread per (token, kv-head) row, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The FP8 sibling of `paged_kv_append_i8_clipped_batched`: each (token, kv-head) row gets its own
// f32 scale `min(absmax, clip[head]) / 448` (448 is the largest finite E4M3 value) and each element
// is stored as the E4M3 byte nearest `x / scale` (round to nearest even, saturating at 448; the NaN
// code 0x7F is never written). Four bytes pack into a `u32` word, element 4w + b in byte b, the
// layout of the int8 tier, so one row scale buffer serves both and the pool geometry is identical.
// `clip[head] == f32::MAX` is the uncalibrated sentinel: the scale is then the row's own absmax.
//   src   : [batch, kv_stride] f32   (kv_stride = n_kv * head_dim)
//   clip  : [n_kv] f32               (per-kv-head calibrated ceiling)
//   pool  : E4M3 bytes packed as u32 words  ([num_blocks*block_size*kv_stride / 4])
//   scales: [num_blocks*block_size*n_kv] f32
struct Params { batch: u32, kv_stride: u32, block_size: u32, head_dim: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:     array<f32>;
@group(0) @binding(2) var<storage, read>       blocks:  array<u32>;
@group(0) @binding(3) var<storage, read>       offsets: array<u32>;
@group(0) @binding(4) var<storage, read>       clip:    array<f32>;
@group(0) @binding(5) var<storage, read_write> pool:    array<u32>;
@group(0) @binding(6) var<storage, read_write> scales:  array<f32>;

// The E4M3 byte nearest `y` (the sign is handled by the caller; `y >= 0`, already <= 448).
fn e4m3_magnitude(y: f32) -> u32 {
    // Below 2^-6 the format is subnormal: a multiple of 2^-9 whose index is the byte itself
    // (index 8 is the smallest normal, byte 0x08, so the carry is exact).
    if (y < 0.015625) { return u32(round(y * 512.0)); }
    // Normal: round the f32 mantissa to 3 bits (ties to even) with integer arithmetic; a carry
    // into the exponent field is the correct rounding up to the next binade.
    let bits = bitcast<u32>(y);
    let r = bits + 0x7FFFFu + ((bits >> 20u) & 1u);
    return min((r >> 20u) - 960u, 126u);
}

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
    var scale = min(mx, clip[head]) / 448.0;
    if (scale == 0.0) { scale = 1.0; }
    let slot = blocks[b] * p.block_size + offsets[b];
    scales[slot * n_kv + head] = scale;
    let base = slot * p.kv_stride + head * hd;
    for (var i: u32 = 0u; i < hd; i = i + 4u) {
        var packed = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let x = src[kbase + i + j] / scale;
            let code = e4m3_magnitude(min(abs(x), 448.0)) | select(0u, 0x80u, x < 0.0);
            packed = packed | (code << (8u * j));
        }
        pool[(base + i) / 4u] = packed;
    }
}
