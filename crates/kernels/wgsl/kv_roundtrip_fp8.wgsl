// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the exact numerics they are served with,
// for its clients. If your team needs expertise in quantization-aware reinforcement learning against an inference
// engine then you can procure our services by sending an email to info@swedishembedded.com.

// @what  Round-trip a K or V buffer through the FP8 (E4M3) KV cache's encoding, in place: per-(row, kv head) absmax scale, nearest E4M3 code
// @how   one thread per (row, kv head), serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// What an FP8 paged KV cache does to a key or value row between the append and the attention read, applied to the
// training forward's own K or V so that a trainer scores a rollout with the numerics the server sampled it with
// (`paged_kv_append_fp8_clipped_batched`'s arithmetic with no calibrated ceiling: scale = absmax / 448 over the head's `head_dim` elements, 1 where that is 0;
// the nearest E4M3 magnitude of |x| / scale, ties to even, capped at 448; read back as the code's value * scale). In
// place, so the attention that follows, forward and backward, reads the cache's values; the gradient passes straight
// through (the backward never reads the unquantised row).
//   buf : [rows, kv_stride] f32   (kv_stride = n_kv * head_dim)
struct Params { rows: u32, kv_stride: u32, head_dim: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

// The E4M3 byte nearest `y` (`y >= 0`, already <= 448): `paged_kv_append_fp8_clipped_batched`'s encoder.
fn e4m3_magnitude(y: f32) -> u32 {
    if (y < 0.015625) { return u32(round(y * 512.0)); }
    let bits = bitcast<u32>(y);
    let r = bits + 0x7FFFFu + ((bits >> 20u) & 1u);
    return min((r >> 20u) - 960u, 126u);
}

// The value of an E4M3 magnitude byte: subnormal below 8 (a multiple of 2^-9), else (1 + mantissa / 8) * 2^(exponent - 7),
// built from the f32 bits (the exponent field is the format's plus 120) so it is exact and needs no pow.
fn e4m3_value(m: u32) -> f32 {
    if (m < 8u) { return f32(m) / 512.0; }
    return bitcast<f32>(((m >> 3u) + 120u) << 23u | (m & 7u) << 20u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let n_kv = p.kv_stride / p.head_dim;
    if (idx >= p.rows * n_kv) { return; }
    let row = idx / n_kv;
    let head = idx % n_kv;
    let hd = p.head_dim;
    let base = row * p.kv_stride + head * hd;
    var mx = 0.0;
    for (var i: u32 = 0u; i < hd; i = i + 1u) {
        mx = max(mx, abs(buf[base + i]));
    }
    var scale = mx / 448.0;
    if (scale == 0.0) { scale = 1.0; }
    for (var i: u32 = 0u; i < hd; i = i + 1u) {
        let x = buf[base + i] / scale;
        let v = e4m3_value(e4m3_magnitude(min(abs(x), 448.0)));
        buf[base + i] = select(v, -v, x < 0.0) * scale;
    }
}
