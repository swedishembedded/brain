// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on the exact numerics they are served with,
// for its clients. If your team needs expertise in quantization-aware reinforcement learning against an inference
// engine then you can procure our services by sending an email to info@swedishembedded.com.

// @what  Round-trip a K or V buffer through the INT8 KV cache's encoding, in place: per-(row, kv head) absmax scale, round, clamp, dequantise
// @how   one thread per (row, kv head), serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// What an INT8 paged KV cache does to a key or value row between the append and the attention read, applied to the
// training forward's own K or V so that a trainer scores a rollout with the numerics the server sampled it with
// (`paged_kv_append_i8_clipped_batched`'s arithmetic with no calibrated ceiling: scale = absmax / 127 over the head's `head_dim` elements, 1 where that is 0,
// code = clamp(round(x / scale), -127, 127), read back as code * scale). In place, so the attention that follows, forward
// and backward, reads the cache's values; the gradient passes straight through (the backward never reads the
// unquantised row), which is the straight-through estimator of the round.
//   buf : [rows, kv_stride] f32   (kv_stride = n_kv * head_dim)
struct Params { rows: u32, kv_stride: u32, head_dim: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

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
    var scale = mx / 127.0;
    if (scale == 0.0) { scale = 1.0; }
    for (var i: u32 = 0u; i < hd; i = i + 1u) {
        let qv = clamp(i32(round(buf[base + i] / scale)), -127, 127);
        buf[base + i] = f32(qv) * scale;
    }
}
