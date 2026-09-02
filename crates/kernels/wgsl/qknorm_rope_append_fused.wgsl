// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  qknorm_rope_fused (see its own header) PLUS the fp32 paged KV append, for K only - M4.2
// @how   64-thread workgroup per (batch, head) row, 1 barrier, re-read after the reduction instead of a register array
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `qwen3::serve`'s fp32-KV branch fed K through `rms` -> `ROPE_PAGED` ->
// `paged_kv_append_batched` (KV_APPEND_B) - three dispatches over the same
// per-head row, the last of which reads back exactly what the second one
// just wrote. This kernel is `qknorm_rope_fused.wgsl` (read that file's
// header for the norm+RoPE derivation and the `workgroup_reductions` gate
// this kernel inherits unchanged) with the paged append folded into the
// SAME post-barrier loop that already computes the rotated pair, so the
// normalized+rotated value is written ONCE to device memory per element,
// straight to its final paged-pool slot, alongside `out` - `out` is kept
// (mirroring `sc.k` in `qwen3::serve::Engine`) because calibration
// (`Engine::calibrate_kv`) and test fixtures read the engine's own
// normalized+rotated K scratch directly, independent of the paged pool's
// packed layout.
//
//   x   : [rows, head_dim]                 out : [rows, head_dim]
//   w   : [head_dim]                       positions/blocks/offsets: [rows / heads]
//   pool: [num_blocks*block_size*heads*head_dim] f32, paged K/V layout
//   params: rows, heads (n_kv_heads), head_dim, eps, rope_base, block_size
//
// Pool addressing matches `paged_kv_append_batched.wgsl`'s own contract
// exactly: `pool[(blocks[b]*block_size+offsets[b])*kv_stride + c]` for
// `kv_stride = heads*head_dim`, `c = head*head_dim + m` - this kernel just
// derives `b`/`head` from `row = b*heads + head` (the same flattening
// `qwen3::serve::Engine::rms`'s `b * nkv` row count already assumes) instead
// of taking a flat `c` directly.

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    eps: f32,
    rope_base: f32,
    block_size: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:         array<f32>;
@group(0) @binding(2) var<storage, read>       w:         array<f32>;
@group(0) @binding(3) var<storage, read>       positions: array<u32>;
@group(0) @binding(4) var<storage, read>       blocks:    array<u32>;
@group(0) @binding(5) var<storage, read>       offsets:   array<u32>;
@group(0) @binding(6) var<storage, read_write> out:       array<f32>;
@group(0) @binding(7) var<storage, read_write> pool:      array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let d = p.head_dim;
    let half = d / 2u;
    let base = row * d;
    var acc = 0.0;
    for (var c = t; c < d; c = c + 64u) {
        let v = x[base + c];
        acc = acc + v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(d) + p.eps);
    let batch = row / p.heads;
    let head = row % p.heads;
    let pos = positions[batch];
    let kv_stride = p.heads * d;
    let slot = blocks[batch] * p.block_size + offsets[batch];
    let pbase = slot * kv_stride + head * d;
    for (var m = t; m < half; m = m + 64u) {
        let x0 = x[base + m]        * inv * w[m];
        let x1 = x[base + m + half] * inv * w[m + half];
        let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(d));
        let cs = cos(angle);
        let sn = sin(angle);
        let y0 = x0 * cs - x1 * sn;
        let y1 = x1 * cs + x0 * sn;
        out[base + m]        = y0;
        out[base + m + half] = y1;
        pool[pbase + m]        = y0;
        pool[pbase + m + half] = y1;
    }
}
