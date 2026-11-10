// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Strided row copy between batched row tables: dst[b, dst_off + r] = src[b, src_off + r] for r < n
// @how   one thread per copied element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Packs and unpacks the per-image token tables of Mirage (cells, registers, proposal slots). A source stride of
// zero broadcasts one table to every image. Rows are `d` wide.
// Params: batch, n, d, src_stride, src_off, dst_stride, dst_off. Dispatch: batch*n*d threads.

struct Params { batch: u32, n: u32, d: u32, src_stride: u32, src_off: u32, dst_stride: u32, dst_off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.n * p.d) { return; }
    let c = idx % p.d;
    let r = (idx / p.d) % p.n;
    let b = idx / (p.d * p.n);
    dst[((b * p.dst_stride + p.dst_off + r) * p.d) + c] = src[((b * p.src_stride + p.src_off + r) * p.d) + c];
}
