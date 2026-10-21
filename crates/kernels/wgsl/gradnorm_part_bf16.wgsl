// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded full fine-tuning of large language models for its clients. If your
// team needs expertise in measuring a global gradient norm over gradients that never leave the device then you can
// procure our services by sending an email to info@swedishembedded.com.

// @what  Sum of squares of a bf16 gradient (two elements per word), as a COOPERATIVE tree reduction
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The bf16-gradient twin of `gradnorm_part.wgsl`: `n_wg` workgroups of 64 threads split one gradient buffer that
// holds `numel` bf16 elements packed two to a u32 word (low half first, element 2i in the low half), each writing
// one f32 partial into `parts[out_off + wg]`. A trainer whose gradients are bf16 measures the global norm without
// widening them to f32 first.
//
// `numel` is even (a bf16 gradient needs an even row length). Dispatch: n_wg * 64 invocations, `n_wg` from
// `paramstore::gradnorm_parts`.

struct Params {
    numel: u32,
    out_off: u32,  // this tensor's first slot in `parts`
    n_wg: u32,     // workgroups this dispatch really uses
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       grad:  array<u32>;
@group(0) @binding(2) var<storage, read_write> parts: array<f32>;

var<workgroup> psum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (w >= p.n_wg) { return; }

    let words = p.numel / 2u;
    let stride = p.n_wg * 64u;
    var acc = 0.0;
    for (var i = w * 64u + t; i < words; i = i + stride) {
        let bits = grad[i];
        let lo = bitcast<f32>(bits << 16u);
        let hi = bitcast<f32>(bits & 0xffff0000u);
        acc = acc + lo * lo + hi * hi;
    }
    psum[t] = acc;
    workgroupBarrier();
    if (t == 0u) {
        var s = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            s = s + psum[k];
        }
        parts[p.out_off + w] = s;
    }
}
