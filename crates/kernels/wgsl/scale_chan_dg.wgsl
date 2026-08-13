// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-channel scale backward (gain grad), the scale_chan companion
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel scale backward (gain grad), the scale_chan companion:
//   dscale[c] += Σ_{rows,inner} x[r,c,i] · dy[r,c,i]
// One invocation per channel (generic [rows, C, inner] layout; dx is just
// scale_chan with the same gain). Accumulates like the other *_dw kernels.

struct Params {
    total: u32, // rows*C*inner
    c: u32,
    inner: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dg: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.c) { return; }
    let rows = p.total / (p.c * p.inner);
    var acc = 0.0;
    for (var r = 0u; r < rows; r = r + 1u) {
        for (var i = 0u; i < p.inner; i = i + 1u) {
            let idx = (r * p.c + c) * p.inner + i;
            acc = acc + x[idx] * dy[idx];
        }
    }
    dg[c] = dg[c] + acc;
}
