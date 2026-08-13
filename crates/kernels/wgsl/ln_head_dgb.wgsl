// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Strided per-head LayerNorm backward (parameter grads)
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Strided per-head LayerNorm backward (parameter grads): one invocation per
// head-dim channel c, reducing over all (row, head) vectors.
//   dgamma[c] += Σ dy[.,c]·x̂[.,c] ;  dbeta[c] += Σ dy[.,c]
// Accumulates (+=) into the grad buffers like the other *_dw kernels.

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    row_stride: u32,
    off: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       dy:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(4) var<storage, read_write> dbeta:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * (nwg.x * 64u) + gid.x;
    let hd = p.head_dim;
    if (c >= hd) { return; }
    var dg = 0.0;
    var db = 0.0;
    for (var row = 0u; row < p.rows; row = row + 1u) {
        for (var h = 0u; h < p.heads; h = h + 1u) {
            let base = row * p.row_stride + p.off + h * hd;
            var mean = 0.0;
            for (var k = 0u; k < hd; k = k + 1u) { mean = mean + x[base + k]; }
            mean = mean / f32(hd);
            var va = 0.0;
            for (var k = 0u; k < hd; k = k + 1u) {
                let d = x[base + k] - mean;
                va = va + d * d;
            }
            let inv = inverseSqrt(va / f32(hd) + p.eps);
            let xh = (x[base + c] - mean) * inv;
            dg = dg + dy[base + c] * xh;
            db = db + dy[base + c];
        }
    }
    dgamma[c] = dgamma[c] + dg;
    dbeta[c] = dbeta[c] + db;
}
