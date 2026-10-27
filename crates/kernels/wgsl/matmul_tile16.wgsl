// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exact-arithmetic skinny products in low-latency LLM decode for its
// clients. If your team needs expertise in keeping small fp32 projections off the critical path of a token loop then
// you can procure our services by sending an email to info@swedishembedded.com.

// @what  Exact fp32 matmul (out = x @ W^T) in 16 x 16 output tiles, one output per thread, k staged 64 at a time - the small-N, many-row product
// @how   256 threads as a 16 x 16 tile, x and W tiles staged in shared memory, k ascending per output (no TF32, no reassociation)
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
//   x : [M, K]   W : [N, K]   out : [M, N]   (row-major; `Params{m, k, n}` and binding order as `matmul.wgsl`)
//   Dispatch: ceil(M / 16) * ceil(N / 16) workgroups of 256 threads.
//
// Each output sums its k terms in ascending order into one f32 accumulator, which is the order of `matmul.wgsl`, so the
// result is that kernel's. It exists for the shape the skinny-M kernel serves badly: a few hundred activation rows
// against a narrow weight (the anchor memory's admission projections at a large decode batch), where one workgroup per
// output column leaves most of the device idle and the register-tiled 128 x 128 kernels launch a handful of tiles.
// Here a batch of 256 rows against 256 columns is 256 workgroups.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> xs: array<f32, 1024>;   // [row 16][k 64]
var<workgroup> ws: array<f32, 1040>;   // [col 16][k 64 + 1 pad]: the pad keeps the 16 columns a warp reads in different banks

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tiles_n = (p.n + 15u) / 16u;
    let tile = wg.y * nwg.x + wg.x;
    let tm = tile / tiles_n;
    let tn = tile % tiles_n;
    let ty = li.x / 16u;     // output row within the tile
    let tx = li.x % 16u;     // output column within the tile
    let row = tm * 16u + ty;
    let col = tn * 16u + tx;
    let xr = tm * 16u + ty;
    let wr = tn * 16u + ty;
    var acc = 0.0;
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 64u) {
        // This thread stages four x elements and four W elements of the 64-wide chunk: (row ty, k tx + 16 q) and
        // (column ty, k tx + 16 q). All eight loads are issued before the barrier.
        for (var q = 0u; q < 4u; q = q + 1u) {
            let kk = k0 + tx + 16u * q;
            var xv = 0.0;
            var wv = 0.0;
            if (kk < p.k && xr < p.m) { xv = x[xr * p.k + kk]; }
            if (kk < p.k && wr < p.n) { wv = w[wr * p.k + kk]; }
            xs[ty * 64u + tx + 16u * q] = xv;
            ws[ty * 65u + tx + 16u * q] = wv;
        }
        workgroupBarrier();
        let kn = min(64u, p.k - k0);
        for (var j = 0u; j < kn; j = j + 1u) {
            acc = acc + xs[ty * 64u + j] * ws[tx * 65u + j];
        }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) { out[row * p.n + col] = acc; }
}
