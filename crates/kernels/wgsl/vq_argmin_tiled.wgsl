// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Vector-quantization nearest-codebook assignment (Euclidean) for 4 queries per WORKGROUP over a transposed codebook - the row-tiled form of `vq_argmin.wgsl`
// @how   256-thread workgroup, 4-row tile, 2 barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// `vq_argmin.wgsl`'s assignment for a tile of 4 query vectors per workgroup:
//
//   x  : [m, d]  (d <= 256)   cbt: [d, k], the codebook TRANSPOSED
//   out: [2m]   params: m, k, d
//   out[2m]   = f32(argmin_k ||x[m] - cb[k]||^2)   (index, tie -> LOWEST k)
//   out[2m+1] = the minimum squared distance
//
// The 4 queries are staged in workgroup memory; lane `t` takes codebook rows
// `t, t + 256, ...` and reads them from the transposed codebook, so a warp's
// lanes load consecutive words (one coalesced read where the row-major layout
// is rows a kilobyte apart), and each word loaded serves all 4 queries. The
// partial minima fold per query after one barrier. `vq_argmin_rows` reads the
// whole codebook once per query; this reads it once per 4, with as many
// threads per query (64) as that twin. Dispatch: ceil(m / 4) workgroups.
//
// NUMERICS: bit-identical to `vq_argmin`: every distance is the same serial
// sum over `d` in the same order, and the minimum with its lowest index is
// exact whatever order the lanes fold in.

struct Params {
    m: u32,
    k: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       cbt: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const ROWS: u32 = 4u;
const LANES: u32 = 256u;

var<workgroup> xs: array<f32, 1024>; // ROWS * 256 (d <= 256)
var<workgroup> pd: array<f32, 1024>; // ROWS * LANES
var<workgroup> pk: array<u32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for a 1D dispatch).
    let tile = wg.y * nwg.x + wg.x;
    let m0 = tile * ROWS;
    if (m0 >= p.m) { return; }
    let t = li.x;
    let rows = min(ROWS, p.m - m0);
    for (var e = t; e < rows * p.d; e = e + LANES) {
        xs[e] = x[m0 * p.d + e];
    }
    workgroupBarrier();
    var best_d = array<f32, 4>(3.402823e38, 3.402823e38, 3.402823e38, 3.402823e38);
    var best_k = array<u32, 4>(0u, 0u, 0u, 0u);
    for (var k = t; k < p.k; k = k + LANES) {
        var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        for (var i: u32 = 0u; i < p.d; i = i + 1u) {
            let c = cbt[i * p.k + k];
            for (var r = 0u; r < ROWS; r = r + 1u) {
                let diff = xs[r * p.d + i] - c;
                acc[r] = acc[r] + diff * diff;
            }
        }
        for (var r = 0u; r < ROWS; r = r + 1u) {
            if (acc[r] < best_d[r]) {
                best_d[r] = acc[r];
                best_k[r] = k;
            }
        }
    }
    for (var r = 0u; r < ROWS; r = r + 1u) {
        pd[r * LANES + t] = best_d[r];
        pk[r * LANES + t] = best_k[r];
    }
    workgroupBarrier();
    if (t < rows) {
        var gd = pd[t * LANES];
        var gk = pk[t * LANES];
        for (var i = 1u; i < LANES; i = i + 1u) {
            let dd = pd[t * LANES + i];
            let kk = pk[t * LANES + i];
            if (dd < gd || (dd == gd && kk < gk)) {
                gd = dd;
                gk = kk;
            }
        }
        out[2u * (m0 + t)] = f32(gk);
        out[2u * (m0 + t) + 1u] = gd;
    }
}
