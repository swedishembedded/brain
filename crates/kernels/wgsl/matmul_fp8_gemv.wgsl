// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast decode for FP8 and NVFP4 checkpoints for its clients. If your team needs
// expertise in bandwidth-bound quantized GEMVs on one accelerator then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Float-read GEMV of an FP8 E4M3 weight with block scales against up to 8 f32 activation rows - the decode-regime projection
// @how   16 lanes per weight row, 16-byte vector loads, 256-entry E4M3 table in shared memory, one block scale per 16-weight chunk, ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The decode-regime product of a stored FP8 weight (`Weight::FP8`: E4M3 bytes four to a word along `k`, one f32 scale
// per `block x block` tile of the logical `[n, k]` weight): `y[mi, c] = sum_k x[mi, k] * (e4m3(w[row, k]) * scale)`,
// `row = first + c`. The tile decode of `model::ops::tiled` would write the whole decoded weight to scratch for every
// token; this kernel reads each weight byte once and keeps the decode in registers, so a step costs the weight's own
// bytes. At up to eight rows the activations (a few KiB) sit in cache and the weight is read once for all of them.
//
//   params : u32 [m, k, count, first, block, cb, xr0, rpw]   k a multiple of 16, block a multiple of 16; rpw = 4-row tiles
//            one workgroup walks (amortising the table): `ceil(count / 4 / rpw)` workgroups are dispatched
//   x      : [xr0 + m, k] f32           rows xr0 .. xr0 + m are the activations
//   w      : [n, k/4] u32               four E4M3 bytes per word
//   s      : [ceil(n/block), cb] f32    cb = ceil(k / block)
//   y      : [m, count] f32             y[mi, c] for the weight rows first .. first + count
//
// ## Order of the arithmetic
//
// Lane `l` owns the 16-weight chunks `l, l + 16, ...` of a row. A chunk's 16 products are summed left to right in f32,
// multiplied by the chunk's scale and added to the lane's accumulator in ascending chunk order; the 16 lane
// accumulators are folded in ascending lane order. One workgroup (64 invocations) covers 4 weight rows.

struct Params {
    m: u32,
    k: u32,
    count: u32,
    first: u32,
    block: u32,
    cb: u32,
    xr0: u32,
    rpw: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read>       s: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;
// The most activation rows one launch carries: a tunable (`kernels::template::gemv_variants` registers 1, 2, 4 and 8) so
// the per-row accumulators are registers, not an indexed local array.
const MAXM: u32 = 8u;

var<workgroup> e4m3_lut: array<f32, 256>;
var<workgroup> partial: array<f32, 128u * MAXM>;   // [round parity][thread][activation row]

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

fn dot4(w4: u32, xv: vec4<f32>) -> f32 {
    return e4m3_lut[w4 & 255u] * xv.x + e4m3_lut[(w4 >> 8u) & 255u] * xv.y + e4m3_lut[(w4 >> 16u) & 255u] * xv.z + e4m3_lut[w4 >> 24u] * xv.w;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = li.x; i < 256u; i = i + 64u) { e4m3_lut[i] = e4m3(i); }
    workgroupBarrier();
    let blk = wg.y * nwg.x + wg.x;
    let lane = li.x % LANES;
    for (var g = 0u; g < p.rpw; g = g + 1u) {
        let col = (blk * p.rpw + g) * COLS + li.x / LANES;
        let live = col < p.count;
        var acc: array<f32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
        if (live) {
            let kv = p.k / 16u;                       // 16-weight chunks (one vec4<u32>) per row
            let row = p.first + col;
            let wbase = row * kv;
            let sbase = (row / p.block) * p.cb;
            for (var j = lane; j < kv; j = j + LANES) {
                let w4 = w[wbase + j];
                let sc = s[sbase + (j * 16u) / p.block];
                for (var r = 0u; r < MAXM; r = r + 1u) {
                    if (r < p.m) {
                        let x0 = (p.xr0 + r) * (p.k / 4u) + 4u * j;
                        let chunk = dot4(w4.x, x[x0]) + dot4(w4.y, x[x0 + 1u]) + dot4(w4.z, x[x0 + 2u]) + dot4(w4.w, x[x0 + 3u]);
                        acc[r] = acc[r] + chunk * sc;
                    }
                }
            }
        }
        // Two sets of partials alternate by round, so one barrier per round orders both the writes and the reads.
        let bank = (g & 1u) * 64u * MAXM;
        for (var r = 0u; r < MAXM; r = r + 1u) { partial[bank + li.x * MAXM + r] = acc[r]; }
        workgroupBarrier();
        if (lane == 0u && col < p.count) {
            let c0 = (li.x / LANES) * LANES;
            for (var r = 0u; r < p.m; r = r + 1u) {
                var t = 0.0;
                for (var i = 0u; i < LANES; i = i + 1u) { t = t + partial[bank + (c0 + i) * MAXM + r]; }
                y[r * p.count + col] = t;
            }
        }
    }
}
