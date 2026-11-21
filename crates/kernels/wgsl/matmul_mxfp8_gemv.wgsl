// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast decode for MXFP8 checkpoints for its clients. If your team needs expertise in
// bandwidth-bound microscaled GEMVs then you can procure our services by sending an email to info@swedishembedded.com.

// @what  Float-read GEMV of an MXFP8 weight (E4M3 elements, an E8M0 power-of-two scale per 32) against up to 8 f32 activation rows - the decode-regime projection
// @how   16 lanes per weight row, 16-byte vector loads of 16 E4M3 bytes, register decode, one top-level barrier, ordered fold
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The decode-regime product of a stored MXFP8 weight (`Weight::Micro` of kind `Mxfp8E4m3`: E4M3 bytes four to a word
// along `k`, one E8M0 scale byte per 32 weights, four to a word): `y[mi, c] = sum_k x[mi, k] * e4m3(w[row, k]) *
// 2^(e8m0(s[row, k / 32]) - 127)`, `row = first + c`. The weight's bytes are read once, decoded in registers, and
// applied to every activation row of the launch. No global scale exists (MXFP8 has none).
//
//   params : u32 [m, k, count, first, xr0, pad, pad, pad]   k a multiple of 128 (a row's scales start on a word)
//   x      : [xr0 + m, k] f32
//   w      : [n, k/4] u32       four E4M3 bytes per word
//   s      : [n, k/128] u32     four E8M0 scale bytes per word
//   y      : [m, count] f32
//
// Dispatch: `ceil(count / 4)` workgroups of 64 (four weight rows each).
//
// ## Order of the arithmetic (shared with `moe_mxfp8_gemv_gather.wgsl` and `moe_mxfp8_grouped.wgsl`, to the bit)
//
// Sixteen lanes serve a row; lane `l` owns the 16-weight chunks `l, l + 16, ...`. A chunk's 16 products are summed in
// four groups of four (left to right within a group, the groups in order), multiplied by its block's scale and added
// to the lane's accumulator in ascending chunk order. The 16 lane accumulators are folded in ascending lane order.
// The kernel has exactly one top-level barrier and no workgroup table, so the CPU JIT runs it too.

struct Params {
    m: u32,
    k: u32,
    count: u32,
    first: u32,
    xr0: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read>       s: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;
// The most activation rows one launch carries: a tunable (`kernels::template::gemv_variants` registers 1, 2, 4 and 8) so
// the per-row accumulators are registers, not an indexed local array.
const MAXM: u32 = 8u;

var<workgroup> partial: array<f32, 64u * MAXM>;   // [thread][activation row]

// An OCP E4M3 byte (the code 0x7f / 0xff, NaN in the format, is not produced by a quantizer and reads as 480).
fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

// An E8M0 scale byte: 2^(b - 127); 0 is 2^-127 (an f32 subnormal), 255 is NaN.
fn e8m0(b: u32) -> f32 {
    if (b == 0u) { return bitcast<f32>(0x00400000u); }
    if (b == 255u) { return bitcast<f32>(0x7fc00000u); }
    return bitcast<f32>(b << 23u);
}

fn dot4(w4: u32, xv: vec4<f32>) -> f32 {
    return e4m3(w4 & 255u) * xv.x + e4m3((w4 >> 8u) & 255u) * xv.y + e4m3((w4 >> 16u) & 255u) * xv.z + e4m3(w4 >> 24u) * xv.w;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg.y * nwg.x + wg.x;
    let col = blk * COLS + li.x / LANES;
    let lane = li.x % LANES;
    var acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
    if (col < p.count) {
        let kv = p.k / 16u;                       // 16-weight chunks (one vec4<u32>) per row
        let row = p.first + col;
        let wbase = row * kv;
        let sbase = row * (p.k / 128u);
        for (var j = lane; j < kv; j = j + LANES) {
            let w4 = w[wbase + j];
            let blk32 = j / 2u;
            let sc = e8m0((s[sbase + blk32 / 4u] >> ((blk32 % 4u) * 8u)) & 255u);
            for (var r = 0u; r < MAXM; r = r + 1u) {
                if (r < p.m) {
                    let x0 = (p.xr0 + r) * (p.k / 4u) + 4u * j;
                    let chunk = dot4(w4.x, x[x0]) + dot4(w4.y, x[x0 + 1u]) + dot4(w4.z, x[x0 + 2u]) + dot4(w4.w, x[x0 + 3u]);
                    acc[r] = acc[r] + chunk * sc;
                }
            }
        }
    }
    for (var r = 0u; r < MAXM; r = r + 1u) { partial[li.x * MAXM + r] = acc[r]; }
    workgroupBarrier();
    // After the barrier the row is located again, not carried across it (the CPU JIT re-initialises locals there).
    let col2 = (wg.y * nwg.x + wg.x) * COLS + li.x / LANES;
    if (li.x % LANES == 0u && col2 < p.count) {
        let c0 = (li.x / LANES) * LANES;
        for (var r = 0u; r < p.m; r = r + 1u) {
            var t = 0.0;
            for (var i = 0u; i < LANES; i = i + 1u) { t = t + partial[(c0 + i) * MAXM + r]; }
            y[r * p.count + col2] = t;
        }
    }
}
