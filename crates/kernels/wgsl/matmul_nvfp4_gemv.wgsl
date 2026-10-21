// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast decode for FP8 and NVFP4 checkpoints for its clients. If your team needs
// expertise in bandwidth-bound quantized GEMVs on one accelerator then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Float-read GEMV of an NVFP4 weight (E2M1 codes, E4M3 scale per 16, f32 tensor scale) against up to 8 f32 activation rows - the decode-regime projection and LM head
// @how   16 lanes per weight row, 16-byte vector loads of 32 codes, byte-pair E2M1 table in shared memory, ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// The NVFP4 sibling of `matmul_fp8_gemv.wgsl` for a dense `Weight::Micro` (the LM head and any NVFP4 projection that
// is not an expert bank): `y[mi, c] = global * sum_k x[mi, k] * e2m1(w[row, k]) * e4m3(scale[row, k / 16])`,
// `row = first + c`. The arithmetic and layout are `moe_nvfp4_gemv_gather.wgsl`'s with a plain row index in place of an
// expert's block, so a projection and an expert agree to the bit on equal inputs.
//
//   params : u32 [m, k, count, first, xr0, global_bits, rpw, pad]   k a multiple of 64; rpw = 4-row tiles one workgroup
//            walks (amortising the table): `ceil(count / 4 / rpw)` workgroups are dispatched
//   x      : [xr0 + m, k] f32
//   w      : [n, k/8] u32       eight codes per word
//   s      : [n, k/64] u32      four E4M3 scale bytes per word
//   y      : [m, count] f32

struct Params {
    m: u32,
    k: u32,
    count: u32,
    first: u32,
    xr0: u32,
    global_bits: u32,
    rpw: u32,
    pad1: u32,
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

var<workgroup> pair_lut: array<vec2<f32>, 256>;
var<workgroup> partial: array<f32, 128u * MAXM>;   // [round parity][thread][activation row]

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

fn fp4_value(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    var v = f32(m) * 0.5;
    if (e != 0u) { v = bitcast<f32>(((126u + e) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}

fn dot8(wd: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    let p0 = pair_lut[wd & 255u];
    let p1 = pair_lut[(wd >> 8u) & 255u];
    let p2 = pair_lut[(wd >> 16u) & 255u];
    let p3 = pair_lut[wd >> 24u];
    return p0.x * xa.x + p0.y * xa.y + p1.x * xa.z + p1.y * xa.w + p2.x * xb.x + p2.y * xb.y + p3.x * xb.z + p3.y * xb.w;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    for (var i = li.x; i < 256u; i = i + 64u) { pair_lut[i] = vec2<f32>(fp4_value(i & 15u), fp4_value(i >> 4u)); }
    workgroupBarrier();
    let blk = wg.y * nwg.x + wg.x;
    let lane = li.x % LANES;
    for (var g = 0u; g < p.rpw; g = g + 1u) {
        let col = (blk * p.rpw + g) * COLS + li.x / LANES;
        let live = col < p.count;
        var acc: array<f32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
        if (live) {
            let kv = p.k / 32u;
            let row = p.first + col;
            let wbase = row * kv;
            let sbase = row * (p.k / 64u);
            for (var j = lane; j < kv; j = j + LANES) {
                let wd = w[wbase + j];
                let sw = s[sbase + j / 2u] >> ((j % 2u) * 16u);
                let s0 = e4m3(sw & 255u);
                let s1 = e4m3((sw >> 8u) & 255u);
                for (var r = 0u; r < MAXM; r = r + 1u) {
                    if (r < p.m) {
                        let x0 = (p.xr0 + r) * (p.k / 4u) + 8u * j;
                        let b0 = dot8(wd.x, x[x0], x[x0 + 1u]) + dot8(wd.y, x[x0 + 2u], x[x0 + 3u]);
                        let b1 = dot8(wd.z, x[x0 + 4u], x[x0 + 5u]) + dot8(wd.w, x[x0 + 6u], x[x0 + 7u]);
                        acc[r] = acc[r] + b0 * s0;
                        acc[r] = acc[r] + b1 * s1;
                    }
                }
            }
        }
        let bank = (g & 1u) * 64u * MAXM;
        for (var r = 0u; r < MAXM; r = r + 1u) { partial[bank + li.x * MAXM + r] = acc[r]; }
        workgroupBarrier();
        if (lane == 0u && col < p.count) {
            let c0 = (li.x / LANES) * LANES;
            for (var r = 0u; r < p.m; r = r + 1u) {
                var t = 0.0;
                for (var i = 0u; i < LANES; i = i + 1u) { t = t + partial[bank + (c0 + i) * MAXM + r]; }
                y[r * p.count + col] = t * bitcast<f32>(p.global_bits);
            }
        }
    }
}
