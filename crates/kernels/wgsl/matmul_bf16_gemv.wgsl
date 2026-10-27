// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast decode for bf16 checkpoints for its clients. If your team needs expertise in
// bandwidth-bound GEMVs on one accelerator then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Float-read GEMV of a bf16 weight against up to 8 f32 activation rows - the decode-regime projection (registered at one row)
// @how   16 lanes per weight row, 16-byte vector loads of eight weights, f32 accumulation in lane order, ordered fold across lanes
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `y[mi, c] = sum_k x[xr0 + mi, k] * bf16(w[first + c, k])` for the dense bf16 tier (`Weight::BF16`: two weights to a
// word along `k`, the even one in the low half). The library product would first cast the activations to bf16 (a launch of
// its own) and run a tensor-core GEMM whose skinny shapes leave the card's bandwidth idle; this kernel reads each weight
// byte once for all of the (at most eight) rows, takes the f32 activations as they are, and accumulates in f32.
//
//   params : u32 [m, k, count, first, xr0, rpw, 0, 0]   k a multiple of 8; rpw = 4-row tiles one workgroup walks
//   x      : [xr0 + m, k] f32
//   w      : [n, k/8] vec4<u32>      eight bf16 weights per vec4
//   y      : [m, count] f32
//
// Lane `l` owns the 8-weight chunks `l, l + 16, ...` of a row, summed in ascending chunk order; the 16 lane accumulators are
// folded in ascending lane order. One workgroup (64 invocations) covers 4 weight rows.

struct Params {
    m: u32,
    k: u32,
    count: u32,
    first: u32,
    xr0: u32,
    rpw: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const LANES: u32 = 16u;
const COLS: u32 = 4u;
// The most activation rows one launch carries: a tunable (`kernels::template::gemv_variants` registers 1, 2, 4 and 8).
const MAXM: u32 = 8u;

var<workgroup> partial: array<f32, 128u * MAXM>;   // [round parity][thread][activation row]

fn lo(u: u32) -> f32 { return bitcast<f32>(u << 16u); }
fn hi(u: u32) -> f32 { return bitcast<f32>(u & 0xFFFF0000u); }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg.y * nwg.x + wg.x;
    let lane = li.x % LANES;
    for (var g = 0u; g < p.rpw; g = g + 1u) {
        let col = (blk * p.rpw + g) * COLS + li.x / LANES;
        let live = col < p.count;
        var acc: array<f32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
        if (live) {
            let kv = p.k / 8u;                        // 8-weight chunks (one vec4<u32>) per row
            let wbase = (p.first + col) * kv;
            // Four chunks per iteration: their loads are issued together, so each thread keeps 64 bytes of the
            // weight in flight (the device needs megabytes in flight to stream at its roof).
            for (var j0 = 0u; j0 < kv; j0 = j0 + 4u * LANES) {
                var w4: array<vec4<u32>, 4>;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let j = j0 + q * LANES + lane;
                    if (j < kv) { w4[q] = w[wbase + j]; } else { w4[q] = vec4<u32>(0u); }
                }
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let j = j0 + q * LANES + lane;
                    if (j < kv) {
                        for (var r = 0u; r < MAXM; r = r + 1u) {
                            if (r < p.m) {
                                let x0 = (p.xr0 + r) * (p.k / 4u) + 2u * j;
                                let xa = x[x0];
                                let xb = x[x0 + 1u];
                                var a = acc[r];
                                a = a + lo(w4[q].x) * xa.x;
                                a = a + hi(w4[q].x) * xa.y;
                                a = a + lo(w4[q].y) * xa.z;
                                a = a + hi(w4[q].y) * xa.w;
                                a = a + lo(w4[q].z) * xb.x;
                                a = a + hi(w4[q].z) * xb.y;
                                a = a + lo(w4[q].w) * xb.z;
                                a = a + hi(w4[q].w) * xb.w;
                                acc[r] = a;
                            }
                        }
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
