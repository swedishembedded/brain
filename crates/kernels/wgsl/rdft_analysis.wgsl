// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Truncated real orthonormal Fourier analysis:  z[r,:] = S^T x[r,:]  (D samples -> m coefficients) by radix-2 FFT
// @how   one 256-thread workgroup per row, in-place radix-2 FFT in workgroup memory, log2(D) barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
//   x: [rows, D]   z: [rows, m]
//   params: rows, D, m, log2(D)      dispatch: one workgroup per row
//   D is a power of two, 2 <= D <= 2048;  m <= D.
//
// The transpose of `rdft_synth` (see its header for the basis S). With
// X[k] = sum_j x[j] e^{-2 pi i k j/D}:
//
//   z_0 = Re X[0] / sqrt(D),  z_{2k-1} = sqrt(2/D) Re X[k],
//   z_{2k} = -sqrt(2/D) Im X[k],  z_{D-1} = Re X[D/2] / sqrt(D).
//
// Only the first m coefficients are written. It is also the backward pass of
// `rdft_synth` (dx -> dz = S^T dx). GPU-only for the reason given there.

struct Params {
    rows: u32,
    d: u32,
    m: u32,
    log2d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;

const WG: u32 = 256u;
const TWO_PI: f32 = 6.283185307179586;

var<workgroup> buf: array<vec2<f32>, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let d = p.d;
    let shift = 32u - p.log2d;

    for (var j = tid; j < d; j = j + WG) {
        buf[reverseBits(j) >> shift] = vec2<f32>(x[row * d + j], 0.0);
    }
    workgroupBarrier();

    // Decimation-in-time butterflies with the -i exponent: the forward DFT.
    for (var s = 1u; s <= p.log2d; s = s + 1u) {
        let half = 1u << (s - 1u);
        for (var b = tid; b < d / 2u; b = b + WG) {
            let pos = b % half;
            let i0 = (b / half) * 2u * half + pos;
            let i1 = i0 + half;
            let angle = -TWO_PI * f32(pos) / f32(2u * half);
            let w = vec2<f32>(cos(angle), sin(angle));
            let v = buf[i1];
            let t = vec2<f32>(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);
            let u = buf[i0];
            buf[i0] = u + t;
            buf[i1] = u - t;
        }
        workgroupBarrier();
    }

    let inv_sqrt_d = inverseSqrt(f32(d));
    let sqrt_2_over_d = sqrt(2.0 / f32(d));
    for (var i = tid; i < p.m; i = i + WG) {
        var out = 0.0;
        if (i == 0u) {
            out = buf[0].x * inv_sqrt_d;
        } else if (i == d - 1u) {
            out = buf[d / 2u].x * inv_sqrt_d;
        } else {
            let k = (i + 1u) / 2u;
            if ((i & 1u) == 1u) {
                out = sqrt_2_over_d * buf[k].x;
            } else {
                out = -sqrt_2_over_d * buf[k].y;
            }
        }
        z[row * p.m + i] = out;
    }
}
