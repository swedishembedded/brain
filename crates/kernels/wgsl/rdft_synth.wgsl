// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Truncated real orthonormal Fourier synthesis:  x[r,:] = S z[r,:]  (m coefficients -> D samples) by radix-2 FFT
// @how   one 256-thread workgroup per row, in-place radix-2 FFT in workgroup memory, log2(D) barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
//   z: [rows, m]   x: [rows, D]
//   params: rows, D, m, log2(D)      dispatch: one workgroup per row
//   D is a power of two, 2 <= D <= 2048;  m <= D.
//
// S is the D x m matrix whose columns are the lowest m functions of the real
// orthonormal Fourier basis of R^D, in frequency order:
//
//   phi_0     = 1/sqrt(D)
//   phi_{2k-1}(j) = sqrt(2/D) cos(2 pi k j / D)       k = 1 .. D/2-1
//   phi_{2k}  (j) = sqrt(2/D) sin(2 pi k j / D)
//   phi_{D-1} (j) = (-1)^j / sqrt(D)                  (Nyquist)
//
// This is `model::hostmath::real_fourier_basis` (the reference) applied as
// a matrix-vector product, computed in O(D log D) instead of O(D m). The
// coefficients become the Hermitian spectrum C[0] = z0/sqrt(D),
// C[k] = sqrt(2/D)(a_k - i b_k)/2 and C[D-k] = conj(C[k]) for the cos/sin pair
// (a_k, b_k), C[D/2] = z_{D-1}/sqrt(D), and x[j] = sum_k C[k] e^{+2 pi i k j/D}
// is real. Because S has orthonormal columns this kernel is also the backward
// pass of `rdft_analysis` (dz -> dx = S dz), and `rdft_analysis` is its own.
//
// One row per workgroup with the twiddles computed in the butterfly: at
// D = 512 this measured faster than a per-workgroup twiddle table shared by
// several rows per workgroup.
//
// GPU-only (`@cpu no`): the CPU JIT executes at most one top-level barrier,
// and the FFT needs log2(D). A caller selects it only when the device reports
// workgroup reductions and otherwise runs the same product as a dense matmul
// against the stored basis.

struct Params {
    rows: u32,
    d: u32,
    m: u32,
    log2d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       z: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

const WG: u32 = 256u;
const TWO_PI: f32 = 6.283185307179586;

var<workgroup> buf: array<vec2<f32>, 2048>;

fn coef(row: u32, i: u32) -> f32 {
    if (i < p.m) { return z[row * p.m + i]; }
    return 0.0;
}

// The Hermitian spectrum bin k (0 <= k < D) of the band-limited signal.
fn spectrum(row: u32, k: u32) -> vec2<f32> {
    let d = p.d;
    let half = d / 2u;
    let inv_sqrt_d = inverseSqrt(f32(d));
    let half_sqrt_2_over_d = 0.5 * sqrt(2.0 / f32(d));
    if (k == 0u) { return vec2<f32>(coef(row, 0u) * inv_sqrt_d, 0.0); }
    if (k == half) {
        if (d - 1u < p.m) { return vec2<f32>(coef(row, d - 1u) * inv_sqrt_d, 0.0); }
        return vec2<f32>(0.0, 0.0);
    }
    if (k < half) {
        return half_sqrt_2_over_d * vec2<f32>(coef(row, 2u * k - 1u), -coef(row, 2u * k));
    }
    let kk = d - k;
    return half_sqrt_2_over_d * vec2<f32>(coef(row, 2u * kk - 1u), coef(row, 2u * kk));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * nwg.x + wid.x;
    if (row >= p.rows) { return; }
    let tid = lid.x;
    let d = p.d;
    let shift = 32u - p.log2d;

    for (var k = tid; k < d; k = k + WG) {
        buf[reverseBits(k) >> shift] = spectrum(row, k);
    }
    workgroupBarrier();

    // Iterative decimation-in-time butterflies over bit-reversed input; the
    // +i exponent makes this the unnormalized inverse DFT.
    for (var s = 1u; s <= p.log2d; s = s + 1u) {
        let half = 1u << (s - 1u);
        for (var b = tid; b < d / 2u; b = b + WG) {
            let pos = b % half;
            let i0 = (b / half) * 2u * half + pos;
            let i1 = i0 + half;
            let angle = TWO_PI * f32(pos) / f32(2u * half);
            let w = vec2<f32>(cos(angle), sin(angle));
            let v = buf[i1];
            let t = vec2<f32>(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);
            let u = buf[i0];
            buf[i0] = u + t;
            buf[i1] = u - t;
        }
        workgroupBarrier();
    }

    for (var j = tid; j < d; j = j + WG) {
        x[row * d + j] = buf[j].x;
    }
}
