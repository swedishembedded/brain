// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Tiled matmul — same math as matmul.wgsl (out = x @ W^T) but GPU-parallelised.
//   x  : [M, K]  row-major (activations)
//   W  : [N, K]  row-major (W[n,k] = weight row n = output feature n)
//   out: [M, N]  row-major,  out[m,n] = sum_k x[m,k] * W[n,k]
//
// Each workgroup computes a BM×BN output tile; its 64 invocations form an 8×8
// grid, each owning a TM×TN (4×4) micro-tile. Per K-chunk the workgroup
// cooperatively stages As[BM][BK] and Bs[BN][BK] into workgroup memory (every A/B
// element read from global memory once per tile instead of once per output — the
// reuse the naive one-invocation-per-output matmul lacks), then accumulates in
// registers. Opt-in fast path for the transformer forecasters (Chronos-2/Kronos).
//
// Dispatch: launch ceil(M/BM)*ceil(N/BN) workgroups (thread count = that*64); the
// flat workgroup id maps to (tile_row, tile_col) via ceil(N/BN).
//
// The CPU JIT skips this kernel (it has a barrier inside the K-loop, which the
// work-group execution model can't express); on CPU it runs the AVX2 fast path
// (`backend-cpu`, keyed on the "matmul_tiled" name) — same math, validated.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 32u;
const BN: u32 = 32u;
const BK: u32 = 8u;
const TM: u32 = 4u;
const TN: u32 = 4u;

var<workgroup> As: array<f32, 256>;  // BM*BK
var<workgroup> Bs: array<f32, 256>;  // BN*BK

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;           // 0..63
    let ty = tid / 8u;         // 0..7  (row group)
    let tx = tid % 8u;         // 0..7  (col group)
    let wg = wgid.y * nwg.x + wgid.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let row0 = (wg / tiles_n) * BM;
    let col0 = (wg % tiles_n) * BN;

    var acc: array<f32, 16>;   // TM*TN
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    let nchunks = (p.k + BK - 1u) / BK;
    for (var c = 0u; c < nchunks; c = c + 1u) {
        let k0 = c * BK;
        // cooperative load: 256 As + 256 Bs elements, 4 per invocation.
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = tid + e * 64u;   // 0..255
            let r = idx / BK;          // 0..31 (row within tile)
            let kk = idx % BK;         // 0..7  (k within chunk)
            let gk = k0 + kk;
            let ar = row0 + r;
            if (ar < p.m && gk < p.k) { As[idx] = x[ar * p.k + gk]; } else { As[idx] = 0.0; }
            let br = col0 + r;
            if (br < p.n && gk < p.k) { Bs[idx] = w[br * p.k + gk]; } else { Bs[idx] = 0.0; }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            for (var i = 0u; i < TM; i = i + 1u) {
                let a = As[(ty * TM + i) * BK + kk];
                for (var j = 0u; j < TN; j = j + 1u) {
                    acc[i * TN + j] = acc[i * TN + j] + a * Bs[(tx * TN + j) * BK + kk];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < TM; i = i + 1u) {
        let m = row0 + ty * TM + i;
        if (m < p.m) {
            for (var j = 0u; j < TN; j = j + 1u) {
                let nn = col0 + tx * TN + j;
                if (nn < p.n) { out[m * p.n + nn] = acc[i * TN + j]; }
            }
        }
    }
}
