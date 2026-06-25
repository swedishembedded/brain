// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 5x5 max-pool backward, GATHER form (no scatter / no atomics). One invocation
// per INPUT element (n,c,hi,wi) with flat index `ii`. We accumulate dy from
// every output position whose KxK window covers (hi,wi) AND selected this input
// as its max (argmax == ii).
//
// Window coverage: output (ho,wo) has window input-rows [ho-pad, ho+pad]. So
// (hi,wi) is covered by ho exactly when ho-pad <= hi <= ho+pad, i.e.
//   ho in [hi-pad, hi+pad]  (clamped to [0,H)),  and symmetrically for wo.
// That spans at most K values in each axis. Stride is 1 so there are no holes.

struct Params {
    N:   u32,
    C:   u32,
    H:   u32,
    W:   u32,
    K:   u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:     array<f32>;
@group(0) @binding(2) var<storage, read>       argmax: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let ii = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (ii >= total) { return; }

    // Decompose input flat index into (n,c,hi,wi).
    let wi = ii % p.W;
    let t1 = ii / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // Output rows covering this input row: [hi-pad, hi+pad] clamped to [0,H).
    let ho_lo = max(i32(hi) - i32(p.pad), 0);
    let ho_hi = min(i32(hi) + i32(p.pad), i32(p.H) - 1);
    let wo_lo = max(i32(wi) - i32(p.pad), 0);
    let wo_hi = min(i32(wi) + i32(p.pad), i32(p.W) - 1);

    var acc: f32 = 0.0;
    for (var ho: i32 = ho_lo; ho <= ho_hi; ho = ho + 1) {
        for (var wo: i32 = wo_lo; wo <= wo_hi; wo = wo + 1) {
            let oi = ((n * p.C + c) * p.H + u32(ho)) * p.W + u32(wo);
            if (u32(argmax[oi]) == ii) {
                acc = acc + dy[oi];
            }
        }
    }
    dx[ii] = acc;
}
