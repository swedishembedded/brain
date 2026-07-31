// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Generic KxK max-pool INPUT gradient, NCHW, arbitrary STRIDE + symmetric pad.
// GATHER form (no scatter / no atomics).
//   dy     : [N, C, Ho, Wo]   idx = ((n*C + c)*Ho + ho)*Wo + wo
//   argmax : [N, C, Ho, Wo]   the forward's side-output, same indexing as dy
//   dx     : [N, C, H,  W ]   read_write, idx = ((n*C + c)*H + hi)*W + wi
//
// One invocation per INPUT element (n, c, hi, wi) with flat index `ii`, so every
// dx[ii] is written by exactly one invocation and nothing is accumulated across
// invocations. We sum dy from every output position whose KxK window covers
// (hi,wi) AND selected this input as its max (argmax == ii). Terminal write is a
// plain overwrite; dx does NOT need pre-zeroing.
//
// The generalization of maxpool5_dx.wgsl (that kernel is this one pinned at
// stride=1). Coverage, inverting maxpool2d.wgsl's window
// `[ho*stride - pad, ho*stride - pad + K - 1]`:
//   ho*stride <= hi + pad   and   ho*stride >= hi + pad - K + 1
// so, with hp = hi + pad,
//   ho in [ ceil((hp - K + 1)/stride) , floor(hp/stride) ]  clamped to [0, Ho).
// That set is CONTIGUOUS in ho at every stride — a max-pool tap is a window, not
// a point, so unlike conv2d_gd_dx there is no `(...) % stride == 0` divisibility
// test to apply. At stride > K the interval is simply empty for the input pixels
// no window reaches, and those correctly receive dx = 0. Both bounds are computed
// in i32 because `hp - K + 1` is routinely negative near the top-left border;
// doing it in u32 wraps and silently pulls in the whole first output row.
//
// Ties: the forward records ONE winner, so a plateau sends all of dy to that
// single input — brain's frozen-argmax convention, identical to maxpool5.
//
// All-padding windows (pad >= K only): the forward writes argmax = 0 there
// without having selected anything. That pointer is never believed here, because
// the coverage interval is what gates the read: an output covers input (0,0) on
// BOTH axes only if row 0 and col 0 lie inside its window, and such a window has
// an in-bounds tap, so its argmax is real. Every fabricated pointer therefore
// sits outside every input's interval and its dy is dropped — the correct
// derivative of a constant output. This is the one regime where the gather form
// deliberately differs from a naive scatter (`dx[argmax[o]] += dy[o]`), which
// would dump that dy on input 0.

struct Params {
    N:      u32,
    C:      u32,
    H:      u32,
    W:      u32,
    K:      u32,
    stride: u32,
    pad:    u32,
    Ho:     u32,
    Wo:     u32,
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

    // Decompose input flat index into (n, c, hi, wi).
    let wi = ii % p.W;
    let t1 = ii / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let s = i32(p.stride);
    let kk = i32(p.K);

    // Output rows/cols whose window covers this input pixel.
    let hp = i32(hi) + i32(p.pad);
    var ho_lo: i32 = 0;
    let th = hp - kk + 1;
    if (th > 0) { ho_lo = (th + s - 1) / s; }   // ceil(th / stride), th > 0
    let ho_hi = min(hp / s, i32(p.Ho) - 1);

    let wp = i32(wi) + i32(p.pad);
    var wo_lo: i32 = 0;
    let tw = wp - kk + 1;
    if (tw > 0) { wo_lo = (tw + s - 1) / s; }
    let wo_hi = min(wp / s, i32(p.Wo) - 1);

    var acc: f32 = 0.0;
    for (var ho: i32 = ho_lo; ho <= ho_hi; ho = ho + 1) {
        for (var wo: i32 = wo_lo; wo <= wo_hi; wo = wo + 1) {
            let oi = ((n * p.C + c) * p.Ho + u32(ho)) * p.Wo + u32(wo);
            if (u32(argmax[oi]) == ii) {
                acc = acc + dy[oi];
            }
        }
    }
    dx[ii] = acc;
}
