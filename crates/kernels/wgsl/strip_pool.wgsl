// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Strip pooling: mean over ONE spatial axis, NCHW.
//   axis = 0: mean over W  ->  y : [N, C, H, 1]   (one invocation per (n,c,h))
//   axis = 1: mean over H  ->  y : [N, C, 1, W]   (one invocation per (n,c,w))
//
// ZipDepth's StripPoolingAttention takes both and broadcast-adds them:
//   gate = sigmoid(BN(dwconv1x1( x.mean(W) + x.mean(H) )))
// One kernel with an axis switch rather than two near-identical files; the
// reduction is serial within an invocation, so no atomics and no barrier — each
// output row/column is owned by exactly one thread.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    axis: u32,   // 0 = mean over W (keep H), 1 = mean over H (keep W)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let keep = select(p.W, p.H, p.axis == 0u);   // length of the surviving axis
    let total = p.N * p.C * keep;
    if (idx >= total) { return; }

    let k  = idx % keep;
    let t1 = idx / keep;
    let c  = t1 % p.C;
    let n  = t1 / p.C;
    let base = (n * p.C + c) * p.H;

    var acc = 0.0;
    if (p.axis == 0u) {
        // k is h; average the row.
        for (var wi: u32 = 0u; wi < p.W; wi = wi + 1u) {
            acc = acc + x[(base + k) * p.W + wi];
        }
        y[idx] = acc / f32(p.W);
    } else {
        // k is w; average the column.
        for (var hi: u32 = 0u; hi < p.H; hi = hi + 1u) {
            acc = acc + x[(base + hi) * p.W + k];
        }
        y[idx] = acc / f32(p.H);
    }
}
