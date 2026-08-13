// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Convex 3x3 upsample: gradient wrt the MASK
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Convex 3x3 upsample: gradient wrt the MASK.
//   dy   : [N, 1,     H*S, W*S]
//   d    : [N, 1,     H,   W  ]   the half-res depth (forward input)
//   dmask: [N, 9*S*S, H,   W  ]   read_write (one invocation per MASK element)
//
//   dmask[k, sub, h, w] = dy[h*S + sh, w*S + sw] * d[clamp(h+dy_k), clamp(w+dx_k)]
//
// Each mask element feeds exactly ONE output (its own sub-pixel), so this is a
// pure gather with no accumulation — the easy half of the backward.
// `sub = sh*S + sw` inverts to the output pixel (h*S + sh, w*S + sw).

struct Params {
    N: u32,
    H: u32,
    W: u32,
    S: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:    array<f32>;
@group(0) @binding(2) var<storage, read>       d:     array<f32>;
@group(0) @binding(3) var<storage, read_write> dmask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let ss = p.S * p.S;
    let total = p.N * (9u * ss) * p.H * p.W;
    if (idx >= total) { return; }

    // Decode mask coordinate (n, ch = k*ss + sub, h, w).
    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let t2 = t1 / p.H;
    let ch = t2 % (9u * ss);
    let n  = t2 / (9u * ss);

    let k   = ch / ss;
    let sub = ch % ss;
    let sh  = sub / p.S;
    let sw  = sub % p.S;

    // The neighbour this mask weight multiplies (same clamp as the forward).
    let kh = k / 3u;
    let kw = k % 3u;
    var hi = h;
    if (kh == 0u) { if (h > 0u) { hi = h - 1u; } } else if (kh == 2u) { hi = min(h + 1u, p.H - 1u); }
    var wi = w;
    if (kw == 0u) { if (w > 0u) { wi = w - 1u; } } else if (kw == 2u) { wi = min(w + 1u, p.W - 1u); }

    let Wo = p.W * p.S;
    let dy_idx = (n * (p.H * p.S) + (h * p.S + sh)) * Wo + (w * p.S + sw);
    let d_idx  = (n * p.H + hi) * p.W + wi;
    dmask[idx] = dy[dy_idx] * d[d_idx];
}
