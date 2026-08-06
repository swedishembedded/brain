// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Convex 3x3 upsample forward (ZipDepth's FastConvexUpsample, unfold path)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Convex 3x3 upsample forward (ZipDepth's FastConvexUpsample, unfold path).
//   mask : [N, 9*S*S, H, W]   already softmax-normalized over the 9 axis
//   d    : [N, 1,     H, W]   the half-resolution depth map
//   y    : [N, 1,     H*S, W*S]   one invocation per OUTPUT element
//
//   y[2h+sh, 2w+sw] = sum_{k=0..8} mask[k, sh*S+sw, h, w] * d[clamp(h+dy_k),
//                                                             clamp(w+dx_k)]
// with k -> (dy, dx) = (k/3 - 1, k%3 - 1) and clamp = replicate padding, which is
// exactly F.pad(d, 1, mode='replicate') + F.unfold(.,3) in the reference.
//
// The mask's softmax over the 9 neighbours makes each output a CONVEX combination
// of the 3x3 input neighbourhood (weights >= 0 summing to 1) — hence the name,
// and hence the output cannot overshoot the local depth range.
//
// The reference fuses this with a pixel_shuffle; here the shuffle is folded into
// the output index (sub-pixel c = sh*S + sw), so no separate shuffle dispatch and
// no [N,9,S*S,H,W] intermediate. mask's channel layout is CRD-compatible with
// pixel_shuffle.wgsl: channel = k*(S*S) + (sh*S + sw).
//
// Nothing here needs floor/fract: the neighbour offsets are integers and the
// replicate clamp is integer min/max, so this compiles on both backends.

struct Params {
    N: u32,
    H: u32,
    W: u32,
    S: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mask: array<f32>;
@group(0) @binding(2) var<storage, read>       d:    array<f32>;
@group(0) @binding(3) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let Ho = p.H * p.S;
    let Wo = p.W * p.S;
    let total = p.N * Ho * Wo;
    if (idx >= total) { return; }

    let wo = idx % Wo;
    let t1 = idx / Wo;
    let ho = t1 % Ho;
    let n  = t1 / Ho;

    let h  = ho / p.S;
    let sh = ho % p.S;
    let w  = wo / p.S;
    let sw = wo % p.S;
    let sub = sh * p.S + sw;          // sub-pixel index in [0, S*S)
    let ss  = p.S * p.S;

    var acc = 0.0;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        // Neighbour offset, replicate-clamped to the map (integer arithmetic:
        // guard the low side before subtracting, then min the high side).
        let kh = k / 3u;
        let kw = k % 3u;
        var hi = h;
        if (kh == 0u) { if (h > 0u) { hi = h - 1u; } } else if (kh == 2u) { hi = min(h + 1u, p.H - 1u); }
        var wi = w;
        if (kw == 0u) { if (w > 0u) { wi = w - 1u; } } else if (kw == 2u) { wi = min(w + 1u, p.W - 1u); }

        let m_idx = ((n * (9u * ss) + (k * ss + sub)) * p.H + h) * p.W + w;
        let d_idx = (n * p.H + hi) * p.W + wi;
        acc = acc + mask[m_idx] * d[d_idx];
    }
    y[idx] = acc;
}
