// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Convex 3x3 upsample: gradient wrt the half-res DEPTH map
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Convex 3x3 upsample: gradient wrt the half-res DEPTH map.
//   dy   : [N, 1,     H*S, W*S]
//   mask : [N, 9*S*S, H,   W  ]
//   dd   : [N, 1,     H,   W  ]   read_write (one invocation per INPUT element)
//
// The hard half. Each depth pixel is read by up to 9*S*S outputs (every
// neighbour position of every sub-pixel of every adjacent half-res cell), so the
// adjoint is a scatter. Inverted to a GATHER, as everywhere else in brain: this
// pixel sums over the 3x3 block of half-res cells that could reference it, times
// their S*S sub-pixels.
//
//   dd[h,w] = sum over neighbours (hn,wn) of the 3x3 block around (h,w),
//             over k such that clamp(hn + dy_k) == h and clamp(wn + dx_k) == w,
//             over sub in [0,S*S):   dy[hn*S + sh, wn*S + sw] * mask[k, sub, hn, wn]
//
// The reference cell (hn,wn) contributes through the OPPOSITE offset: if cell
// (h-1,w) reads its k with dy_k = +1, that lands on h. Rather than invert the
// clamp analytically — which is where replicate padding gets subtle, because the
// border pixels are referenced through MULTIPLE k at once — each candidate cell
// re-evaluates the forward's own clamp and tests it. The adjoint then holds by
// construction, the same discipline as resize_bilinear_dx.wgsl.
//
// Bounded: at most 9 candidate cells x 9 k x S*S sub-pixels.

struct Params {
    N: u32,
    H: u32,
    W: u32,
    S: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> dd:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.H * p.W;
    if (idx >= total) { return; }

    let w  = idx % p.W;
    let t1 = idx / p.W;
    let h  = t1 % p.H;
    let n  = t1 / p.H;

    let ss = p.S * p.S;
    let Ho = p.H * p.S;
    let Wo = p.W * p.S;

    var acc = 0.0;
    // Candidate referencing cells: the 3x3 block around (h,w), clamped to the map.
    var hn_lo = h; if (h > 0u) { hn_lo = h - 1u; }
    let hn_hi = min(h + 1u, p.H - 1u);
    var wn_lo = w; if (w > 0u) { wn_lo = w - 1u; }
    let wn_hi = min(w + 1u, p.W - 1u);

    for (var hn: u32 = hn_lo; hn <= hn_hi; hn = hn + 1u) {
        for (var wn: u32 = wn_lo; wn <= wn_hi; wn = wn + 1u) {
            for (var k: u32 = 0u; k < 9u; k = k + 1u) {
                // Re-evaluate the FORWARD's clamped neighbour for cell (hn,wn).
                let kh = k / 3u;
                let kw = k % 3u;
                var hi = hn;
                if (kh == 0u) { if (hn > 0u) { hi = hn - 1u; } } else if (kh == 2u) { hi = min(hn + 1u, p.H - 1u); }
                var wi = wn;
                if (kw == 0u) { if (wn > 0u) { wi = wn - 1u; } } else if (kw == 2u) { wi = min(wn + 1u, p.W - 1u); }
                if (hi == h && wi == w) {
                    for (var sub: u32 = 0u; sub < ss; sub = sub + 1u) {
                        let sh = sub / p.S;
                        let sw = sub % p.S;
                        let dy_idx = (n * Ho + (hn * p.S + sh)) * Wo + (wn * p.S + sw);
                        let m_idx = ((n * (9u * ss) + (k * ss + sub)) * p.H + hn) * p.W + wn;
                        acc = acc + dy[dy_idx] * mask[m_idx];
                    }
                }
            }
        }
    }
    dd[idx] = acc;
}
