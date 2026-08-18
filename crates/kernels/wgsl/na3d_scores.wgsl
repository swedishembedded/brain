// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D neighborhood-attention (windowed, self-attention) QK scores
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// 3D neighborhood-attention QK scores (LTX diffusion video VAE decoder's
// NATTEN windowed self-attention, gather-then-dense form): for every query
// position (qt,qh,qw) and head, dot Q against every K position inside a
// window of exactly `kt*kh*kw` positions. The window is ALWAYS full size
// (never shrunk/masked) - near a boundary it SHIFTS INWARD instead, so every
// query always attends to exactly `kt*kh*kw` keys:
//   half = k/2u (integer division); lo = length - k
//   start = clamp(query - half, 0, lo)
// This is NATTEN's own border convention (verified against the reference's
// pure-torch fallback, `ltx_core...transformer.fallback_na.eager._window_bounds`
// - the fallback's `_pick_tiles`/grouping machinery is a batching optimization
// over the SAME per-axis bounds, not a different definition of them). Callers
// must ensure `t>=kt, h>=kh, w>=kw` (the reference's own
// `NeighborhoodAttention3D.forward` raises `ValueError` otherwise on the
// production NATTEN/eager-fallback path - there is no smaller-than-kernel
// case to support here).
//
// Self-attention: Q and K share ONE (t,h,w) volume (no separate K grid).
// Q is assumed PRE-SCALED by the caller (`attn.scale = head_dim**-0.5`,
// applied once after Q's RMSNorm - matches the reference passing `scale=1.0`
// into its own attention backend since it scales Q up front); this kernel
// applies no additional scale.
//
//   q, k  : [t*h*w, heads, head_dim]  row-major (query-major, head-minor)
//   scores: [heads, t*h*w, kt*kh*kw]  row-major
// One invocation per (head, query, window-relative index).

struct Params {
    t: u32,
    h: u32,
    w: u32,
    heads: u32,
    head_dim: u32,
    kt: u32,
    kh: u32,
    kw: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let nq = p.t * p.h * p.w;
    let window = p.kt * p.kh * p.kw;
    let total = p.heads * nq * window;
    let idx = gidx;
    if (idx >= total) { return; }

    let widx = idx % window;
    let r1 = idx / window;
    let qi = r1 % nq;
    let head = r1 / nq;

    // Decode query flat index -> (qt, qh, qw).
    let qw = qi % p.w;
    let r2 = qi / p.w;
    let qh = r2 % p.h;
    let qt = r2 / p.h;

    // Decode window-relative index -> (dt, dh, dw).
    let dw = widx % p.kw;
    let r3 = widx / p.kw;
    let dh = r3 % p.kh;
    let dt = r3 / p.kh;

    // Per-axis inward-shifting window start (NATTEN's own border rule).
    let half_t = i32(p.kt / 2u);
    let lo_t = i32(p.t) - i32(p.kt);
    let st_t = clamp(i32(qt) - half_t, 0, lo_t);
    let half_h = i32(p.kh / 2u);
    let lo_h = i32(p.h) - i32(p.kh);
    let st_h = clamp(i32(qh) - half_h, 0, lo_h);
    let half_w = i32(p.kw / 2u);
    let lo_w = i32(p.w) - i32(p.kw);
    let st_w = clamp(i32(qw) - half_w, 0, lo_w);

    let kt_idx = u32(st_t) + dt;
    let kh_idx = u32(st_h) + dh;
    let kw_idx = u32(st_w) + dw;
    let ki = (kt_idx * p.h + kh_idx) * p.w + kw_idx;

    let hd = p.head_dim;
    let q_base = (qi * p.heads + head) * hd;
    let k_base = (ki * p.heads + head) * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[idx] = s;
}
