// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D neighborhood attention (windowed self-attention), fused: scores, softmax and apply in one pass with an online softmax, no score buffer
// @how   one thread per (query, head), the window streamed through registers
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The LTX diffusion video VAE decoder's NATTEN windowed self-attention: every
// query position (qt,qh,qw) and head attends to exactly `kt*kh*kw` key
// positions, a window that SHIFTS INWARD at a boundary instead of shrinking:
//   half = k/2u (integer division); lo = length - k
//   start = clamp(query - half, 0, lo)
// (NATTEN's own border rule, the reference's `fallback_na.eager._window_bounds`).
// Callers ensure `t>=kt, h>=kh, w>=kw`.
//
// Fused rather than gather-then-dense: a materialized `[heads, nq, window]`
// score buffer grows with the clip times the window (11x11x11 in the last
// stage), which is what bounded the decodable volume. Each thread keeps its
// query and output accumulator in registers and streams the window once with
// an online softmax (running max and sum), so no score ever reaches memory.
// No shared memory and no barrier: it runs unchanged on the CPU JIT.
//
// Q is PRE-SCALED by the caller (`attn.scale = head_dim**-0.5`); no extra
// scale is applied here.
//
//   q, k, v: [t*h*w, heads, head_dim] row-major (query-major, head-minor)
//   out:     [t*h*w, heads, head_dim]
// head_dim <= 128.

const HD: u32 = 128u;

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
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let nq = p.t * p.h * p.w;
    if (idx >= nq * p.heads) { return; }
    let head = idx % p.heads;
    let qi = idx / p.heads;

    let qw = qi % p.w;
    let r2 = qi / p.w;
    let qh = r2 % p.h;
    let qt = r2 / p.h;
    let st_t = u32(clamp(i32(qt) - i32(p.kt / 2u), 0, i32(p.t) - i32(p.kt)));
    let st_h = u32(clamp(i32(qh) - i32(p.kh / 2u), 0, i32(p.h) - i32(p.kh)));
    let st_w = u32(clamp(i32(qw) - i32(p.kw / 2u), 0, i32(p.w) - i32(p.kw)));

    let hd = p.head_dim;
    let q_base = (qi * p.heads + head) * hd;
    var qr: array<f32, 128>;
    var o: array<f32, 128>;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        qr[d] = q[q_base + d];
        o[d] = 0.0;
    }

    var m = -3.4e38;
    var l = 0.0;
    for (var dt: u32 = 0u; dt < p.kt; dt = dt + 1u) {
        for (var dh: u32 = 0u; dh < p.kh; dh = dh + 1u) {
            for (var dw: u32 = 0u; dw < p.kw; dw = dw + 1u) {
                let ki = ((st_t + dt) * p.h + st_h + dh) * p.w + st_w + dw;
                let kv_base = (ki * p.heads + head) * hd;
                var s = 0.0;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    s = s + qr[d] * k[kv_base + d];
                }
                let m_new = max(m, s);
                let corr = exp(m - m_new);
                let pj = exp(s - m_new);
                l = l * corr + pj;
                for (var d: u32 = 0u; d < hd; d = d + 1u) {
                    o[d] = o[d] * corr + pj * v[kv_base + d];
                }
                m = m_new;
            }
        }
    }

    let inv = 1.0 / l;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        out[q_base + d] = o[d] * inv;
    }
}
