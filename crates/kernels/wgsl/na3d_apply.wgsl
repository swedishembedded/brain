// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D neighborhood-attention (windowed, self-attention) probs*V apply
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The `na3d_scores.wgsl` twin: `out[q,head,d] = sum_w probs[head,q,w] *
// v[key(q,w),head,d]`, `key(q,w)` re-deriving the SAME per-axis
// inward-shifting window start `na3d_scores.wgsl`'s header documents (cheap
// arithmetic, recomputed rather than staged through a side buffer).
//
//   probs: [heads, t*h*w, kt*kh*kw]   row-major (post row-softmax)
//   v     : [t*h*w, heads, head_dim]  row-major
//   out   : [t*h*w, heads*head_dim]   row-major (channels-last, ready for
//           the output projection - matches `attn.forward`'s own
//           `out.reshape(batch,t,h,w,dim)` before `proj`)
// One invocation per (query, head, dim).

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
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v:     array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let nq = p.t * p.h * p.w;
    let hd = p.head_dim;
    let total = nq * p.heads * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let head = r1 % p.heads;
    let qi = r1 / p.heads;

    let qw = qi % p.w;
    let r2 = qi / p.w;
    let qh = r2 % p.h;
    let qt = r2 / p.h;

    let half_t = i32(p.kt / 2u);
    let lo_t = i32(p.t) - i32(p.kt);
    let st_t = clamp(i32(qt) - half_t, 0, lo_t);
    let half_h = i32(p.kh / 2u);
    let lo_h = i32(p.h) - i32(p.kh);
    let st_h = clamp(i32(qh) - half_h, 0, lo_h);
    let half_w = i32(p.kw / 2u);
    let lo_w = i32(p.w) - i32(p.kw);
    let st_w = clamp(i32(qw) - half_w, 0, lo_w);

    let window = p.kt * p.kh * p.kw;
    let p_base = (head * nq + qi) * window;

    var acc = 0.0;
    var widx: u32 = 0u;
    for (var dt: u32 = 0u; dt < p.kt; dt = dt + 1u) {
        let kt_idx = u32(st_t) + dt;
        for (var dh: u32 = 0u; dh < p.kh; dh = dh + 1u) {
            let kh_idx = u32(st_h) + dh;
            for (var dw: u32 = 0u; dw < p.kw; dw = dw + 1u) {
                let kw_idx = u32(st_w) + dw;
                let ki = (kt_idx * p.h + kh_idx) * p.w + kw_idx;
                let v_val = v[(ki * p.heads + head) * hd + d];
                acc = acc + probs[p_base + widx] * v_val;
                widx = widx + 1u;
            }
        }
    }
    out[qi * (p.heads * hd) + head * hd + d] = acc;
}
