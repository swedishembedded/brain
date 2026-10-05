// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of ct_state_scan: gradients of the visit inputs, the gates and each subject's share of the rate and population-state gradients
// @how   one thread per (subject, channel), a serial loop back over that subject's visits
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The exact adjoint of ct_state_scan, from dz and the states hs it saved (the
// state before visit v is hs of the previous slot, or m before the first):
//   z = m + A (h - m), A = exp(-r T):    dh = A dz, dm += (1 - A) dz, dr += -T A (h - m) dz
//   per present visit, back to front, with q = m + a (h_prev - m), h = q + i (x - q):
//     dx = i dh;  dpre = (x - q) dh * i (1 - i);  dq = (1 - i) dh
//     dh_prev = a dq;  dm += (1 - a) dq;  dr += -gap a (h_prev - m) dq
//   finally dm += dh (the state starts at m).
// d_xg is written (absent visits get zero); part[b, c] = dr * sigmoid(state[c])
// (the gradient of the raw rate through softplus) and part[b, D + c] = dm are
// each subject's share, summed over subjects by the caller.

struct Params {
    subjects: u32,
    visits: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xg:    array<f32>;
@group(0) @binding(2) var<storage, read>       dt:    array<f32>;
@group(0) @binding(3) var<storage, read>       state: array<f32>;
@group(0) @binding(4) var<storage, read>       hs:    array<f32>;
@group(0) @binding(5) var<storage, read>       dz:    array<f32>;
@group(0) @binding(6) var<storage, read_write> d_xg:  array<f32>;
@group(0) @binding(7) var<storage, read_write> part:  array<f32>;

fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let d = p.d;
    if (idx >= p.subjects * d) { return; }
    let b = idx / d;
    let c = idx % d;
    let r = softplus(state[c]);
    let m = state[d + c];
    let vs = p.visits;
    let last = b * vs + vs - 1u;
    var h_last = m;
    if (vs > 0u) { h_last = hs[last * d + c]; }
    let tail = dt[p.subjects * vs + b];
    let big_a = exp(-r * tail);
    var dh = big_a * dz[idx];
    var dm = (1.0 - big_a) * dz[idx];
    var dr = -tail * big_a * (h_last - m) * dz[idx];
    for (var k = 0u; k < vs; k = k + 1u) {
        let v = vs - 1u - k;
        let s = b * vs + v;
        let gap = dt[s];
        if (gap < 0.0) {
            d_xg[s * 2u * d + c] = 0.0;
            d_xg[s * 2u * d + d + c] = 0.0;
        } else {
            var h_prev = m;
            if (v > 0u) { h_prev = hs[(s - 1u) * d + c]; }
            let a = exp(-r * gap);
            let q = m + a * (h_prev - m);
            let x = xg[s * 2u * d + c];
            let i = 1.0 / (1.0 + exp(-xg[s * 2u * d + d + c]));
            d_xg[s * 2u * d + c] = i * dh;
            d_xg[s * 2u * d + d + c] = (x - q) * dh * i * (1.0 - i);
            let dq = (1.0 - i) * dh;
            dm = dm + (1.0 - a) * dq;
            dr = dr - gap * a * (h_prev - m) * dq;
            dh = a * dq;
        }
    }
    dm = dm + dh;
    part[b * 2u * d + c] = dr / (1.0 + exp(-state[c]));
    part[b * 2u * d + d + c] = dm;
}
