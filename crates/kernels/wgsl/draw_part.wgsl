// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Keyed Gumbel-max draw over logits rows, PARTIAL stage
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Keyed Gumbel-max draw over logits rows, PARTIAL stage: each thread reduces
// one strided slice of one row (columns c, c + P, c + 2P, ...) to the slice's
// best perturbed score and its column, and the slice's softmax mass relative to
// its own maximum, so draw_final can fold a row's P slices into the drawn token
// and its log-probability. The drawn token is
//
//   argmax_t (logit_t / temp + g_t),   g_t = -ln(-ln(u_t)),
//
// over the row's survivors, which is a draw from the softmax over them; u_t is
// the keyed uniform model::serve::token_uniform defines (the host reference,
// bit for bit the same integer arithmetic), so a draw depends on the row's key
// (the request's seed and position) and never on its batch slot.
//
// Pass 0 keeps every token of positive weight; pass 1 (the min-p rows of draw flag 1) keeps the tokens whose
// weight exp(logit/temp - best) is at least min_p; pass 2 (the nucleus rows, draw flag 2) keeps the nucleus that
// nucleus_part, nucleus_step and nucleus_cut found (every token above the threshold logit, and the tokens at it up
// to the cut column, lowest column first) and, within it, the min-p floor. best is the row's maximum of
// logit/temp that pass 0's draw_final wrote.
//
//   logits: [M, N]
//   spec  : [M, 8] u32 per row: temp bits, min_p bits, unused, draw flag (0 = not drawn, 1 = drawn over the row,
//           2 = drawn over its nucleus), top_p bits, three unused; then [W, M] the rows' keys of each step of a
//           decode window (step 0 for a single step): a window's steps differ only in their keys
//   state : [M, 4] per row: best logit/temp (written by pass 0's draw_final), the nucleus threshold and the
//           search's upper bound as ordered keys, the cut column (bit patterns, see nucleus_step)
//   part  : [M, P, 4] per (row, slice): [max logit/temp, mass relative to it, best score, its column (as f32)]
//
// params: m, n, p (slices per row), stage (the pass: 0, 1 or 2), step (of the decode window)

struct Params {
    m: u32,
    n: u32,
    p: u32,
    stage: u32,
    step: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       spec:   array<u32>;
@group(0) @binding(3) var<storage, read>       state:  array<f32>;
@group(0) @binding(4) var<storage, read_write> part:   array<f32>;

fn mix32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// The float an ordered key stands for (see nucleus_step).
fn unkey(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) { return bitcast<f32>(k & 0x7fffffffu); }
    return bitcast<f32>(~k);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= pr.m * pr.p) { return; }
    let row = t / pr.p;
    let c = t % pr.p;
    let flag = spec[8u * row + 3u];
    let temp = max(bitcast<f32>(spec[8u * row]), 1e-6);
    let min_p = bitcast<f32>(spec[8u * row + 1u]);
    let key = spec[8u * pr.m + pr.step * pr.m + row];
    if (flag == 0u || (pr.stage == 1u && (flag != 1u || !(min_p > 0.0))) || (pr.stage == 2u && flag != 2u)) { return; }
    var top = 0.0;
    var tau = 0.0;
    var cut = 0u;
    if (pr.stage != 0u) { top = state[4u * row]; }
    if (pr.stage == 2u) {
        tau = unkey(bitcast<u32>(state[4u * row + 1u]));
        cut = bitcast<u32>(state[4u * row + 3u]);
    }
    var mx = -3.402823e38;
    var mass = 0.0;
    var bk = -3.402823e38;
    var bi = 0u;
    for (var i = c; i < pr.n; i = i + pr.p) {
        let l = logits[row * pr.n + i];
        let s = l / temp;
        // NaN and -inf weigh nothing, as in the host sampler.
        if (!(s > -3.402823e38)) { continue; }
        if (pr.stage != 0u) {
            let w = exp(s - top);
            if (!(w > 0.0 && w >= min_p)) { continue; }
        }
        if (pr.stage == 2u && !(l > tau || (l == tau && i <= cut))) { continue; }
        if (s > mx) {
            mass = mass * exp(mx - s) + 1.0;
            mx = s;
        } else {
            mass = mass + exp(s - mx);
        }
        let h = mix32(key ^ mix32(i + 0x9e3779b9u));
        let u = (f32(h >> 9u) + 0.5) * (1.0 / 8388608.0);
        let k = s - log(-log(u));
        if (k > bk) {
            bk = k;
            bi = i;
        }
    }
    part[4u * t] = mx;
    part[4u * t + 1u] = mass;
    part[4u * t + 2u] = bk;
    part[4u * t + 3u] = f32(bi);
}
