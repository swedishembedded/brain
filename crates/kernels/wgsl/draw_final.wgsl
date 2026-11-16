// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Keyed Gumbel-max draw over logits rows, FINAL stage
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Keyed Gumbel-max draw over logits rows, FINAL stage: one thread per row folds
// the row's P slices from draw_part into the drawn token (the best perturbed
// score, the lower column first among equal scores) and its log-probability
// under the softmax over the survivors,
//
//   logprob = logit_tok / temp - best - ln(sum_t exp(logit_t / temp - best)),
//
// with best the row's maximum of logit/temp. Pass 0 also writes best for the
// min-p pass. A row with no survivor (all NaN or -inf) draws column 0 with
// log-probability 0, as the host sampler's argmax does. Rows whose draw flag is
// 0, and in pass 1 rows without min-p, are left as they were.
//
//   logits: [M, N]
//   spec  : [M, 4] u32 (see draw_part)
//   part  : [M, P, 4]
//   best  : [M]    written by pass 0
//   out   : [M, 2] the drawn column (as f32) and its log-probability
//
// params: m, n, p (slices per row), stage (the pass: 0 or 1)

struct Params {
    m: u32,
    n: u32,
    p: u32,
    stage: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       spec:   array<u32>;
@group(0) @binding(3) var<storage, read>       part:   array<f32>;
@group(0) @binding(4) var<storage, read_write> best:   array<f32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= pr.m) { return; }
    let flag = spec[4u * row + 3u];
    let temp = max(bitcast<f32>(spec[4u * row]), 1e-6);
    let min_p = bitcast<f32>(spec[4u * row + 1u]);
    if (flag == 0u || (pr.stage == 1u && !(min_p > 0.0))) { return; }
    var mx = -3.402823e38;
    for (var c = 0u; c < pr.p; c = c + 1u) {
        mx = max(mx, part[4u * (row * pr.p + c)]);
    }
    var mass = 0.0;
    var bk = -3.402823e38;
    var bi = 0u;
    for (var c = 0u; c < pr.p; c = c + 1u) {
        let b = 4u * (row * pr.p + c);
        let m = part[b + 1u];
        if (m > 0.0) {
            mass = mass + m * exp(part[b] - mx);
        }
        let k = part[b + 2u];
        let i = u32(part[b + 3u]);
        if (k > bk || (k == bk && i < bi)) {
            bk = k;
            bi = i;
        }
    }
    if (pr.stage == 0u) { best[row] = mx; }
    if (!(mass > 0.0)) {
        out[2u * row] = 0.0;
        out[2u * row + 1u] = 0.0;
        return;
    }
    let s = logits[row * pr.n + bi] / temp;
    out[2u * row] = f32(bi);
    out[2u * row + 1u] = (s - mx) - log(mass);
}
