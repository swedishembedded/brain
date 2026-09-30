// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-position cross-entropy AND its softmax row statistics, one WORKGROUP per row - the large-vocab variant of ce_value_masked + ce_stats
// @how   256-thread workgroup tile, online max/sum-exp, 8-step tree reduction
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// For each row `n` of `logits [n_rows, u_bins]`:
//
//   out[n]        = logsumexp(logits[n]) - logits[n, target[n]]   (0 if ignored)
//   stats[2n]     = max_v logits[n, v]                             (0 if ignored)
//   stats[2n + 1] = sum_v exp(logits[n, v] - max)                  (1 if ignored)
//
// exactly the outputs of `ce_value_masked` and `ce_stats`, which walk a row
// with ONE thread each. At a 151936-wide vocab that serial walk is the whole
// cost of both: a thread per row leaves most of every memory sector unused
// and has only the rows themselves as parallelism, and the pair measured at
// ~1% of the card's memory roof, 15% of a Qwen3-0.6B LoRA training step. Here
// 256 threads stride one row (consecutive threads, consecutive columns), each
// keeping an online (max, sum-exp) pair so the row is read once, and the
// pairs are folded by a shared-memory tree. Writing the statistics beside the
// loss lets the backward's `ce_grad_stats` read them instead of walking the
// logits a third time.
//
// Ignored rows keep `ce_stats`' convention (max 0, sum 1), so a gradient
// kernel reading them computes a finite value its own ignore test discards.

const WG: u32 = 256u;

struct Params {
    n_rows: u32,
    u_bins: u32,
    ignore: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> out:     array<f32>;
@group(0) @binding(4) var<storage, read_write> stats:   array<f32>;

var<workgroup> part_m: array<f32, 256>;
var<workgroup> part_s: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index; uniform per workgroup, so both
    // early returns below leave every barrier to all or none of its threads.
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.n_rows) { return; }
    let tgt = targets[row];
    if (tgt == p.ignore) {
        if (t == 0u) {
            out[row] = 0.0;
            stats[2u * row] = 0.0;
            stats[2u * row + 1u] = 1.0;
        }
        return;
    }
    let base = row * p.u_bins;

    // Online (max, sum-exp): one exp per element, rescaling the running sum
    // only when the max moves.
    var m = -3.4e38;
    var s = 0.0;
    for (var c = t; c < p.u_bins; c = c + WG) {
        let x = logits[base + c];
        if (x > m) {
            s = s * exp(m - x) + 1.0;
            m = x;
        } else {
            s = s + exp(x - m);
        }
    }
    part_m[t] = m;
    part_s[t] = s;
    workgroupBarrier();

    // Fold pairs: (m1, s1) + (m2, s2) = (M, s1·e^(m1-M) + s2·e^(m2-M)). A
    // thread whose stride saw no column holds (-3.4e38, 0), which the fold
    // absorbs: its weight e^(-3.4e38 - M) underflows to 0.
    for (var half = WG / 2u; half > 0u; half = half / 2u) {
        if (t < half) {
            let m1 = part_m[t];
            let m2 = part_m[t + half];
            let mm = max(m1, m2);
            part_s[t] = part_s[t] * exp(m1 - mm) + part_s[t + half] * exp(m2 - mm);
            part_m[t] = mm;
        }
        workgroupBarrier();
    }

    if (t == 0u) {
        let mx = part_m[0];
        let sum = part_s[0];
        stats[2u * row] = mx;
        stats[2u * row + 1u] = sum;
        out[row] = (mx + log(sum)) - logits[base + tgt];
    }
}
