// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention backward, step 1, one WORKGROUP per query row - cooperative twin of attn_bwd_dscores
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Attention backward, step 1 - gradient through (probs @ v) and the softmax,
// one WORKGROUP per (b,h,i) query row instead of one THREAD.
//
// attn_bwd_dscores.wgsl gives thread t the whole row t: the causal reduction
// over j is walked serially by a single lane, and its per-iteration read of
// d_out is indexed by the thread-varying row i (`d_out[(b*T+i)*d_model +
// h*hd + d]`), so a warp's 32 lanes read addresses `d_model` floats apart -
// Op::MaxAbsRow's coalescing bug, not merely a slow reduction.
//
// Here 64 threads cooperate on one row: each accumulates a strided partial
// of `dot = sum_j probs[j] * d_prob_j` over j = t, t+64, ... (j <= i), one
// barrier folds the 64 partials (cheaper than a second barrier, and the CPU
// JIT supports exactly one top-level barrier), then every thread recomputes
// d_prob_j for ITS OWN strided j's and writes d_scores[j] directly - no
// second synchronisation needed since each output index is owned by exactly
// one thread. d_out[i,:] is now read by every thread in the workgroup for
// the SAME (b,h,i), which is a workgroup-uniform broadcast rather than a
// strided cross-warp read.
//
// Math is identical to attn_bwd_dscores.wgsl; the `dot` reduction folds in a
// different order (64 partials rather than one running sum), so the two
// agree to floating-point rounding, not to the bit - the same contract
// rmsnorm_rows.wgsl documents for its own reduction-order change.
//
// Dispatch: (bsz * n_heads * tcols) * 64 invocations (one workgroup per row).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,       // T
    head_dim: u32,
    qkv_stride: u32,
    v_off: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_out:    array<f32>;
@group(0) @binding(2) var<storage, read>       qkv:      array<f32>;
@group(0) @binding(3) var<storage, read>       probs:    array<f32>;
@group(0) @binding(4) var<storage, read_write> d_scores: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T;
    if (row >= total) { return; }

    let i = row % T;
    let r = row / T;          // b*n_heads + h
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    let hd = p.head_dim;

    let p_base = (r * T + i) * T;
    let out_base = (b * T + i) * p.d_model + h * hd;

    var acc = 0.0;
    for (var j: u32 = t; j <= i; j = j + 64u) {
        let v_base = (b * T + j) * p.qkv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * qkv[v_base + d];
        }
        acc = acc + probs[p_base + j] * dprob;
    }
    partial[t] = acc;
    workgroupBarrier();
    var dot = 0.0;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        dot = dot + partial[k];
    }

    for (var j: u32 = t; j < T; j = j + 64u) {
        if (j <= i) {
            let v_base = (b * T + j) * p.qkv_stride + p.v_off + h * hd;
            var dprob = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dprob = dprob + d_out[out_base + d] * qkv[v_base + d];
            }
            d_scores[p_base + j] = probs[p_base + j] * (dprob - dot);
        } else {
            d_scores[p_base + j] = 0.0;
        }
    }
}
