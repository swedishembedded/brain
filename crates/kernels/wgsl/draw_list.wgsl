// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Keyed Gumbel-max draw of a top-k row from its best-first candidate list, on the device
// @how   one thread per row
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The draw a row that keeps its best top_k tokens makes from the candidates topk_rows took (best first, equal
// logits lowest id first), so a top-k row needs no host step: the same truncation and the same keyed noise as the
// host reference (model::serve::draw_list over the row's first top_k candidates), operation for operation.
//
//   p_c    = exp(v_c / temp - max_c'(v_c' / temp))                 over the first top_k candidates
//   cut    = the shortest prefix whose p sum reaches top_p of the list's (when 0 < top_p < 1)
//   floor  = the leading p_c >= p_0 * min_p within the cut (at least one)
//   token  = idx of argmax_c (ln p_c + gumbel(key, idx_c)) over the survivors of positive weight
//   logp   = ln p_token - ln(sum of the survivors' p)
//
// The token (as f32) and its log-probability go to out[4 r], out[4 r + 1]; the row's best and mass that pass 0 of
// draw_final wrote (out[4 r + 2], out[4 r + 3]) are left for the whole-row figure. Rows that do not keep a top_k
// inside the vocabulary (or one wider than the list) are left alone.
//
//   spec : [M, 8] u32 per row (see draw_part; slot 2 holds the row's top_k), then [W, M] the keys of each step
//   vals : [M, n] candidate logits, best first
//   idx  : [M, n] their token ids (as f32)
//   out  : [M, 4] see draw_final
//
// params: m, n (candidates per row), vocab, step (of the decode window)

struct Params {
    m: u32,
    n: u32,
    vocab: u32,
    step: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       spec: array<u32>;
@group(0) @binding(2) var<storage, read>       vals: array<f32>;
@group(0) @binding(3) var<storage, read>       idx:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

fn mix32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = gid.y * (nwg.x * 64u) + gid.x;
    if (r >= pr.m) { return; }
    let flag = spec[8u * r + 3u];
    let k = spec[8u * r + 2u];
    if (flag == 0u || k == 0u || k >= pr.vocab || k > pr.n) { return; }
    let temp = max(bitcast<f32>(spec[8u * r]), 1e-6);
    let min_p = bitcast<f32>(spec[8u * r + 1u]);
    let top_p = bitcast<f32>(spec[8u * r + 4u]);
    let key = spec[8u * pr.m + pr.step * pr.m + r];
    let base = r * pr.n;

    // The host's fold(-inf, f32::max): a NaN never wins.
    var mx = bitcast<f32>(0xff800000u);
    for (var c = 0u; c < k; c = c + 1u) {
        let s = vals[base + c] / temp;
        if (s > mx) { mx = s; }
    }
    var sum = 0.0;
    for (var c = 0u; c < k; c = c + 1u) {
        sum = sum + exp(vals[base + c] / temp - mx);
    }
    var cut = k;
    if (top_p > 0.0 && top_p < 1.0 && sum > 0.0) {
        var kept = 0.0;
        for (var c = 0u; c < k; c = c + 1u) {
            kept = kept + exp(vals[base + c] / temp - mx);
            if (kept / sum >= top_p) {
                cut = c + 1u;
                break;
            }
        }
    }
    if (min_p > 0.0) {
        let floor = exp(vals[base] / temp - mx) * min_p;
        var keep = 0u;
        for (var c = 0u; c < cut; c = c + 1u) {
            if (!(exp(vals[base + c] / temp - mx) >= floor)) { break; }
            keep = c + 1u;
        }
        cut = max(keep, 1u);
    }
    var kept_sum = 0.0;
    for (var c = 0u; c < cut; c = c + 1u) {
        kept_sum = kept_sum + exp(vals[base + c] / temp - mx);
    }
    var pick = 0u;
    var top = bitcast<f32>(0xff800000u);
    for (var c = 0u; c < cut; c = c + 1u) {
        let w = exp(vals[base + c] / temp - mx);
        if (!(w > 0.0)) { continue; }
        let t = u32(idx[base + c]);
        let h = mix32(key ^ mix32(t + 0x9e3779b9u));
        let u = (f32(h >> 9u) + 0.5) * (1.0 / 8388608.0);
        let score = log(w) - log(-log(u));
        if (score > top) {
            top = score;
            pick = c;
        }
    }
    out[4u * r] = idx[base + pick];
    out[4u * r + 1u] = log(exp(vals[base + pick] / temp - mx)) - log(kept_sum);
}
