// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Add an IC-LoRA reference's attention bias to materialized scores: lw[q] + lw[k] between generated and reference tokens, masked between earlier conditioning and the reference
// @how   one thread per score
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// An IC-LoRA reference's attention strength (LTX-2's
// `ConditioningItemAttentionStrengthWrapper` + `build_attention_mask`) weights
// how much each appended reference token and the generated tokens attend to
// each other. Tokens `[0, noisy)` are generated, `[noisy, split)` are earlier
// conditioning items, `[split, ..)` the reference. The reference turns its
// `[0, 1]` mask into the additive bias `log(w)` (`f32::MIN` where `w <= 0`).
// Its dense mask is `w[k]` from a generated query to a reference key, `w[q]`
// from a reference query to a generated key, 0 between the reference and the
// earlier items, and 1 everywhere else - so it is the per-token `lw = log w`
// (0 outside the reference) summed over a generated<->reference pair, the
// mask's floor over an earlier-item<->reference pair, and nothing otherwise.
//
//   s:  [heads, nq, nk] scores, already scaled
//   lw: [max(nq, nk)] per-token log weights

struct Params { heads: u32, nq: u32, nk: u32, noisy: u32, split: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lw: array<f32>;
@group(0) @binding(2) var<storage, read_write> s:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.nq * p.nk) { return; }
    let k = idx % p.nk;
    let q = (idx / p.nk) % p.nq;
    let q_ref = q >= p.split;
    if (q_ref != (k >= p.split)) {
        let other = select(q, k, q_ref);
        if (other < p.noisy) {
            s[idx] = s[idx] + lw[q] + lw[k];
        } else {
            s[idx] = s[idx] - 3.4028235e38;
        }
    }
}
