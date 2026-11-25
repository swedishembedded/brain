// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Keyed Gumbel-max draw of a top-k row from its best-first candidate list, a workgroup a row
// @how   one workgroup per row: the candidates' weights and keyed scores in parallel, the sums that decide the cut on one thread
// @opt   4
// @cpu   no
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
// The candidates' scaled logits, weights and keyed scores are computed one a thread and reduced (a maximum and an
// argmax, the lowest candidate on equal scores, which no order changes); the sums that decide the cut and the
// log-probability are added on one thread in the candidates' order over the shared weights, so every figure is the
// serial reference's to the bit (draw_list, one thread a row, is that reference on the device: 34 us for 32 rows of 50
// candidates on a GH200, a thread walking its list five times). Lists of at most MAXC (the drawer's widest) candidates;
// a device without workgroup reductions runs draw_list.
//
// The token (as f32) and its log-probability go to out[4 r], out[4 r + 1]; the row's best and mass that pass 0 of
// draw_final wrote (out[4 r + 2], out[4 r + 3]) are left for the whole-row figure. The number of candidates that
// survived the cut, the support the token was drawn from, goes to kept[r].
//
// A row with no top_k inside the vocabulary whose spec slot 5 is a list length L (a nucleus or min-p row that reports
// its support) is drawn the same way from its first min(L, n) candidates, with the nucleus measured against the
// whole row's mass (out[4 r + 3], scaled from the row's best to the list's), as draw_list draws it. Other rows (no
// top_k and no list length, or a top_k wider than the list) are left alone.
//
//   spec : [M, 8] u32 per row (see draw_part; slot 2 holds the row's top_k, slot 5 a list length or 0), then [W, M]
//          the keys of each step
//   vals : [M, n] candidate logits, best first
//   idx  : [M, n] their token ids (as f32)
//   out  : [M, 4] see draw_final
//   kept : [M] the survivors of the cut (as f32)
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
@group(0) @binding(5) var<storage, read_write> kept: array<f32>;

const WG: u32 = 64u;
const MAXC: u32 = 1024u;

var<workgroup> w: array<f32, 1024>;
var<workgroup> red_v: array<f32, 64>;
var<workgroup> red_i: array<u32, 64>;
var<workgroup> cut_s: u32;
var<workgroup> kept_s: f32;

fn mix32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

fn score_of(key: u32, wc: f32, t: u32) -> f32 {
    let h = mix32(key ^ mix32(t + 0x9e3779b9u));
    let u = (f32(h >> 9u) + 0.5) * (1.0 / 8388608.0);
    return log(wc) - log(-log(u));
}

// One step of the maximum's tree: a NaN never replaces a value.
fn max_step(lid: u32, o: u32) {
    if (lid < o && red_v[lid + o] > red_v[lid]) { red_v[lid] = red_v[lid + o]; }
    workgroupBarrier();
}

// One step of the argmax's tree: the larger score, the lower candidate on equal ones.
fn arg_step(lid: u32, o: u32) {
    if (lid < o) {
        let ov = red_v[lid + o];
        let oi = red_i[lid + o];
        if (ov > red_v[lid] || (ov == red_v[lid] && oi < red_i[lid])) {
            red_v[lid] = ov;
            red_i[lid] = oi;
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let r = wid.y * nwg.x + wid.x;
    // Every branch below is uniform over the workgroup: it depends on the row alone.
    if (r >= pr.m) { return; }
    let flag = spec[8u * r + 3u];
    let k = spec[8u * r + 2u];
    if (flag == 0u) { return; }
    let list = spec[8u * r + 5u];
    let whole_row = k == 0u || k >= pr.vocab;
    if (whole_row && list == 0u) { return; }
    if (!whole_row && k > pr.n) { return; }
    var len = k;
    if (whole_row) { len = min(list, pr.n); }
    if (len > MAXC) { return; }
    let temp = max(bitcast<f32>(spec[8u * r]), 1e-6);
    let min_p = bitcast<f32>(spec[8u * r + 1u]);
    let top_p = bitcast<f32>(spec[8u * r + 4u]);
    let key = spec[8u * pr.m + pr.step * pr.m + r];
    let base = r * pr.n;
    let ninf = bitcast<f32>(0xff800000u);

    // The maximum of the scaled logits: the host's fold(-inf, f32::max), so a NaN never wins.
    var mx = ninf;
    for (var c = lid; c < len; c = c + WG) {
        let s = vals[base + c] / temp;
        if (s > mx) { mx = s; }
    }
    red_v[lid] = mx;
    workgroupBarrier();
    // Unrolled: the generated CUDA tier takes no barrier inside a loop.
    max_step(lid, 32u);
    max_step(lid, 16u);
    max_step(lid, 8u);
    max_step(lid, 4u);
    max_step(lid, 2u);
    max_step(lid, 1u);
    mx = red_v[0];
    workgroupBarrier();

    for (var c = lid; c < len; c = c + WG) {
        w[c] = exp(vals[base + c] / temp - mx);
    }
    workgroupBarrier();
    if (lid == 0u) {
        var sum = 0.0;
        for (var c = 0u; c < len; c = c + 1u) { sum = sum + w[c]; }
        var cut = len;
        // The nucleus against the list a top_k keeps, or against the whole row for a listed row.
        var denom = sum;
        if (whole_row) { denom = out[4u * r + 3u] * exp(out[4u * r + 2u] - mx); }
        if (top_p > 0.0 && top_p < 1.0 && denom > 0.0) {
            var acc = 0.0;
            for (var c = 0u; c < len; c = c + 1u) {
                acc = acc + w[c];
                if (acc / denom >= top_p) {
                    cut = c + 1u;
                    break;
                }
            }
        }
        if (min_p > 0.0) {
            let floor = w[0] * min_p;
            var keep = 0u;
            for (var c = 0u; c < cut; c = c + 1u) {
                if (!(w[c] >= floor)) { break; }
                keep = c + 1u;
            }
            cut = max(keep, 1u);
        }
        var kept_sum = 0.0;
        for (var c = 0u; c < cut; c = c + 1u) { kept_sum = kept_sum + w[c]; }
        cut_s = cut;
        kept_s = kept_sum;
    }
    workgroupBarrier();
    let cut = cut_s;
    // The keyed argmax over the survivors of positive weight, the lowest candidate on equal scores.
    var top = ninf;
    var pick = 0xffffffffu;
    for (var c = lid; c < cut; c = c + WG) {
        let wc = w[c];
        if (!(wc > 0.0)) { continue; }
        let sc = score_of(key, wc, u32(idx[base + c]));
        if (sc > top) {
            top = sc;
            pick = c;
        }
    }
    red_v[lid] = top;
    red_i[lid] = pick;
    workgroupBarrier();
    arg_step(lid, 32u);
    arg_step(lid, 16u);
    arg_step(lid, 8u);
    arg_step(lid, 4u);
    arg_step(lid, 2u);
    arg_step(lid, 1u);
    if (lid == 0u) {
        // No survivor of positive weight: the serial walk's candidate 0.
        let p = select(red_i[0], 0u, red_i[0] == 0xffffffffu);
        out[4u * r] = idx[base + p];
        out[4u * r + 1u] = log(w[p]) - log(kept_s);
        kept[r] = f32(cut);
    }
}
