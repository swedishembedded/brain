// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory admission: one commit step of the per-group top-`capacity` bank by (score desc, position asc), over explicit (score, position) candidates
// @how   one thread per (sequence, group, candidate), rank by counting
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// An anchor memory keeps, per KV group, the `capacity` (K) best-scoring written
// records of all COMPLETED commit blocks. This kernel is the one place that
// order is defined. It merges a bank with up to `chunk` (C) new candidates:
//
//   out bank = best K of (in bank union new candidates), score descending,
//              then position ascending
//
// The order is total (positions are distinct), so the result is independent of
// the thread schedule and of how a prompt is split across calls. The bank is
// held as explicit candidates:
//
//   in_pos / out_pos     : [rows, groups, K]  `position + 1` of each slot, best first, 0 = empty
//   in_score / out_score : [rows, groups, K]  the matching fp32 admission scores
//   new_score            : [rows, groups]     the new candidates' scores, one row per position
//   out_src              : [rows, groups, K]  the candidate index (`m`) that landed in the slot:
//                          `m < K` is bank slot `m`, `m >= K` is new candidate `m - K`; EMPTY
//                          for an empty slot (the caller gathers keys/values by it)
//
// Per sequence (`plan`, 5 words): `in_row` (row of the in arrays; EMPTY = start from
// an empty bank), `out_row`, `new_row0` (first row of its new candidates in
// `new_score`), `new_pos0` (position of the first new candidate), `new_count`
// (0..=C live candidates). With `new_count` 0 the output is the input bank
// unchanged, so a decode step may dispatch this unconditionally and let only the
// step that completes a block carry candidates.
//
// Rank by counting: each candidate counts the candidates strictly better than
// itself and writes itself to slot `rank` when `rank < K`; a slot no candidate
// ranks into is cleared by the thread of the same index, so every output slot
// is written exactly once and no atomics are needed. `out_row` must differ from
// `in_row` (a thread reads the whole in bank). Scores must be finite.

struct Params {
    n_seq: u32,
    groups: u32,
    capacity: u32,
    chunk: u32,
};

const EMPTY: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       plan:      array<u32>;
@group(0) @binding(2) var<storage, read>       in_pos:    array<u32>;
@group(0) @binding(3) var<storage, read>       in_score:  array<f32>;
@group(0) @binding(4) var<storage, read>       new_score: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_pos:   array<u32>;
@group(0) @binding(6) var<storage, read_write> out_score: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_src:   array<u32>;

fn cand_pos(m: u32, s: u32, g: u32) -> u32 {
    let in_row = plan[s * 5u];
    if (m < p.capacity) {
        if (in_row == EMPTY) { return 0u; }
        return in_pos[(in_row * p.groups + g) * p.capacity + m];
    }
    let i = m - p.capacity;
    if (i < plan[s * 5u + 4u]) { return plan[s * 5u + 3u] + i + 1u; }
    return 0u;
}

fn cand_score(m: u32, s: u32, g: u32) -> f32 {
    if (m < p.capacity) {
        return in_score[(plan[s * 5u] * p.groups + g) * p.capacity + m];
    }
    return new_score[(plan[s * 5u + 2u] + (m - p.capacity)) * p.groups + g];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let m_total = p.capacity + p.chunk;
    if (gidx >= p.n_seq * p.groups * m_total) { return; }

    let j = gidx % m_total;
    let r = gidx / m_total;
    let g = r % p.groups;
    let s = r / p.groups;

    let mine = cand_pos(j, s, g);
    var my_score = 0.0;
    if (mine != 0u) { my_score = cand_score(j, s, g); }

    var valid = 0u;
    var rank = 0u;
    for (var m = 0u; m < m_total; m = m + 1u) {
        let other = cand_pos(m, s, g);
        if (other != 0u) {
            valid = valid + 1u;
            let other_score = cand_score(m, s, g);
            if (mine != 0u && (other_score > my_score || (other_score == my_score && other < mine))) {
                rank = rank + 1u;
            }
        }
    }

    let out_base = (plan[s * 5u + 1u] * p.groups + g) * p.capacity;
    if (mine != 0u && rank < p.capacity) {
        out_pos[out_base + rank] = mine;
        out_score[out_base + rank] = my_score;
        out_src[out_base + rank] = j;
    }
    if (j < p.capacity && j >= valid) {
        out_pos[out_base + j] = 0u;
        out_score[out_base + j] = 0.0;
        out_src[out_base + j] = EMPTY;
    }
}
