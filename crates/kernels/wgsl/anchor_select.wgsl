// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory admission: one commit step of the per-group top-`capacity` bank by (score desc, position asc)
// @how   one thread per (sequence, group, candidate), rank by counting
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// An anchor memory keeps, per KV group, the `capacity` best-scoring written
// records of all COMPLETED commit blocks (`chunk` consecutive positions
// each). This kernel advances the bank by ONE block:
//
//   bank[b + 1] = best `capacity` of (bank[b] union tokens of block b)
//
// ordered by score descending, then position ascending. The order is total
// (positions are distinct), so the result is independent of the thread
// schedule and of how a prompt is split across calls.
//
//   scores : [n_seq, t, groups] admission score of every token, fp32
//   refs   : [n_seq, n_blocks, groups, capacity] bank every block READS;
//            block b's row is both an input (b) and an output (b + 1), so the
//            blocks are dispatched in order, `block` = 0 first. A slot holds
//            `position + 1`, 0 for an empty slot, best record first. The
//            `block` = 0 call starts from an empty bank and writes row 0 (all
//            zero) itself, so the buffer needs no initialisation; a call whose
//            row b + 1 does not exist (the last block) writes nothing there.
//
// Rank by counting: each candidate counts the candidates strictly better than
// itself and writes itself to slot `rank` when `rank < capacity`; a slot no
// candidate ranks into (fewer valid candidates than slots) is zeroed by the
// thread of the same index, so every slot of block b + 1 is written exactly
// once and no atomics are needed. Scores must be finite: a NaN compares
// false against everything and would break the order.

struct Params {
    n_seq: u32,
    t: u32,          // tokens per sequence
    groups: u32,
    capacity: u32,
    chunk: u32,      // positions per commit block
    block: u32,      // the block being committed
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> refs:   array<u32>;

// The token (position + 1, 0 = none) candidate `m` stands for: the bank's own
// slots first, then the positions of the block being committed.
fn candidate(m: u32, bank_base: u32) -> u32 {
    if (m < p.capacity) {
        if (p.block == 0u) { return 0u; }
        return refs[bank_base + m];
    }
    let i = m - p.capacity;
    let pos = p.block * p.chunk + i;
    if (i < p.chunk && pos < p.t) {
        return pos + 1u;
    }
    return 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let m_total = p.capacity + p.chunk;
    let total = p.n_seq * p.groups * m_total;
    if (gidx >= total) { return; }

    let j = gidx % m_total;
    let r = gidx / m_total;
    let g = r % p.groups;
    let s = r / p.groups;

    let n_blocks = (p.t + p.chunk - 1u) / p.chunk;
    let bank_base = ((s * n_blocks + p.block) * p.groups + g) * p.capacity;
    let next_base = ((s * n_blocks + p.block + 1u) * p.groups + g) * p.capacity;
    let score_base = s * p.t * p.groups + g;

    let mine = candidate(j, bank_base);
    var my_score = 0.0;
    if (mine != 0u) { my_score = scores[score_base + (mine - 1u) * p.groups]; }

    var valid = 0u;
    var rank = 0u;
    for (var m = 0u; m < m_total; m = m + 1u) {
        let other = candidate(m, bank_base);
        if (other != 0u) {
            valid = valid + 1u;
            let other_score = scores[score_base + (other - 1u) * p.groups];
            if (mine != 0u && (other_score > my_score || (other_score == my_score && other < mine))) {
                rank = rank + 1u;
            }
        }
    }

    if (p.block == 0u && j < p.capacity) {
        refs[bank_base + j] = 0u;
    }
    if (p.block + 1u < n_blocks) {
        if (mine != 0u && rank < p.capacity) {
            refs[next_base + rank] = mine;
        }
        if (j < p.capacity && j >= valid) {
            refs[next_base + j] = 0u;
        }
    }
}
