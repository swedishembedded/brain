// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned variants of one
// model from a single resident copy of its weights for its clients. If your team needs
// expertise in multi-tenant inference engines then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Per-row-adapter LoRA projection t = A_bank[slot[row]] · x[row], one workgroup per 4 rows x 4 rank indices - the multi-adapter first half for a batch
// @how   256-thread workgroup, 16 register accumulators per thread, each loaded x and A value feeds four products, 3 barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The row-tiled form of `lora_xa_rows.wgsl`: the SAME Params, bindings and contract (see
// that file for the layout), different work split. `lora_xa_rows` runs one workgroup per
// (row, rank index): each re-reads its row's activation (`k` floats) and one row of `A`
// (`k` floats) for ONE dot product, so a launch moves `2 · rows · r_max · k` floats
// through the cache hierarchy whatever the rows share. Measured on the 27B (32 rows on
// one adapter, 128 stacked rank rows) that is 164 MB in 33 us, the L2's bandwidth.
//
// Here a workgroup owns a TILE of RT = 4 rows by JT = 4 rank indices. When the four rows run
// under the same slot (every row of a rollout group, every row of a one-adapter server) each
// of the 256 threads strides `k`, loads 4 activations and 4 `A` values per step and feeds 16
// products: a quarter of `lora_xa_rows`'s traffic. A tile whose rows run under different
// slots loads each row's own `A` values and is correct but slower than the row-wise kernel
// (a sixteenth of the workgroups, so too few warps to hide the loads), which is why
// `lora_bank_rows` offers this kernel only to a bank with one slot. The 16 accumulators of
// every thread are folded across the workgroup in two stages (a thread sums 16 partials,
// then 16 threads sum 16) behind three barriers, in a 16 KB workgroup table.
//
// `t[i, j]` is written exactly when `lora_xa_rows` writes it: the row is in range, its
// slot is a real one, and `j` is below the slot's live width (the rank word's low 16
// bits clamped to `r_max`); every other `t` entry is left as it was. Different summation
// order from the row-wise kernel (256 partials folded as 16 x 16), so the pair agrees to
// fp32 accumulation error, not bit for bit.
//
// It needs several barriers, which the CPU JIT mis-executes (`@cpu no`):
// `backend_api::select`'s `Op::LoraBankProject` offers it only where
// `caps.workgroup_reductions` holds, and from the measured row count at which it wins.
//
// Dispatch: ceil(rows / 4) * ceil(r_max / 4) workgroups.

struct Params {
    rows: u32,
    k: u32,
    r_max: u32,
    slots: u32,
    row0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       a_bank: array<f32>;
@group(0) @binding(3) var<storage, read>       slot:   array<u32>;
@group(0) @binding(4) var<storage, read>       rank:   array<u32>;
@group(0) @binding(5) var<storage, read_write> t:      array<f32>;

const RT: u32 = 4u;
const JT: u32 = 4u;
const WG: u32 = 256u;
const LANE: u32 = 64u;
const UNROLL: u32 = 2u;
const NONE: u32 = 0xFFFFFFFFu;

// 16 outputs x 256 threads: the partial sums, then the stage-1 sums in their place.
var<workgroup> partial: array<f32, 4096>;

// The live width of a slot: `j < width` is the rows of `A` that exist.
fn width_of(s: u32) -> u32 {
    return min(rank[s] & 0xFFFFu, p.r_max);
}

// The slot of row `i` when it takes a correction, else NONE.
fn slot_of(i: u32) -> u32 {
    if (i < p.rows) {
        let s = slot[p.row0 + i];
        if (s < p.slots) { return s; }
    }
    return NONE;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = li.x;
    let tile = wg.y * nwg.x + wg.x;
    let jtiles = (p.r_max + JT - 1u) / JT;
    let i0 = (tile / jtiles) * RT;
    let j0 = (tile % jtiles) * JT;

    let s0 = slot_of(i0 + 0u);
    let xb0 = (p.row0 + i0 + 0u) * p.k;
    let s1 = slot_of(i0 + 1u);
    let xb1 = (p.row0 + i0 + 1u) * p.k;
    let s2 = slot_of(i0 + 2u);
    let xb2 = (p.row0 + i0 + 2u) * p.k;
    let s3 = slot_of(i0 + 3u);
    let xb3 = (p.row0 + i0 + 3u) * p.k;
    var first = NONE;
    if (first == NONE) { first = s0; }
    if (first == NONE) { first = s1; }
    if (first == NONE) { first = s2; }
    if (first == NONE) { first = s3; }
    let same = (s0 == NONE || s0 == first) && (s1 == NONE || s1 == first) && (s2 == NONE || s2 == first) && (s3 == NONE || s3 == first);
    // Per (row, rank index): the start of the A row, or NONE when `t[i, j]` is not produced.
    var ab00 = NONE;
    if (s0 != NONE && j0 + 0u < width_of(s0)) { ab00 = (s0 * p.r_max + j0 + 0u) * p.k; }
    var ab01 = NONE;
    if (s0 != NONE && j0 + 1u < width_of(s0)) { ab01 = (s0 * p.r_max + j0 + 1u) * p.k; }
    var ab02 = NONE;
    if (s0 != NONE && j0 + 2u < width_of(s0)) { ab02 = (s0 * p.r_max + j0 + 2u) * p.k; }
    var ab03 = NONE;
    if (s0 != NONE && j0 + 3u < width_of(s0)) { ab03 = (s0 * p.r_max + j0 + 3u) * p.k; }
    var ab10 = NONE;
    if (s1 != NONE && j0 + 0u < width_of(s1)) { ab10 = (s1 * p.r_max + j0 + 0u) * p.k; }
    var ab11 = NONE;
    if (s1 != NONE && j0 + 1u < width_of(s1)) { ab11 = (s1 * p.r_max + j0 + 1u) * p.k; }
    var ab12 = NONE;
    if (s1 != NONE && j0 + 2u < width_of(s1)) { ab12 = (s1 * p.r_max + j0 + 2u) * p.k; }
    var ab13 = NONE;
    if (s1 != NONE && j0 + 3u < width_of(s1)) { ab13 = (s1 * p.r_max + j0 + 3u) * p.k; }
    var ab20 = NONE;
    if (s2 != NONE && j0 + 0u < width_of(s2)) { ab20 = (s2 * p.r_max + j0 + 0u) * p.k; }
    var ab21 = NONE;
    if (s2 != NONE && j0 + 1u < width_of(s2)) { ab21 = (s2 * p.r_max + j0 + 1u) * p.k; }
    var ab22 = NONE;
    if (s2 != NONE && j0 + 2u < width_of(s2)) { ab22 = (s2 * p.r_max + j0 + 2u) * p.k; }
    var ab23 = NONE;
    if (s2 != NONE && j0 + 3u < width_of(s2)) { ab23 = (s2 * p.r_max + j0 + 3u) * p.k; }
    var ab30 = NONE;
    if (s3 != NONE && j0 + 0u < width_of(s3)) { ab30 = (s3 * p.r_max + j0 + 0u) * p.k; }
    var ab31 = NONE;
    if (s3 != NONE && j0 + 1u < width_of(s3)) { ab31 = (s3 * p.r_max + j0 + 1u) * p.k; }
    var ab32 = NONE;
    if (s3 != NONE && j0 + 2u < width_of(s3)) { ab32 = (s3 * p.r_max + j0 + 2u) * p.k; }
    var ab33 = NONE;
    if (s3 != NONE && j0 + 3u < width_of(s3)) { ab33 = (s3 * p.r_max + j0 + 3u) * p.k; }
    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc02 = 0.0;
    var acc03 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc20 = 0.0;
    var acc21 = 0.0;
    var acc22 = 0.0;
    var acc23 = 0.0;
    var acc30 = 0.0;
    var acc31 = 0.0;
    var acc32 = 0.0;
    var acc33 = 0.0;

    if (same) {
        // One slot: the A rows are the first row's and every row reads them.
        var sa0 = NONE;
        if (first != NONE && j0 + 0u < width_of(first)) { sa0 = (first * p.r_max + j0 + 0u) * p.k; }
        var sa1 = NONE;
        if (first != NONE && j0 + 1u < width_of(first)) { sa1 = (first * p.r_max + j0 + 1u) * p.k; }
        var sa2 = NONE;
        if (first != NONE && j0 + 2u < width_of(first)) { sa2 = (first * p.r_max + j0 + 2u) * p.k; }
        var sa3 = NONE;
        if (first != NONE && j0 + 3u < width_of(first)) { sa3 = (first * p.r_max + j0 + 3u) * p.k; }
        // UNROLL strides of the 256 threads per step: every activation and A value of the step is
        // loaded before any product, so the loads are in flight together.
        for (var c0 = tid; c0 < p.k; c0 = c0 + UNROLL * WG) {
            let c0 = c0 + 0u * WG;
            let in0 = c0 < p.k;
            var x00 = 0.0;
            if (in0 && s0 != NONE) { x00 = x[xb0 + c0]; }
            var x01 = 0.0;
            if (in0 && s1 != NONE) { x01 = x[xb1 + c0]; }
            var x02 = 0.0;
            if (in0 && s2 != NONE) { x02 = x[xb2 + c0]; }
            var x03 = 0.0;
            if (in0 && s3 != NONE) { x03 = x[xb3 + c0]; }
            var a00 = 0.0;
            if (in0 && sa0 != NONE) { a00 = a_bank[sa0 + c0]; }
            var a01 = 0.0;
            if (in0 && sa1 != NONE) { a01 = a_bank[sa1 + c0]; }
            var a02 = 0.0;
            if (in0 && sa2 != NONE) { a02 = a_bank[sa2 + c0]; }
            var a03 = 0.0;
            if (in0 && sa3 != NONE) { a03 = a_bank[sa3 + c0]; }
            let c1 = c0 + 1u * WG;
            let in1 = c1 < p.k;
            var x10 = 0.0;
            if (in1 && s0 != NONE) { x10 = x[xb0 + c1]; }
            var x11 = 0.0;
            if (in1 && s1 != NONE) { x11 = x[xb1 + c1]; }
            var x12 = 0.0;
            if (in1 && s2 != NONE) { x12 = x[xb2 + c1]; }
            var x13 = 0.0;
            if (in1 && s3 != NONE) { x13 = x[xb3 + c1]; }
            var a10 = 0.0;
            if (in1 && sa0 != NONE) { a10 = a_bank[sa0 + c1]; }
            var a11 = 0.0;
            if (in1 && sa1 != NONE) { a11 = a_bank[sa1 + c1]; }
            var a12 = 0.0;
            if (in1 && sa2 != NONE) { a12 = a_bank[sa2 + c1]; }
            var a13 = 0.0;
            if (in1 && sa3 != NONE) { a13 = a_bank[sa3 + c1]; }
            acc00 = acc00 + x00 * a00;
            acc01 = acc01 + x00 * a01;
            acc02 = acc02 + x00 * a02;
            acc03 = acc03 + x00 * a03;
            acc10 = acc10 + x01 * a00;
            acc11 = acc11 + x01 * a01;
            acc12 = acc12 + x01 * a02;
            acc13 = acc13 + x01 * a03;
            acc20 = acc20 + x02 * a00;
            acc21 = acc21 + x02 * a01;
            acc22 = acc22 + x02 * a02;
            acc23 = acc23 + x02 * a03;
            acc30 = acc30 + x03 * a00;
            acc31 = acc31 + x03 * a01;
            acc32 = acc32 + x03 * a02;
            acc33 = acc33 + x03 * a03;
            acc00 = acc00 + x10 * a10;
            acc01 = acc01 + x10 * a11;
            acc02 = acc02 + x10 * a12;
            acc03 = acc03 + x10 * a13;
            acc10 = acc10 + x11 * a10;
            acc11 = acc11 + x11 * a11;
            acc12 = acc12 + x11 * a12;
            acc13 = acc13 + x11 * a13;
            acc20 = acc20 + x12 * a10;
            acc21 = acc21 + x12 * a11;
            acc22 = acc22 + x12 * a12;
            acc23 = acc23 + x12 * a13;
            acc30 = acc30 + x13 * a10;
            acc31 = acc31 + x13 * a11;
            acc32 = acc32 + x13 * a12;
            acc33 = acc33 + x13 * a13;
        }
    } else {
        // Mixed slots: each row's own A rows.
        for (var c0 = tid; c0 < p.k; c0 = c0 + UNROLL * WG) {
            let c0 = c0 + 0u * WG;
            let in0 = c0 < p.k;
            var x00 = 0.0;
            if (in0 && s0 != NONE) { x00 = x[xb0 + c0]; }
            var x01 = 0.0;
            if (in0 && s1 != NONE) { x01 = x[xb1 + c0]; }
            var x02 = 0.0;
            if (in0 && s2 != NONE) { x02 = x[xb2 + c0]; }
            var x03 = 0.0;
            if (in0 && s3 != NONE) { x03 = x[xb3 + c0]; }
            let c1 = c0 + 1u * WG;
            let in1 = c1 < p.k;
            var x10 = 0.0;
            if (in1 && s0 != NONE) { x10 = x[xb0 + c1]; }
            var x11 = 0.0;
            if (in1 && s1 != NONE) { x11 = x[xb1 + c1]; }
            var x12 = 0.0;
            if (in1 && s2 != NONE) { x12 = x[xb2 + c1]; }
            var x13 = 0.0;
            if (in1 && s3 != NONE) { x13 = x[xb3 + c1]; }
            if (in0 && ab00 != NONE) { acc00 = acc00 + x00 * a_bank[ab00 + c0]; }
            if (in0 && ab01 != NONE) { acc01 = acc01 + x00 * a_bank[ab01 + c0]; }
            if (in0 && ab02 != NONE) { acc02 = acc02 + x00 * a_bank[ab02 + c0]; }
            if (in0 && ab03 != NONE) { acc03 = acc03 + x00 * a_bank[ab03 + c0]; }
            if (in0 && ab10 != NONE) { acc10 = acc10 + x01 * a_bank[ab10 + c0]; }
            if (in0 && ab11 != NONE) { acc11 = acc11 + x01 * a_bank[ab11 + c0]; }
            if (in0 && ab12 != NONE) { acc12 = acc12 + x01 * a_bank[ab12 + c0]; }
            if (in0 && ab13 != NONE) { acc13 = acc13 + x01 * a_bank[ab13 + c0]; }
            if (in0 && ab20 != NONE) { acc20 = acc20 + x02 * a_bank[ab20 + c0]; }
            if (in0 && ab21 != NONE) { acc21 = acc21 + x02 * a_bank[ab21 + c0]; }
            if (in0 && ab22 != NONE) { acc22 = acc22 + x02 * a_bank[ab22 + c0]; }
            if (in0 && ab23 != NONE) { acc23 = acc23 + x02 * a_bank[ab23 + c0]; }
            if (in0 && ab30 != NONE) { acc30 = acc30 + x03 * a_bank[ab30 + c0]; }
            if (in0 && ab31 != NONE) { acc31 = acc31 + x03 * a_bank[ab31 + c0]; }
            if (in0 && ab32 != NONE) { acc32 = acc32 + x03 * a_bank[ab32 + c0]; }
            if (in0 && ab33 != NONE) { acc33 = acc33 + x03 * a_bank[ab33 + c0]; }
            if (in1 && ab00 != NONE) { acc00 = acc00 + x10 * a_bank[ab00 + c1]; }
            if (in1 && ab01 != NONE) { acc01 = acc01 + x10 * a_bank[ab01 + c1]; }
            if (in1 && ab02 != NONE) { acc02 = acc02 + x10 * a_bank[ab02 + c1]; }
            if (in1 && ab03 != NONE) { acc03 = acc03 + x10 * a_bank[ab03 + c1]; }
            if (in1 && ab10 != NONE) { acc10 = acc10 + x11 * a_bank[ab10 + c1]; }
            if (in1 && ab11 != NONE) { acc11 = acc11 + x11 * a_bank[ab11 + c1]; }
            if (in1 && ab12 != NONE) { acc12 = acc12 + x11 * a_bank[ab12 + c1]; }
            if (in1 && ab13 != NONE) { acc13 = acc13 + x11 * a_bank[ab13 + c1]; }
            if (in1 && ab20 != NONE) { acc20 = acc20 + x12 * a_bank[ab20 + c1]; }
            if (in1 && ab21 != NONE) { acc21 = acc21 + x12 * a_bank[ab21 + c1]; }
            if (in1 && ab22 != NONE) { acc22 = acc22 + x12 * a_bank[ab22 + c1]; }
            if (in1 && ab23 != NONE) { acc23 = acc23 + x12 * a_bank[ab23 + c1]; }
            if (in1 && ab30 != NONE) { acc30 = acc30 + x13 * a_bank[ab30 + c1]; }
            if (in1 && ab31 != NONE) { acc31 = acc31 + x13 * a_bank[ab31 + c1]; }
            if (in1 && ab32 != NONE) { acc32 = acc32 + x13 * a_bank[ab32 + c1]; }
            if (in1 && ab33 != NONE) { acc33 = acc33 + x13 * a_bank[ab33 + c1]; }
        }
    }
    partial[0u * WG + tid] = acc00;
    partial[1u * WG + tid] = acc01;
    partial[2u * WG + tid] = acc02;
    partial[3u * WG + tid] = acc03;
    partial[4u * WG + tid] = acc10;
    partial[5u * WG + tid] = acc11;
    partial[6u * WG + tid] = acc12;
    partial[7u * WG + tid] = acc13;
    partial[8u * WG + tid] = acc20;
    partial[9u * WG + tid] = acc21;
    partial[10u * WG + tid] = acc22;
    partial[11u * WG + tid] = acc23;
    partial[12u * WG + tid] = acc30;
    partial[13u * WG + tid] = acc31;
    partial[14u * WG + tid] = acc32;
    partial[15u * WG + tid] = acc33;
    workgroupBarrier();
    // Stage 1: thread (o, m) sums 16 of output o's 256 partials.
    let o1 = tid >> 4u;
    let m1 = tid & 15u;
    var stage1 = 0.0;
    for (var q = 0u; q < 16u; q = q + 1u) {
        stage1 = stage1 + partial[o1 * WG + q * 16u + m1];
    }
    workgroupBarrier();
    partial[tid] = stage1;
    workgroupBarrier();
    // Stage 2: thread o sums output o's 16 stage-1 sums and writes it.
    if (tid < 16u) {
        let r = tid / JT;
        let q = tid % JT;
        var live = false;
        if (r == 0u && q == 0u && ab00 != NONE) { live = true; }
        if (r == 0u && q == 1u && ab01 != NONE) { live = true; }
        if (r == 0u && q == 2u && ab02 != NONE) { live = true; }
        if (r == 0u && q == 3u && ab03 != NONE) { live = true; }
        if (r == 1u && q == 0u && ab10 != NONE) { live = true; }
        if (r == 1u && q == 1u && ab11 != NONE) { live = true; }
        if (r == 1u && q == 2u && ab12 != NONE) { live = true; }
        if (r == 1u && q == 3u && ab13 != NONE) { live = true; }
        if (r == 2u && q == 0u && ab20 != NONE) { live = true; }
        if (r == 2u && q == 1u && ab21 != NONE) { live = true; }
        if (r == 2u && q == 2u && ab22 != NONE) { live = true; }
        if (r == 2u && q == 3u && ab23 != NONE) { live = true; }
        if (r == 3u && q == 0u && ab30 != NONE) { live = true; }
        if (r == 3u && q == 1u && ab31 != NONE) { live = true; }
        if (r == 3u && q == 2u && ab32 != NONE) { live = true; }
        if (r == 3u && q == 3u && ab33 != NONE) { live = true; }
        if (live) {
            var sum = 0.0;
            for (var m = 0u; m < 16u; m = m + 1u) {
                sum = sum + partial[tid * 16u + m];
            }
            t[(i0 + r) * p.r_max + j0 + q] = sum;
        }
    }
}
