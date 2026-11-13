// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned variants of one
// model from a single resident copy of its weights for its clients. If your team needs
// expertise in multi-tenant inference engines then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Per-row-adapter LoRA epilogue for up to four linears that read one activation, four rows per thread: out_g[i, col0_g + j] += Σ_c t[i, g·r_max + c] · Bt_bank[slot[i], off_g + c·n_g + j]
// @how   one thread per (4 rows, output column), the four rows' sums share each Bt load when they run under one slot, loop bounded by the slot's rank
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The row-tiled form of `lora_delta_rows_multi.wgsl`: the SAME Params and bindings (see
// that file for the layout; `lora_delta_rows_tiled.wgsl` is the one-site form). The row-wise kernel gives every output element to its own
// thread, so each thread re-reads the slot's `Bt` column that its neighbours' rows read: a
// launch of 32 rows over a 17,408-wide gate+up pair moves `rows · r · Σ n` floats of `Bt`
// through the cache. A thread here owns the same column for 4 consecutive rows; when the
// four run under one slot (a rollout group, most of a decode batch) each `Bt` value is
// loaded once and feeds four sums, a quarter of the traffic. A tile with mixed slots does
// each row's own loads.
//
// The loop runs to the slot's RANK, read from the rank word's high 16 bits when set
// (otherwise the low bits clamped to `r_max`, the form a single-site bank stores): the
// rows of `Bt` past it are zero padding, so the sums are the row-wise kernels' bit for bit
// on corrected rows while an adapter of rank 16 in a bank of rank cap 32 reads half as much.
//
// Dispatch: ceil(rows / 4) · Σ n_g threads.

struct Params {
    rows: u32,
    g: u32,
    r_max: u32,
    slots: u32,
    row0: u32,
    n0: u32, s0: u32, c0: u32, o0: u32,
    n1: u32, s1: u32, c1: u32, o1: u32,
    n2: u32, s2: u32, c2: u32, o2: u32,
    n3: u32, s3: u32, c3: u32, o3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       t:       array<f32>;
@group(0) @binding(2) var<storage, read>       bt_bank: array<f32>;
@group(0) @binding(3) var<storage, read>       slot:    array<u32>;
@group(0) @binding(4) var<storage, read>       rank:    array<u32>;
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;
@group(0) @binding(8) var<storage, read_write> out3: array<f32>;

const RT: u32 = 4u;
const NONE: u32 = 0xFFFFFFFFu;

fn add(site: u32, o: u32, v: f32) {
    if (site == 0u) {
        out0[o] = out0[o] + v;
    } else if (site == 1u) {
        out1[o] = out1[o] + v;
    } else if (site == 2u) {
        out2[o] = out2[o] + v;
    } else {
        out3[o] = out3[o] + v;
    }
}

// The slot of row `i` when it takes a correction (a real slot with a live width), else NONE.
fn slot_of(i: u32) -> u32 {
    if (i < p.rows) {
        let s = slot[p.row0 + i];
        if (s < p.slots && (rank[s] & 0xFFFFu) != 0u) { return s; }
    }
    return NONE;
}

// The rank bounding the loop of slot `s`: the rank word's high half when set, else its low half clamped.
fn rank_of(s: u32) -> u32 {
    if (s == NONE) { return 0u; }
    let hi = rank[s] >> 16u;
    return select(min(rank[s] & 0xFFFFu, p.r_max), min(hi, p.r_max), hi != 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let ntot = p.n0 + p.n1 + p.n2 + p.n3;
    let tiles = (p.rows + RT - 1u) / RT;
    if (idx >= tiles * ntot) { return; }
    let i0 = (idx / ntot) * RT;
    var cl = idx - (idx / ntot) * ntot;

    var site = 0u;
    var n = p.n0;
    var stride = p.s0;
    var col0 = p.c0;
    var off = p.o0;
    if (cl >= p.n0) {
        cl = cl - p.n0;
        site = 1u; n = p.n1; stride = p.s1; col0 = p.c1; off = p.o1;
        if (cl >= p.n1) {
            cl = cl - p.n1;
            site = 2u; n = p.n2; stride = p.s2; col0 = p.c2; off = p.o2;
            if (cl >= p.n2) {
                cl = cl - p.n2;
                site = 3u; n = p.n3; stride = p.s3; col0 = p.c3; off = p.o3;
            }
        }
    }

    // The slot and live rank of each of the four rows, NONE / 0 when it takes no correction.
    let s0 = slot_of(i0);
    let s1 = slot_of(i0 + 1u);
    let s2 = slot_of(i0 + 2u);
    let s3 = slot_of(i0 + 3u);
    let k0 = rank_of(s0);
    let k1 = rank_of(s1);
    let k2 = rank_of(s2);
    let k3 = rank_of(s3);
    var first = NONE;
    if (first == NONE) { first = s0; }
    if (first == NONE) { first = s1; }
    if (first == NONE) { first = s2; }
    if (first == NONE) { first = s3; }
    if (first == NONE) { return; }
    let same = (s0 == NONE || s0 == first) && (s1 == NONE || s1 == first) && (s2 == NONE || s2 == first) && (s3 == NONE || s3 == first);

    let width = p.g * p.r_max;
    let rs = p.r_max;
    let t0 = i0 * width + site * rs;
    let tw = width;
    let row_stride = p.r_max * (p.n0 + p.n1 + p.n2 + p.n3);
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    if (same) {
        // One slot for every row that corrects: each Bt value is loaded once.
        let bbase = first * row_stride + off + cl;
        let lim = max(max(k0, k1), max(k2, k3));
        var c = 0u;
        // Four ranks per step: the four Bt loads and sixteen t loads are independent, so they
        // are in flight together instead of one dependent round trip per rank.
        for (; c + 4u <= lim; c = c + 4u) {
            let b0 = bt_bank[bbase + c * n];
            let b1 = bt_bank[bbase + (c + 1u) * n];
            let b2 = bt_bank[bbase + (c + 2u) * n];
            let b3 = bt_bank[bbase + (c + 3u) * n];
            if (k0 != 0u) { a0 = a0 + t[t0 + c] * b0 + t[t0 + c + 1u] * b1 + t[t0 + c + 2u] * b2 + t[t0 + c + 3u] * b3; }
            if (k1 != 0u) { a1 = a1 + t[t0 + tw + c] * b0 + t[t0 + tw + c + 1u] * b1 + t[t0 + tw + c + 2u] * b2 + t[t0 + tw + c + 3u] * b3; }
            if (k2 != 0u) { a2 = a2 + t[t0 + 2u * tw + c] * b0 + t[t0 + 2u * tw + c + 1u] * b1 + t[t0 + 2u * tw + c + 2u] * b2 + t[t0 + 2u * tw + c + 3u] * b3; }
            if (k3 != 0u) { a3 = a3 + t[t0 + 3u * tw + c] * b0 + t[t0 + 3u * tw + c + 1u] * b1 + t[t0 + 3u * tw + c + 2u] * b2 + t[t0 + 3u * tw + c + 3u] * b3; }
        }
        for (; c < lim; c = c + 1u) {
            let b = bt_bank[bbase + c * n];
            if (k0 != 0u) { a0 = a0 + t[t0 + c] * b; }
            if (k1 != 0u) { a1 = a1 + t[t0 + tw + c] * b; }
            if (k2 != 0u) { a2 = a2 + t[t0 + 2u * tw + c] * b; }
            if (k3 != 0u) { a3 = a3 + t[t0 + 3u * tw + c] * b; }
        }
    } else {
        // Mixed slots: one loop, the four rows' own Bt loads in flight together.
        let bo = off + cl;
        let lim = max(max(k0, k1), max(k2, k3));
        let r0 = s0 * row_stride + bo;
        let r1 = s1 * row_stride + bo;
        let r2 = s2 * row_stride + bo;
        let r3 = s3 * row_stride + bo;
        for (var c = 0u; c < lim; c = c + 1u) {
            var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
            if (c < k0) { b0 = bt_bank[r0 + c * n]; }
            if (c < k1) { b1 = bt_bank[r1 + c * n]; }
            if (c < k2) { b2 = bt_bank[r2 + c * n]; }
            if (c < k3) { b3 = bt_bank[r3 + c * n]; }
            if (c < k0) { a0 = a0 + t[t0 + c] * b0; }
            if (c < k1) { a1 = a1 + t[t0 + tw + c] * b1; }
            if (c < k2) { a2 = a2 + t[t0 + 2u * tw + c] * b2; }
            if (c < k3) { a3 = a3 + t[t0 + 3u * tw + c] * b3; }
        }
    }
    let o0 = (p.row0 + i0) * stride + col0 + cl;
    if (s0 != NONE) { add(site, o0, a0); }
    if (s1 != NONE) { add(site, o0 + stride, a1); }
    if (s2 != NONE) { add(site, o0 + 2u * stride, a2); }
    if (s3 != NONE) { add(site, o0 + 3u * stride, a3); }
}
