// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned variants of one
// model from a single resident copy of its weights for its clients. If your team needs
// expertise in multi-tenant inference engines then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Per-row-adapter LoRA epilogue for one linear, four rows per thread: out[i, col0 + j] += Σ_c t[i, c] · Bt_bank[slot[i], c, j]
// @how   one thread per (4 rows, output column), the four rows' sums share each Bt load when they run under one slot, loop bounded by the slot's rank
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The row-tiled form of `lora_delta_rows.wgsl`: the SAME Params and bindings (see that
// file for the layout), one thread per (four consecutive rows, output column) instead of
// one per (row, column). When the four rows run under one slot each `Bt` value is loaded
// once and feeds four sums, a quarter of the traffic; a tile with mixed slots does each
// row's own loads. The loop runs to the slot's rank (the rank word's high 16 bits when
// set, otherwise its low bits clamped to `r_max`); the rows of `Bt` past the rank are never
// read, so the sums are `lora_delta_rows`'s bit for bit.
//
// Dispatch: ceil(rows / 4) · n threads.

struct Params {
    rows: u32,
    n: u32,
    r_max: u32,
    slots: u32,
    out_stride: u32,
    out_col0: u32,
    row0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out:     array<f32>;
@group(0) @binding(2) var<storage, read>       t:       array<f32>;
@group(0) @binding(3) var<storage, read>       bt_bank: array<f32>;
@group(0) @binding(4) var<storage, read>       slot:    array<u32>;
@group(0) @binding(5) var<storage, read>       rank:    array<u32>;

const RT: u32 = 4u;
const NONE: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let tiles = (p.rows + RT - 1u) / RT;
    if (idx >= tiles * p.n) { return; }
    let i0 = (idx / p.n) * RT;
    let j = idx - (idx / p.n) * p.n;

    // The slot and live rank of each of the four rows, NONE / 0 when it takes no correction.
    var s0 = NONE; var s1 = NONE; var s2 = NONE; var s3 = NONE;
    var k0 = 0u; var k1 = 0u; var k2 = 0u; var k3 = 0u;
    var first = NONE;
    var same = true;
    for (var r = 0u; r < RT; r = r + 1u) {
        var sl = NONE;
        var rk = 0u;
        if (i0 + r < p.rows) {
            let s = slot[p.row0 + i0 + r];
            if (s < p.slots && (rank[s] & 0xFFFFu) != 0u) {
                let hi = rank[s] >> 16u;
                sl = s;
                rk = select(min(rank[s] & 0xFFFFu, p.r_max), min(hi, p.r_max), hi != 0u);
                if (first == NONE) { first = s; }
                if (s != first) { same = false; }
            }
        }
        if (r == 0u) { s0 = sl; k0 = rk; }
        if (r == 1u) { s1 = sl; k1 = rk; }
        if (r == 2u) { s2 = sl; k2 = rk; }
        if (r == 3u) { s3 = sl; k3 = rk; }
    }
    if (first == NONE) { return; }

    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    let t0 = i0 * p.r_max;
    if (same) {
        // One slot for every row that corrects: each Bt value is loaded once.
        let bbase = first * p.r_max * p.n + j;
        let lim = max(max(k0, k1), max(k2, k3));
        let rs = p.r_max;
        var c = 0u;
        // Four ranks per step: the four Bt loads and sixteen t loads are independent, so they
        // are in flight together instead of one dependent round trip per rank.
        for (; c + 4u <= lim; c = c + 4u) {
            let b0 = bt_bank[bbase + c * p.n];
            let b1 = bt_bank[bbase + (c + 1u) * p.n];
            let b2 = bt_bank[bbase + (c + 2u) * p.n];
            let b3 = bt_bank[bbase + (c + 3u) * p.n];
            if (k0 != 0u) { a0 = a0 + t[t0 + c] * b0 + t[t0 + c + 1u] * b1 + t[t0 + c + 2u] * b2 + t[t0 + c + 3u] * b3; }
            if (k1 != 0u) { a1 = a1 + t[t0 + rs + c] * b0 + t[t0 + rs + c + 1u] * b1 + t[t0 + rs + c + 2u] * b2 + t[t0 + rs + c + 3u] * b3; }
            if (k2 != 0u) { a2 = a2 + t[t0 + 2u * rs + c] * b0 + t[t0 + 2u * rs + c + 1u] * b1 + t[t0 + 2u * rs + c + 2u] * b2 + t[t0 + 2u * rs + c + 3u] * b3; }
            if (k3 != 0u) { a3 = a3 + t[t0 + 3u * rs + c] * b0 + t[t0 + 3u * rs + c + 1u] * b1 + t[t0 + 3u * rs + c + 2u] * b2 + t[t0 + 3u * rs + c + 3u] * b3; }
        }
        for (; c < lim; c = c + 1u) {
            let b = bt_bank[bbase + c * p.n];
            if (k0 != 0u) { a0 = a0 + t[t0 + c] * b; }
            if (k1 != 0u) { a1 = a1 + t[t0 + rs + c] * b; }
            if (k2 != 0u) { a2 = a2 + t[t0 + 2u * rs + c] * b; }
            if (k3 != 0u) { a3 = a3 + t[t0 + 3u * rs + c] * b; }
        }
    } else {
        // Mixed slots: one loop, the four rows' own Bt loads in flight together.
        let r0 = s0 * p.r_max * p.n + j;
        let r1 = s1 * p.r_max * p.n + j;
        let r2 = s2 * p.r_max * p.n + j;
        let r3 = s3 * p.r_max * p.n + j;
        let lim = max(max(k0, k1), max(k2, k3));
        for (var c = 0u; c < lim; c = c + 1u) {
            var b0 = 0.0; var b1 = 0.0; var b2 = 0.0; var b3 = 0.0;
            if (c < k0) { b0 = bt_bank[r0 + c * p.n]; }
            if (c < k1) { b1 = bt_bank[r1 + c * p.n]; }
            if (c < k2) { b2 = bt_bank[r2 + c * p.n]; }
            if (c < k3) { b3 = bt_bank[r3 + c * p.n]; }
            if (c < k0) { a0 = a0 + t[t0 + c] * b0; }
            if (c < k1) { a1 = a1 + t[t0 + p.r_max + c] * b1; }
            if (c < k2) { a2 = a2 + t[t0 + 2u * p.r_max + c] * b2; }
            if (c < k3) { a3 = a3 + t[t0 + 3u * p.r_max + c] * b3; }
        }
    }
    let o0 = (p.row0 + i0) * p.out_stride + p.out_col0 + j;
    if (s0 != NONE) { out[o0] = out[o0] + a0; }
    if (s1 != NONE) { out[o0 + p.out_stride] = out[o0 + p.out_stride] + a1; }
    if (s2 != NONE) { out[o0 + 2u * p.out_stride] = out[o0 + 2u * p.out_stride] + a2; }
    if (s3 != NONE) { out[o0 + 3u * p.out_stride] = out[o0 + 3u * p.out_stride] + a3; }
}
