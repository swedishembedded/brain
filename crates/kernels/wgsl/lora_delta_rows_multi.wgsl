// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row-adapter LoRA epilogue for up to four linears that read one activation: out_g[i, col0_g + j] += Σ_c t[i, g·r_max + c] · Bt_bank[slot[i], off_g + c·n_g + j], one dispatch
// @how   one thread per output element of the four outputs laid end to end, serial reduction over the rank axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// `lora_delta_rows` for a GROUP of linears that share their input: a Gated
// DeltaNet layer's `in_proj_qkv / z / b / a`, a full-attention layer's
// `q / k / v`, the MLP's `gate / up`. The group's projection is ONE
// `lora_xa_rows` over the group's A matrices stacked along the rank axis
// (`t [rows, g·r_max]`: site `g`'s `r_max` columns start at `g·r_max`, each site
// zero-padded to `r_max` so a rank below it, or a site the adapter does not
// correct, adds exactly zero). This kernel then adds every site's correction in
// one launch instead of one launch per site - what a decode step's dispatch
// count pays for.
//
//   t       : [rows, g·r_max]       from `lora_xa_rows` over the stacked A bank
//   bt_bank : [slots, r_max · Σ n]  per slot, site g's `Bt [r_max, n_g]` (alpha/r
//                                   folded in, zero past its rank) at `off_g`
//                                   = r_max · Σ_{h<g} n_h
//   slot    : [rows] u32            >= `slots` (NONE is u32::MAX): the thread returns
//                                   and no output is touched, so a base row keeps its bits
//   rank    : [slots] u32           non-zero when the slot's adapter corrects any site
//                                   of the group; 0 leaves the row alone
//   out0..3 : read_write            site g's output, `s_g` floats per row, the
//                                   correction added at `[c_g, c_g + n_g)`; a group of
//                                   fewer than four binds any buffer for the rest and
//                                   gives them `n = 0`
//   params  : rows, g, r_max, slots, row0, then per site (n, out_stride, out_col0, off)
//
// Dispatch: rows · Σ n_g threads; thread `idx` is row `idx / Σ n`, and its column
// `idx % Σ n` falls in the first site whose cumulative width exceeds it. Same
// arithmetic as `lora_delta_rows` per output element (the zero terms past a
// rank add nothing), so the two agree bit for bit on rows that are corrected.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let ntot = p.n0 + p.n1 + p.n2 + p.n3;
    if (idx >= p.rows * ntot) { return; }
    let i = idx / ntot;
    var cl = idx - i * ntot;
    let s = slot[p.row0 + i];
    if (s >= p.slots) { return; }
    if (rank[s] == 0u) { return; }

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

    let tbase = i * (p.g * p.r_max) + site * p.r_max;
    let bbase = s * (p.r_max * ntot) + off + cl;
    var acc = 0.0;
    for (var c = 0u; c < p.r_max; c = c + 1u) {
        acc = acc + t[tbase + c] * bt_bank[bbase + c * n];
    }
    let o = (p.row0 + i) * stride + col0 + cl;
    if (site == 0u) {
        out0[o] = out0[o] + acc;
    } else if (site == 1u) {
        out1[o] = out1[o] + acc;
    } else if (site == 2u) {
        out2[o] = out2[o] + acc;
    } else {
        out3[o] = out3[o] + acc;
    }
}
