// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decomposed relative-position bias backward - d(rel_h), the CONTIGUOUS segment sums of d_scores
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Decomposed relative-position bias backward - d(rel_h).
//
//   d_rel_h[h, q0+i, kh] = sum_kw d_scores[h, i, kh*kw_ext + kw]
//
// i.e. the ROW sums of each query row's [kh_ext, kw_ext] block. Unlike its
// sibling `attn_relpos_drw` (strided columns, naturally coalesced with one
// thread per output), each output here reduces a CONTIGUOUS run of `kw_ext`
// floats. One thread per output would make consecutive invocations read
// addresses `kw_ext` apart - at SAM's window extent (kw_ext = 14) that is
// roughly a 14x sector amplification on the dominant read of the whole
// backward. So the reduction is COOPERATIVE instead.
//
// `seg` invocations (a power of two, 1..64, chosen by the host as
// `min(64, kw_ext.next_power_of_two())`) share one output: lane `l` walks
// `kw = l, l+seg, l+2*seg, ...`, so a segment's invocations read consecutive
// addresses. A workgroup packs `64/seg` such segments, which for a small
// `kw_ext` keeps whole workgroups busy instead of idling 50 of 64 lanes - and
// because the segments are themselves adjacent in `d_scores`, a workgroup
// still reads one contiguous run per round.
//
// ONE top-level barrier (partials -> lane 0 folds its segment): the CPU JIT
// splits a body at exactly one and corrupts memory beyond that, so a
// per-segment tree with log(seg) barriers is not an option here. Lane 0 doing
// the `seg`-wide fold serially costs at most 64 adds and is invisible next to
// the `kw_ext` loads it follows.
//
// ASSIGNS - see `attn_relpos_drw`'s header for why no `acc` flag belongs here.
//
// d_scores layout: ((b*H + h)*qn + i)*kn + j, b == 1, j = kh*kw_ext + kw.
// d_rel_h  layout: (h*span_qn + row)*kh_ext + kh.
//
// Dispatch: ceil(heads*qn*kh_ext / (64/seg)) workgroups (thread count = *64).

struct Params {
    heads: u32,
    qn: u32,           // this chunk's query rows
    kn: u32,           // span key count == kh_ext*kw_ext
    q0: u32,           // chunk's span-local first query row
    span_qn: u32,      // d_rel_h row stride (the span's full query rows)
    kh_ext: u32,
    kw_ext: u32,
    seg: u32,          // invocations per output; a power of two in 1..=64
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_rel_h:  array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let wg = wgid.y * nwg.x + wgid.x;
    let t = lid.x;
    let seg = max(p.seg, 1u);
    let per = 64u / seg;              // outputs per workgroup
    let lane = t % seg;
    let grp = t / seg;
    let o = wg * per + grp;           // flat output index (h, i, kh)
    let total = p.heads * p.qn * p.kh_ext;

    // Indices are computed unconditionally so they survive the barrier without
    // a carried branch; every memory access below is guarded by `o < total`.
    let kh = o % p.kh_ext;
    let r1 = o / p.kh_ext;
    let i = r1 % p.qn;
    let h = r1 / p.qn;

    var acc = 0.0;
    if (o < total) {
        let base = (h * p.qn + i) * p.kn + kh * p.kw_ext;
        for (var kw: u32 = lane; kw < p.kw_ext; kw = kw + seg) {
            acc = acc + d_scores[base + kw];
        }
    }
    partial[t] = acc;
    workgroupBarrier();

    if (lane == 0u && o < total) {
        var s = 0.0;
        for (var v: u32 = 0u; v < seg; v = v + 1u) {
            s = s + partial[grp * seg + v];
        }
        d_rel_h[(h * p.span_qn + p.q0 + i) * p.kh_ext + kh] = s;
    }
}
