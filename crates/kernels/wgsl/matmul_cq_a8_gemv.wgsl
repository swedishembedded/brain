// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats and int8 activation paths then you can procure
// our services by sending an email to info@swedishembedded.com.

// @what  Decode-regime product of a CQ2/CQ3/CQ4 weight read in place against int8 rotated activation rows with integer group dot products, MAXM rows per workgroup - the cooperative form of matmul_cq_a8_gemm
// @how   one 64-thread workgroup per (four weight rows, block of MAXM activation rows), sixteen threads per row each take a 32-lane quarter of four groups at a time, the int8 table built once per workgroup, the four quarter dot products of a group added as integers in workgroup memory behind barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// A group's integer dot product must be complete before its float scale is applied once, so the four 32-lane quarters
// of a group (`BITS` words each, read coalesced by consecutive threads) are summed as exact integers through workgroup
// memory, and the thread owning the group's first quarter adds the group's term: `((table_scale * norm) * f32(dot)) *
// act_scale`, bit-identical to `matmul_cq_a8_gemm`'s. Four groups are in flight per row; their terms and the four
// slots' partial sums are added in slot order, so a row of at most four groups sums in the host's order and a longer one
// differs from it by f32 reassociation only. More than MAXM rows run as several workgroups per row group, as in
// `matmul_cq_gemv`. BITS and MAXM are template constants.
//
//   params : m, kpad, n, 0
//   q, a, w, s, book, y : as matmul_cq_a8_gemm

struct Params {
    m: u32,
    kpad: u32,
    n: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<u32>;
@group(0) @binding(2) var<storage, read>       a: array<f32>;
@group(0) @binding(3) var<storage, read>       w: array<u32>;
@group(0) @binding(4) var<storage, read>       s: array<u32>;
@group(0) @binding(5) var<storage, read>       book: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;

const BITS: u32 = 2u;
const MAXM: u32 = 8u;
const GROUP: u32 = 128u;
const LEVELS: u32 = 1u << BITS;
const LANES: u32 = 16u;
const COLS: u32 = 4u;
const SLOTS: u32 = 4u;

var<workgroup> table: array<i32, 16>;
var<workgroup> table_scale_shared: array<f32, 1>;
var<workgroup> quarter_dots: array<i32, 64u * MAXM>;
var<workgroup> partial: array<f32, 64u * MAXM>;

fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

// Round half away from zero, exact for every f32.
fn round_away(v: f32) -> f32 {
    let t = trunc(abs(v));
    let r = t + select(0.0, 1.0, abs(v) - t >= 0.5);
    return select(r, -r, v < 0.0);
}

// Signed byte `lane` (0..3) of a packed word.
fn sbyte(word: u32, lane: u32) -> i32 {
    return (i32((word >> (lane * 8u)) & 0xFFu) ^ 128) - 128;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = li.x;
    let lin = wg.y * nwg.x + wg.x;
    let tiles = (p.n + COLS - 1u) / COLS;
    let col = (lin % tiles) * COLS + t / LANES;
    let lane = t % LANES;
    let r0 = (lin / tiles) * MAXM;
    let live = col < p.n && r0 < p.m;
    let groups = p.kpad / GROUP;
    let row_words = p.kpad * BITS / 32u;
    // The int8 table and its scale, built once per workgroup: thread `i < LEVELS` rounds level `i`.
    if (t < LEVELS) {
        var maxabs = 0.0;
        for (var i = 0u; i < LEVELS; i = i + 1u) { maxabs = max(maxabs, abs(book[i])); }
        let scale = maxabs / 127.0;
        table[t] = i32(clamp(round_away(book[t] / scale), -128.0, 127.0));
        if (t == 0u) { table_scale_shared[0] = scale; }
    }
    workgroupBarrier();
    let table_scale = table_scale_shared[0];
    let quarter = lane % 4u;
    let slot = lane / 4u;
    var acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
    // Every thread runs every round (the barriers are uniform); a thread past the last group adds zeros.
    for (var round = 0u; round < (groups + SLOTS - 1u) / SLOTS; round = round + 1u) {
        let g = slot + SLOTS * round;
        let owner = live && g < groups;
        var dot: array<i32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { dot[r] = 0; }
        if (owner) {
            let uw = col * row_words + g * 4u * BITS + quarter * BITS;
            for (var j = 0u; j < 32u; j = j + 1u) {
                let bit = j * BITS;
                let word = bit >> 5u;
                let sh = bit & 31u;
                var v = w[uw + word] >> sh;
                if (sh + BITS > 32u) { v = v | (w[uw + word + 1u] << (32u - sh)); }
                let level = table[v & ((1u << BITS) - 1u)];
                let at = g * GROUP + quarter * 32u + j;
                for (var r = 0u; r < MAXM; r = r + 1u) {
                    if (r0 + r < p.m) { dot[r] = dot[r] + level * sbyte(q[((r0 + r) * p.kpad + at) / 4u], at % 4u); }
                }
            }
        }
        for (var r = 0u; r < MAXM; r = r + 1u) { quarter_dots[t * MAXM + r] = dot[r]; }
        workgroupBarrier();
        if (owner && quarter == 0u) {
            let flat = col * groups + g;
            let unit = table_scale * f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
            for (var r = 0u; r < MAXM; r = r + 1u) {
                if (r0 + r < p.m) {
                    let total = quarter_dots[t * MAXM + r] + quarter_dots[(t + 1u) * MAXM + r] + quarter_dots[(t + 2u) * MAXM + r] + quarter_dots[(t + 3u) * MAXM + r];
                    acc[r] = acc[r] + unit * f32(total) * a[(r0 + r) * groups + g];
                }
            }
        }
        workgroupBarrier();
    }
    for (var r = 0u; r < MAXM; r = r + 1u) { partial[t * MAXM + r] = acc[r]; }
    workgroupBarrier();
    if (lane == 0u && live) {
        for (var r = 0u; r < MAXM; r = r + 1u) {
            if (r0 + r < p.m) {
                var sum = 0.0;
                for (var i = 0u; i < SLOTS; i = i + 1u) { sum = sum + partial[(t + 4u * i) * MAXM + r]; }
                y[(r0 + r) * p.n + col] = sum;
            }
        }
    }
}
