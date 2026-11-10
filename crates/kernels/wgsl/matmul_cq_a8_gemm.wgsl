// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats and int8 activation paths then you can procure
// our services by sending an email to info@swedishembedded.com.

// @what  Batched product of int8 rotated activations with a CQ2/CQ3/CQ4 weight read in place, group dot products accumulated in integers: y[m, n] = sum_g (table scale * norm * act scale) * (int table[idx] . q) - the prefill, encoder and batched-decode regime of the A8 precision mode
// @how   one thread per (weight row, block of MAXM activation rows): each code is looked up in an int8 table once and applied to all MAXM rows in i32, one float multiply chain per group, no barriers
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The W2A8 product the vendor engine runs. The codebook is reduced to an int8 table (scale = max|level| / 127,
// table = round-half-away(level / scale)), the activation groups were quantised by `cq_rotate_a8`, so a group's dot
// product is an exact `i32` (at most 128 * 127 * 127). A group contributes `((table_scale * norm) * f32(dot)) *
// act_scale`; the groups are summed in order. Opt-in precision mode only; `matmul_cq_gemm` is the f32 path.
// BITS (2, 3 or 4) and MAXM are template constants. No barriers, so the CPU JIT runs it.
//
//   params : m, kpad, n, 0
//   q      : [m, kpad / 4] u32      int8 lanes, four to a word
//   a      : [m, kpad / 128] f32    activation group scales
//   w, s, book, y : as matmul_cq_gemm

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let blocks = (p.m + MAXM - 1u) / MAXM;
    if (idx >= p.n * blocks) { return; }
    let col = idx % p.n;
    let r0 = (idx / p.n) * MAXM;
    let groups = p.kpad / GROUP;
    let row_words = p.kpad * BITS / 32u;
    var maxabs = 0.0;
    for (var i = 0u; i < LEVELS; i = i + 1u) { maxabs = max(maxabs, abs(book[i])); }
    let table_scale = maxabs / 127.0;
    var table: array<i32, 16>;
    for (var i = 0u; i < LEVELS; i = i + 1u) { table[i] = i32(clamp(round_away(book[i] / table_scale), -128.0, 127.0)); }
    var y_acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { y_acc[r] = 0.0; }
    for (var g = 0u; g < groups; g = g + 1u) {
        let flat = col * groups + g;
        let unit = table_scale * f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
        var dot: array<i32, MAXM>;
        for (var r = 0u; r < MAXM; r = r + 1u) { dot[r] = 0; }
        let gw = col * row_words + g * 4u * BITS;
        for (var j = 0u; j < GROUP; j = j + 1u) {
            let bit = j * BITS;
            let word = bit >> 5u;
            let sh = bit & 31u;
            var v = w[gw + word] >> sh;
            if (sh + BITS > 32u) { v = v | (w[gw + word + 1u] << (32u - sh)); }
            let level = table[v & ((1u << BITS) - 1u)];
            let lane = g * GROUP + j;
            for (var r = 0u; r < MAXM; r = r + 1u) {
                if (r0 + r < p.m) { dot[r] = dot[r] + level * sbyte(q[((r0 + r) * p.kpad + lane) / 4u], lane % 4u); }
            }
        }
        for (var r = 0u; r < MAXM; r = r + 1u) {
            if (r0 + r < p.m) { y_acc[r] = y_acc[r] + unit * f32(dot[r]) * a[(r0 + r) * groups + g]; }
        }
    }
    for (var r = 0u; r < MAXM; r = r + 1u) {
        if (r0 + r < p.m) { y[(r0 + r) * p.n + col] = y_acc[r]; }
    }
}
