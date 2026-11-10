// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Decode-regime product of a CQ2/CQ3/CQ4 weight read in place against rotated f32 activation rows, MAXM rows per workgroup
// @how   one 64-thread workgroup per (four weight rows, block of MAXM activation rows), 16 threads per row stream its 32-lane units, codebook in shared memory, one barrier then an ordered fold
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The bandwidth-bound sibling of `matmul_cq_gemm`: the packed row (`kpad * BITS / 8` bytes) is read once, coalesced
// (16 consecutive threads take consecutive `BITS`-word units of one row), and applied to every activation row. A
// unit is 32 lanes = `BITS` words, four per 128-lane group, so a unit's norm is its group's. Sixteen threads per row
// keep a 512-wide row (16 units) on one thread each, and a workgroup serves four rows. Same operands and layout as
// `matmul_cq_gemm`; BITS and MAXM (1, 2, 4, 8) are template constants. More than MAXM rows run as several
// workgroups per row group (workgroup `i` is row group `i % ceil(n / 4)`, row block `i / ceil(n / 4)`): the packed
// rows are re-read once per block from cache, which is cheap while the weight is small against the activations'
// arithmetic.
//
//   params : m, kpad, n, 0
//   xr     : [m, kpad] f32   (rotated by cq_rotate)
//   w, s, book, y : as matmul_cq_gemm

struct Params {
    m: u32,
    kpad: u32,
    n: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xr: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<u32>;
@group(0) @binding(3) var<storage, read>       s: array<u32>;
@group(0) @binding(4) var<storage, read>       book: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

const BITS: u32 = 2u;
const MAXM: u32 = 8u;
const LANES: u32 = 16u;
const COLS: u32 = 4u;

var<workgroup> levels: array<f32, 16>;
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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = li.x;
    if (t < (1u << BITS)) { levels[t] = book[t]; }
    workgroupBarrier();
    let lin = wg.y * nwg.x + wg.x;
    let tiles = (p.n + COLS - 1u) / COLS;
    let col = (lin % tiles) * COLS + t / LANES;
    let lane = t % LANES;
    let r0 = (lin / tiles) * MAXM;
    let live = col < p.n && r0 < p.m;
    let groups = p.kpad / 128u;
    let units = p.kpad / 32u;
    let row_words = p.kpad * BITS / 32u;
    var acc: array<f32, MAXM>;
    for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = 0.0; }
    if (live) {
        for (var u = lane; u < units; u = u + LANES) {
            let flat = col * groups + u / 4u;
            let norm = f16_to_f32((s[flat >> 1u] >> ((flat & 1u) * 16u)) & 0xFFFFu);
            let uw = col * row_words + u * BITS;
            var part: array<f32, MAXM>;
            for (var r = 0u; r < MAXM; r = r + 1u) { part[r] = 0.0; }
            for (var j = 0u; j < 32u; j = j + 1u) {
                let bit = j * BITS;
                let word = bit >> 5u;
                let sh = bit & 31u;
                var v = w[uw + word] >> sh;
                if (sh + BITS > 32u) { v = v | (w[uw + word + 1u] << (32u - sh)); }
                let level = levels[v & ((1u << BITS) - 1u)];
                for (var r = 0u; r < MAXM; r = r + 1u) {
                    if (r0 + r < p.m) { part[r] = part[r] + level * xr[(r0 + r) * p.kpad + u * 32u + j]; }
                }
            }
            for (var r = 0u; r < MAXM; r = r + 1u) { acc[r] = acc[r] + norm * part[r]; }
        }
    }
    for (var r = 0u; r < MAXM; r = r + 1u) { partial[t * MAXM + r] = acc[r]; }
    workgroupBarrier();
    if (lane == 0u && live) {
        let base = (t / LANES) * LANES;
        for (var r = 0u; r < MAXM; r = r + 1u) {
            if (r0 + r < p.m) {
                var sum = 0.0;
                for (var i = 0u; i < LANES; i = i + 1u) { sum = sum + partial[(base + i) * MAXM + r]; }
                y[(r0 + r) * p.n + col] = sum;
            }
        }
    }
}
