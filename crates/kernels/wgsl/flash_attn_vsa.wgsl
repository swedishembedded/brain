// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: block-sparse flash attention, two 64-row query tiles per workgroup over the key tiles either keeps
// @how   `flash_attn_bidir_reg2`'s two-query-row register block on a pair of query tiles of one head, rows gathered through the tile slot map, a key tile staged when either query tile keeps it and masked per row when its own does not, 3 barriers per staged chunk
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The attention of `minimaxh3::vsa`: a query tile of 64 rows attends the key
// tiles its mask row keeps. `flash_attn_bidir_reg2`'s 128-row register block
// is exactly two tiles: thread (rg, lane) owns slot rg of query tile 2w and
// slot rg of query tile 2w+1, so every staged key and value row is read from
// shared memory once for two queries.
//
// A key tile is staged when EITHER of the two tiles keeps it (the decision
// is the same for the whole workgroup, so the skip never diverges); a row
// whose own tile drops it gives each of its keys a weight of exactly zero -
// not exp(-inf - max), which would be 1 for a row that has seen nothing yet -
// so each query attends its own selection and nothing else. Rows are read
// through `slots`, the packed row in each of a tile's 64 slots; neither q/k/v
// nor the output is ever permuted.
//
// A kept tile is staged in chunks of BC rows up to its live count; keys past
// the live count score -inf, query slots past it are computed and never
// written. Shared use, lane ownership and the bank argument are reg2's:
// 48 KiB, which Pascal's compute limit holds exactly. The keep bits and live
// counts are read-only storage loads at workgroup-uniform addresses, so the
// barriers that depend on them need no shared copy.

const BR: u32 = 64u;     // query rows per tile (two tiles per workgroup)
const RG: u32 = 64u;     // row groups: one row of each tile per thread
const BC: u32 = 16u;     // key/value rows per staged chunk
const LANES: u32 = 4u;   // threads cooperating on one query row
const V4: u32 = 8u;      // vec4 slots per lane (V4*4*LANES == HD)
const HDV: u32 = 32u;    // vec4 slots per key row (HD/4)
const NEG: f32 = -3.4e38;

struct Params {
    tiles: u32,
    n_heads: u32,
    head_dim: u32,     // <= 128
    qkv_stride: u32,   // 3 * d_model
    q_off: u32,        // 0
    k_off: u32,        // d_model
    v_off: u32,        // 2 * d_model
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;
@group(0) @binding(3) var<storage, read>       slots: array<u32>;
@group(0) @binding(4) var<storage, read>       live:  array<u32>;
@group(0) @binding(5) var<storage, read>       mask:  array<u32>;

var<workgroup> ksh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> vsh:   array<vec4<f32>, 512>;  // BC*HDV  ->  8 KiB
var<workgroup> part0: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB
var<workgroup> part1: array<vec4<f32>, 1024>; // BC*RG   -> 16 KiB

fn load4(base: u32, hd: u32, d0: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (d0 + 0u < hd) { v.x = qkv[base + d0 + 0u]; }
    if (d0 + 1u < hd) { v.y = qkv[base + d0 + 1u]; }
    if (d0 + 2u < hd) { v.z = qkv[base + d0 + 2u]; }
    if (d0 + 3u < hd) { v.w = qkv[base + d0 + 3u]; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (head, pair of query tiles).
    let pairs = (p.tiles + 1u) / 2u;
    let wg = wgid.y * nwg.x + wgid.x;
    let pr = wg % pairs;
    let h = wg / pairs;
    if (h >= p.n_heads) { return; }
    let qa = 2u * pr;
    let qb = qa + 1u;            // may be past the last tile
    let has_b = qb < p.tiles;

    let lt = lid.x;              // 0..255
    let rg = lt / LANES;         // 0..63 -> slot in each tile
    let lane = lt % LANES;       // 0..3  -> vec4-slot phase
    let live0 = rg < live[qa];
    var live1 = false;
    if (has_b) { live1 = rg < live[qb]; }
    let i0 = select(0u, slots[qa * BR + rg], live0);
    let i1 = select(0u, slots[min(qb, p.tiles - 1u) * BR + rg], live1);

    var q0: array<vec4<f32>, 8>;
    var q1: array<vec4<f32>, 8>;
    var o0: array<vec4<f32>, 8>;
    var o1: array<vec4<f32>, 8>;
    let qb0 = i0 * p.qkv_stride + p.q_off + h * hd;
    let qb1 = i1 * p.qkv_stride + p.q_off + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        q0[c] = select(vec4<f32>(0.0), load4(qb0, hd, d0), live0);
        q1[c] = select(vec4<f32>(0.0), load4(qb1, hd, d0), live1);
        o0[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        o1[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var m0 = NEG;
    var l0 = 0.0;
    var m1 = NEG;
    var l1 = 0.0;
    var pj0: array<f32, 16>;
    var pj1: array<f32, 16>;

    let row_a = (h * p.tiles + qa) * p.tiles;
    let row_b = (h * p.tiles + min(qb, p.tiles - 1u)) * p.tiles;
    for (var kt = 0u; kt < p.tiles; kt = kt + 1u) {
        // Read-only loads at an address the whole workgroup shares, so uniform:
        // every barrier below may depend on them.
        let ka = mask[row_a + kt];
        let kb = select(0u, mask[row_b + kt], has_b);
        let klive = live[kt];
        if (ka == 0u && kb == 0u) { continue; }

        for (var kc = 0u; kc < klive; kc = kc + BC) {
            // Stage BC of the key tile's rows through the slot map; channels
            // past head_dim and slots past the live count are zeroed.
            for (var e = lt; e < BC * HDV; e = e + 256u) {
                let js = kc + e / HDV;
                let d0 = (e % HDV) * 4u;
                var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                if (js < klive) {
                    let base = slots[kt * BR + js] * p.qkv_stride + h * hd;
                    kv = load4(base + p.k_off, hd, d0);
                    vv = load4(base + p.v_off, hd, d0);
                }
                ksh[e] = kv;
                vsh[e] = vv;
            }
            workgroupBarrier();

            // Partial dot products for both query rows off one tile read.
            for (var j = 0u; j < BC; j = j + 1u) {
                var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                let ko = j * HDV + lane;
                for (var c = 0u; c < V4; c = c + 1u) {
                    let kv = ksh[ko + c * LANES];
                    a0 = a0 + q0[c] * kv;
                    a1 = a1 + q1[c] * kv;
                }
                let po = j * RG + rg;
                part0[po][lane] = a0.x + a0.y + a0.z + a0.w;
                part1[po][lane] = a1.x + a1.y + a1.z + a1.w;
            }
            workgroupBarrier();

            let krows = min(BC, klive - kc);
            var tmax0 = NEG;
            var tmax1 = NEG;
            for (var j = 0u; j < BC; j = j + 1u) {
                var s0 = NEG;
                var s1 = NEG;
                if (j < krows) {
                    let po = j * RG + rg;
                    let v0 = part0[po];
                    let v1 = part1[po];
                    if (ka != 0u) { s0 = (v0.x + v0.y + v0.z + v0.w) * scale; }
                    if (kb != 0u) { s1 = (v1.x + v1.y + v1.z + v1.w) * scale; }
                }
                pj0[j] = s0;
                pj1[j] = s1;
                tmax0 = max(tmax0, s0);
                tmax1 = max(tmax1, s1);
            }
            let mn0 = max(m0, tmax0);
            let mn1 = max(m1, tmax1);
            let corr0 = exp(m0 - mn0);
            let corr1 = exp(m1 - mn1);
            var ls0 = 0.0;
            var ls1 = 0.0;
            for (var j = 0u; j < BC; j = j + 1u) {
                // A dropped or past-the-end key weighs exactly nothing.
                let e0 = select(0.0, exp(pj0[j] - mn0), pj0[j] > NEG);
                let e1 = select(0.0, exp(pj1[j] - mn1), pj1[j] > NEG);
                pj0[j] = e0;
                pj1[j] = e1;
                ls0 = ls0 + e0;
                ls1 = ls1 + e1;
            }
            l0 = l0 * corr0 + ls0;
            l1 = l1 * corr1 + ls1;
            m0 = mn0;
            m1 = mn1;

            for (var c = 0u; c < V4; c = c + 1u) {
                let vo = c * LANES + lane;
                var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                for (var j = 0u; j < BC; j = j + 1u) {
                    let vv = vsh[j * HDV + vo];
                    a0 = a0 + pj0[j] * vv;
                    a1 = a1 + pj1[j] * vv;
                }
                o0[c] = o0[c] * corr0 + a0;
                o1[c] = o1[c] * corr1 + a1;
            }
            workgroupBarrier(); // done reading the chunk before it is overwritten
        }
    }

    let ob0 = i0 * p.d_model + h * hd;
    let ob1 = i1 * p.d_model + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        if (live0) {
            let v = o0[c] / l0;
            if (d0 + 0u < hd) { out[ob0 + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[ob0 + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[ob0 + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[ob0 + d0 + 3u] = v.w; }
        }
        if (live1) {
            let v = o1[c] / l1;
            if (d0 + 0u < hd) { out[ob1 + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[ob1 + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[ob1 + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[ob1 + d0 + 3u] = v.w; }
        }
    }
}
