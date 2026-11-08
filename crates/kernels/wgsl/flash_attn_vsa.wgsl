// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: block-sparse flash attention, each query tile over the key tiles its mask keeps
// @how   `flash_attn_bidir_reg`'s 256-thread lane-split workgroup on one 64-row query tile and head, rows gathered through the tile slot map, key tiles skipped by mask, 6 barriers (3 per key tile, 3 per staged chunk)
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// `flash_attn_bidir_reg`'s arithmetic over a TILED sequence
// (`minimaxh3::vsa::TileGeometry`): a workgroup owns query tile `qt` of head
// `h` - reg's 64-row tile is exactly one VSA tile - and walks every key tile,
// skipping each one `mask[h, qt, kt]` drops. The test is the same for the
// whole workgroup, so the skip never diverges. Rows are read through
// `slots`, the packed row in each of a tile's 64 slots, so neither q/k/v nor
// the output is ever permuted into tile order.
//
// A kept tile is staged in chunks of BC rows up to its live count: every
// chunk holds at least one live key, which keeps the online softmax's
// running max finite from the first chunk on. Keys past the live count
// score -inf; query rows past it are computed and never written.
//
// Shared use, lane ownership, bank argument and barrier order are reg's. Two
// storage buffers more than reg (slots, live, mask).

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const V4: u32 = 8u;      // vec4 slots per lane (V4*4*LANES == HD)
const HDV: u32 = 32u;    // vec4 slots per key row (HD/4)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

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

var<workgroup> ksh:  array<vec4<f32>, 256>;  // BC*HDV -> 4 KiB
var<workgroup> vsh:  array<vec4<f32>, 256>;  // BC*HDV -> 4 KiB
var<workgroup> part: array<f32, 2048>;       // BC*BR*LANES -> 8 KiB
var<workgroup> kkept: u32;                   // the key tile's mask bit
var<workgroup> klive_sh: u32;                // the key tile's live rows

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (head, query tile).
    let wg = wgid.y * nwg.x + wgid.x;
    let qt = wg % p.tiles;
    let h = wg / p.tiles;
    if (h >= p.n_heads) { return; }

    let lt = lid.x;                 // 0..255
    let row = lt / LANES;           // 0..63  -> query slot within the tile
    let lane = lt % LANES;          // 0..3   -> vec4-slot phase
    let live_q = row < live[qt];
    let i = select(0u, slots[qt * BR + row], live_q);

    // This thread's slice of q, and its slice of the output accumulator.
    // Slot c holds channels 4*(c*LANES + lane) .. +3.
    var q4: array<vec4<f32>, 8>;
    var o4: array<vec4<f32>, 8>;
    let q_base = i * p.qkv_stride + p.q_off + h * hd;
    for (var c = 0u; c < V4; c = c + 1u) {
        let d0 = (c * LANES + lane) * 4u;
        var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (live_q) {
            if (d0 + 0u < hd) { v.x = qkv[q_base + d0 + 0u]; }
            if (d0 + 1u < hd) { v.y = qkv[q_base + d0 + 1u]; }
            if (d0 + 2u < hd) { v.z = qkv[q_base + d0 + 2u]; }
            if (d0 + 3u < hd) { v.w = qkv[q_base + d0 + 3u]; }
        }
        q4[c] = v;
        o4[c] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    let mrow = (h * p.tiles + qt) * p.tiles;
    for (var kt = 0u; kt < p.tiles; kt = kt + 1u) {
      // The key tile's mask bit and live count, read once and made uniform:
      // every barrier below sits in control flow that depends on them.
      if (lt == 0u) {
          kkept = mask[mrow + kt];
          klive_sh = live[kt];
      }
      let kept = workgroupUniformLoad(&kkept);
      let klive = workgroupUniformLoad(&klive_sh);
      workgroupBarrier(); // everyone has read both before thread 0 rewrites them
      if (kept == 0u) { continue; }
      for (var kc = 0u; kc < klive; kc = kc + BC) {
        // Stage BC of the key tile's rows [BC x HDV vec4] into shared, read
        // through the slot map. Channels past head_dim and slots past the
        // live count are zeroed, which is what makes the fixed loops correct.
        for (var e = lt; e < BC * HDV; e = e + 256u) {
            let jr = e / HDV;
            let d0 = (e % HDV) * 4u;
            let js = kc + jr;
            var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var vv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (js < klive) {
                let base = slots[kt * BR + js] * p.qkv_stride + h * hd + d0;
                let bk = base + p.k_off;
                let bv = base + p.v_off;
                if (d0 + 3u < hd) {
                    kv = vec4<f32>(qkv[bk], qkv[bk + 1u], qkv[bk + 2u], qkv[bk + 3u]);
                    vv = vec4<f32>(qkv[bv], qkv[bv + 1u], qkv[bv + 2u], qkv[bv + 3u]);
                } else {
                    if (d0 + 0u < hd) { kv.x = qkv[bk + 0u]; vv.x = qkv[bv + 0u]; }
                    if (d0 + 1u < hd) { kv.y = qkv[bk + 1u]; vv.y = qkv[bv + 1u]; }
                    if (d0 + 2u < hd) { kv.z = qkv[bk + 2u]; vv.z = qkv[bv + 2u]; }
                }
            }
            ksh[e] = kv;
            vsh[e] = vv;
        }
        workgroupBarrier();

        // Partial dot products: this lane's 32 channels of every key in the
        // tile, four channels per shared load.
        for (var j = 0u; j < BC; j = j + 1u) {
            var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            let ko = j * HDV + lane;
            for (var c = 0u; c < V4; c = c + 1u) {
                acc = acc + q4[c] * ksh[ko + c * LANES];
            }
            part[j * (BR * LANES) + row * LANES + lane] = acc.x + acc.y + acc.z + acc.w;
        }
        workgroupBarrier();

        // Every lane re-sums the LANES partials (redundant but register-cheap,
        // and it leaves each lane with the p_j it needs for its own channels).
        var krows = BC;
        let rem = klive - kc;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows) {
                let po = j * (BR * LANES) + row * LANES;
                s = (part[po] + part[po + 1u] + part[po + 2u] + part[po + 3u]) * scale;
            }
            pj[j] = s;
            tmax = max(tmax, s);
        }
        let m_new = max(m, tmax);
        let corr = exp(m - m_new);
        var lsum = 0.0;
        for (var j = 0u; j < BC; j = j + 1u) {
            let e = exp(pj[j] - m_new);
            pj[j] = e;
            lsum = lsum + e;
        }
        l = l * corr + lsum;
        m = m_new;

        // One rescale of o per TILE, not per key; four channels per load.
        for (var c = 0u; c < V4; c = c + 1u) {
            let vo = c * LANES + lane;
            var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            for (var j = 0u; j < BC; j = j + 1u) {
                acc = acc + pj[j] * vsh[j * HDV + vo];
            }
            o4[c] = o4[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the chunk before it is overwritten
      }
    }

    if (live_q) {
        let inv = 1.0 / l;
        let o_base = i * p.d_model + h * hd;
        for (var c = 0u; c < V4; c = c + 1u) {
            let d0 = (c * LANES + lane) * 4u;
            let v = o4[c] * inv;
            if (d0 + 0u < hd) { out[o_base + d0 + 0u] = v.x; }
            if (d0 + 1u < hd) { out[o_base + d0 + 1u] = v.y; }
            if (d0 + 2u < hd) { out[o_base + d0 + 2u] = v.z; }
            if (d0 + 3u < hd) { out[o_base + d0 + 3u] = v.w; }
        }
    }
}
