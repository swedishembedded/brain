// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA flash attention backward, dK and dV: one workgroup per key tile, summing every query head of its group
// @how   256-thread workgroup tile, 3 barriers, causal early-start over Q tiles
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The key-side half of the backward of `flash_attn_causal_gqa.wgsl` (the
// query side is `flash_attn_causal_gqa_bwd_dq.wgsl`, which must run first: it
// writes `dsum`). With `P_ij = exp(scale·q_i·k_j - lse_i)`, `dP_ij = dO_i·v_j`
// and `D_i = dsum_i`:
//
//   dV_j = Σ_h Σ_{i>=j} P_ij·dO_i        dK_j = scale · Σ_h Σ_{i>=j} P_ij·(dP_ij - D_i)·q_i
//
// where `h` runs over the query heads sharing kv head `hkv`. Owning a key tile
// per workgroup and walking the whole group inside it is what makes GQA's
// many-to-one head map an ordinary sum: no atomics, no second reduction pass,
// and every `dK`/`dV` element is written exactly once.
//
// Work split mirrors the forward with the roles of queries and keys swapped:
// BR = 64 KEY rows per workgroup, LANES = 4 threads per row, CH = 32 channels
// per lane, so a thread's k, v, dK and dV slices live in registers. Query and
// dO tiles of BC = 8 rows (plus their lse and D) are staged in shared memory;
// the causal mask makes every query tile that ends before this key tile starts
// all-zero, so the walk starts at the first tile that can see it. Shared use
// is 24 KiB; callers gate on the device's workgroup memory.

const BR: u32 = 64u;     // key rows per workgroup
const BC: u32 = 8u;      // query rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one key row
const CH: u32 = 32u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:    array<f32>;
@group(0) @binding(2) var<storage, read>       k:    array<f32>;
@group(0) @binding(3) var<storage, read>       v:    array<f32>;
@group(0) @binding(4) var<storage, read>       d_o:  array<f32>;  // grad wrt ctx
@group(0) @binding(5) var<storage, read>       lse:  array<f32>;  // [B, n_heads, T]
@group(0) @binding(6) var<storage, read>       dsum: array<f32>;  // D, [B, n_heads, T]
@group(0) @binding(7) var<storage, read_write> dk:   array<f32>;
@group(0) @binding(8) var<storage, read_write> dv:   array<f32>;

var<workgroup> qsh:    array<f32, 1024>;  // BC*HD
var<workgroup> dosh:   array<f32, 1024>;  // BC*HD
var<workgroup> part_s: array<f32, 2048>;  // BC*BR*LANES
var<workgroup> part_p: array<f32, 2048>;  // BC*BR*LANES
var<workgroup> lse_sh: array<f32, 8>;     // BC
var<workgroup> d_sh:   array<f32, 8>;     // BC

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let T = p.tcols;
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (b, kv head, key-tile); uniform per workgroup.
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_k = (T + BR - 1u) / BR;
    let kt = wg % ntiles_k;
    let r = wg / ntiles_k;
    let hkv = r % p.n_kv_heads;
    let b = r / p.n_kv_heads;
    if (b >= p.bsz) { return; }

    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;
    let row = lt / LANES;
    let lane = lt % LANES;
    let j = kt * BR + row;           // this thread's key row
    let live = j < T;

    var kv: array<f32, 32>;
    var vv: array<f32, 32>;
    var dk_acc: array<f32, 32>;
    var dv_acc: array<f32, 32>;
    let kv_base = (b * T + j) * kv_row + hkv * hd;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * LANES + lane;
        if (live && d < hd) {
            kv[c] = k[kv_base + d];
            vv[c] = v[kv_base + d];
        } else {
            kv[c] = 0.0;
            vv[c] = 0.0;
        }
        dk_acc[c] = 0.0;
        dv_acc[c] = 0.0;
    }

    // Causal: query i sees key j only when i >= j, so no query tile ending
    // before this workgroup's first key contributes.
    let first_qt = (kt * BR) / BC;
    let ntiles_q = (T + BC - 1u) / BC;

    for (var g = 0u; g < p.group; g = g + 1u) {
        let h = hkv * p.group + g;
        for (var qt = first_qt; qt < ntiles_q; qt = qt + 1u) {
            for (var e = lt; e < BC * HD; e = e + 256u) {
                let ir = e / HD;
                let d = e % HD;
                let i = qt * BC + ir;
                if (i < T && d < hd) {
                    let qo = (b * T + i) * q_row + h * hd + d;
                    qsh[e] = q[qo];
                    dosh[e] = d_o[qo];
                } else {
                    qsh[e] = 0.0;
                    dosh[e] = 0.0;
                }
            }
            if (lt < BC) {
                let i = qt * BC + lt;
                if (i < T) {
                    let stat = (b * p.n_heads + h) * T + i;
                    lse_sh[lt] = lse[stat];
                    d_sh[lt] = dsum[stat];
                } else {
                    lse_sh[lt] = 0.0;
                    d_sh[lt] = 0.0;
                }
            }
            workgroupBarrier();

            for (var ir = 0u; ir < BC; ir = ir + 1u) {
                var s = 0.0;
                var dp = 0.0;
                let qo = ir * HD + lane;
                for (var c = 0u; c < CH; c = c + 1u) {
                    s = s + kv[c] * qsh[qo + c * LANES];
                    dp = dp + vv[c] * dosh[qo + c * LANES];
                }
                let slot = ir * (BR * LANES) + row * LANES + lane;
                part_s[slot] = s;
                part_p[slot] = dp;
            }
            workgroupBarrier();

            for (var ir = 0u; ir < BC; ir = ir + 1u) {
                let i = qt * BC + ir;
                if (live && i < T && i >= j) {
                    let so = ir * (BR * LANES) + row * LANES;
                    let s = (part_s[so] + part_s[so + 1u] + part_s[so + 2u] + part_s[so + 3u]) * scale;
                    let dp = part_p[so] + part_p[so + 1u] + part_p[so + 2u] + part_p[so + 3u];
                    let pw = exp(s - lse_sh[ir]);
                    let ds = pw * (dp - d_sh[ir]);
                    let qo = ir * HD + lane;
                    for (var c = 0u; c < CH; c = c + 1u) {
                        dv_acc[c] = dv_acc[c] + pw * dosh[qo + c * LANES];
                        dk_acc[c] = dk_acc[c] + ds * qsh[qo + c * LANES];
                    }
                }
            }
            workgroupBarrier(); // done reading the tile before it is overwritten
        }
    }

    if (live) {
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) {
                dk[kv_base + d] = dk_acc[c] * scale;
                dv[kv_base + d] = dv_acc[c];
            }
        }
    }
}
