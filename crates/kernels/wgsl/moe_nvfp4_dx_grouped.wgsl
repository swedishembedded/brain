// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fine-tuning of quantized mixture-of-experts models on their published weights for its
// clients. If your team needs expertise in training through frozen microscaled expert banks then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  Input gradient of the NVFP4 expert-bank grouped product: dx[s, :] = W[e]^T dy[s, :] for an expert's slots taken together, each weight decoded once for up to 8 slots - the prefill-shape backward of moe_nvfp4_grouped
// @how   one thread per (8-slot tile, input column), a decoded weight applied to every slot of the tile, ascending fold over the bank rows, no barrier
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// `moe_nvfp4_dx_gather.wgsl`'s arithmetic, bit for bit per slot, over slots grouped by expert the way
// `moe_nvfp4_grouped.wgsl` reads them (`moe_route_scan.wgsl` / `moe_route_emit.wgsl` tables): a tile is up to `MR`
// slots of one expert, and every weight element decoded serves all of them.
//
//   params : u32 [tiles, k, n, ne, accumulate]   tiles bounds the table's tile count (`ne + ceil(slots / 8)`); k a multiple of 64
//   dy     : [slots, n] f32
//   tab    : [2 * (ne + 1)] u32                  slot offsets then tile offsets per expert
//   perm   : [slots] u32                         the slots in expert-major order
//   wq     : [blocks * n, k/8] u32     eight E2M1 codes per word, low nibble first (the forward's bank)
//   sc     : [blocks * n, k/64] u32    four E4M3 block scales (one per 16 inputs) per word
//   gl     : [blocks] f32              each expert's second-level scale
//   dx     : [slots, k] f32                      per slot, (+)= with accumulate
//
// Dispatch: one thread per (tile, input column) (`tiles * k`); a tile past the table's real count does nothing.

struct Params {
    tiles: u32,
    k: u32,
    n: u32,
    ne: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:   array<f32>;
@group(0) @binding(2) var<storage, read>       tab:  array<u32>;
@group(0) @binding(3) var<storage, read>       perm: array<u32>;
@group(0) @binding(4) var<storage, read>       wq:   array<u32>;
@group(0) @binding(5) var<storage, read>       sc:   array<u32>;
@group(0) @binding(6) var<storage, read>       gl:   array<f32>;
@group(0) @binding(7) var<storage, read_write> dx:   array<f32>;

const MR: u32 = 8u;

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    var v = f32(m) * (1.0 / 512.0);
    if (e != 0u) { v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    if ((b & 128u) != 0u) { v = -v; }
    return v;
}

// The value of an E2M1 code: sign, 2-bit exponent (bias 1), 1-bit mantissa.
fn fp4_value(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    var v = f32(m) * 0.5;
    if (e != 0u) { v = bitcast<f32>(((126u + e) << 23u) | (m << 22u)); }
    if ((c & 8u) != 0u) { v = -v; }
    return v;
}

// Element `i` of bank row `row_w`, without the bank's per-expert scale: code (low nibble first) times its E4M3 scale.
fn weight(row_w: u32, i: u32) -> f32 {
    let code = (wq[row_w * (p.k / 8u) + i / 8u] >> ((i % 8u) * 4u)) & 15u;
    let g = i / 16u;
    let s = (sc[row_w * (p.k / 64u) + g / 4u] >> ((g % 4u) * 8u)) & 255u;
    return fp4_value(code) * e4m3(s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.tiles * p.k) { return; }
    let tile = idx / p.k;
    let i = idx % p.k;
    if (tile >= tab[2u * p.ne + 1u]) { return; }
    // The expert whose tile range holds `tile`: the last e with tile_start[e] <= tile.
    var lo = 0u;
    var hi = p.ne;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (tab[p.ne + 1u + mid] <= tile) { lo = mid; } else { hi = mid; }
    }
    let e = lo;
    let first = tab[e] + (tile - tab[p.ne + 1u + e]) * MR;
    let cnt = min(MR, tab[e + 1u] - first);
    var acc: array<f32, 8>;
    var slot: array<u32, 8>;
    for (var r = 0u; r < MR; r = r + 1u) {
        acc[r] = 0.0;
        slot[r] = 0u;
        if (r < cnt) { slot[r] = perm[first + r]; }
    }
    for (var col = 0u; col < p.n; col = col + 1u) {
        let w = weight(e * p.n + col, i);
        for (var r = 0u; r < MR; r = r + 1u) {
            if (r < cnt) { acc[r] = acc[r] + w * dy[slot[r] * p.n + col]; }
        }
    }
    for (var r = 0u; r < cnt; r = r + 1u) {
        let o = slot[r] * p.k + i;
        var v = acc[r] * gl[e];
        if (p.accumulate != 0u) { v = v + dx[o]; }
        dx[o] = v;
    }
}
