// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row-adapter LoRA epilogue: out[i, col0 + j] += Σ_c t[i, c] · Bt_bank[slot[i], c, j], skipping rows with no adapter
// @how   one thread per output element, serial reduction over the rank axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Per-row-adapter LoRA epilogue, the second half of
// `y_i += B_slot(i) · (A_slot(i) · x_i)`:
//
//   out[i * out_stride + out_col0 + j] += Σ_{c < rank[slot[i]]} t[i, c] · bt_bank[slot[i], c, j]
//
//   t       : [rows, r_max]        from `lora_xa_rows`
//   bt_bank : [slots, r_max, n]    each slot's `B [n, r]` stored TRANSPOSED as
//                                  `[r, n]` with alpha/r (and any strength) folded
//                                  in, exactly `lora_delta`'s `bt`; rows
//                                  `rank..r_max` are never read
//   slot    : [rows] u32           >= `slots` (NONE is u32::MAX) means no adapter:
//                                  the thread returns and `out` is not touched
//                                  (a rank-0 slot likewise, so such rows keep their bits)
//   rank    : [slots] u32          live rank per slot, clamped to `r_max`
//   out     : read_write           `out_stride` floats per row; the correction is
//                                  added to the window `[out_col0, out_col0 + n)`
//   params  : rows, n, r_max, slots, out_stride, out_col0, row0
//
// The window is rows `row0..row0 + rows` of `slot` and `out`; `t` row `i` belongs
// to window row `i` (see `lora_xa_rows` for why the offset is a Param).
//
// `out_stride`/`out_col0` are what let the correction land straight in a fused
// projection - the q, k and v columns of one `[rows, hq + 2·hkv]` row, or the
// gate and up halves of `[rows, 2·ff]` - without a gather or a split first.
//
// Same arithmetic and the same coalescing argument as `lora_delta.wgsl` (the
// rank is 8-32 against widths of thousands, so the kernel is bound by the
// read-modify-write of `out`; consecutive threads take consecutive `j`, so both
// `out` and `bt_bank` are contiguous), with one addition: the weight is
// gathered per row through the slot vector, so a batch of N distinct adapters is
// still one dispatch. It stays a sibling of `lora_delta` rather than a change to
// it - that kernel's contract (one `bt`, contiguous `[m, n]` output) has other
// models behind it.
//
// Dispatch: rows * n threads.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let i = idx / p.n;
    let j = idx - i * p.n;
    let s = slot[p.row0 + i];
    if (s >= p.slots) { return; }
    let r = min(rank[s], p.r_max);
    if (r == 0u) { return; }
    let tbase = i * p.r_max;
    let bbase = s * p.r_max * p.n + j;
    var acc = 0.0;
    for (var c = 0u; c < r; c = c + 1u) {
        acc = acc + t[tbase + c] * bt_bank[bbase + c * p.n];
    }
    let o = (p.row0 + i) * p.out_stride + p.out_col0 + j;
    out[o] = out[o] + acc;
}
