// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row-adapter LoRA projection t = A_bank[slot[row]] · x[row], one WORKGROUP per (row, rank index) - the multi-adapter first half
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Per-row-adapter LoRA projection: `t[i, j] = Σ_c x[i, c] · A_bank[slot[i], j, c]`
// for `j < rank[slot[i]]`.
//
//   x      : [rows, k]
//   a_bank : [slots, r_max, k]   each slot's `A [r, k]` (the GEMM-weight layout),
//                                rows `rank..r_max` of a slot are never read
//   slot   : [rows] u32          the adapter of each row; any value >= `slots`
//                                (the NONE sentinel is u32::MAX) means "no adapter"
//   rank   : [slots] u32         the live rank of each slot, clamped to `r_max`
//   t      : [rows, r_max]       columns `rank..r_max` and every NONE row are
//                                left as they were - `lora_delta_rows` reads the
//                                same slot/rank and never looks at them
//   params : rows, k, r_max, slots, row0
//
// The window is rows `row0..row0 + rows` of `x` and `slot`; `t` row `i` belongs
// to window row `i`. The offset is a Param rather than a bound sub-range
// because a bind offset must be a multiple of the device's storage-buffer
// offset alignment (256 bytes), which an arbitrary row start is not.
//
// This is the first half of `y_i += B_slot(i) · (A_slot(i) · x_i)`, the runtime
// LoRA of ONE batch whose rows use DIFFERENT adapters (or none). The slot vector
// is the gather: rows need not be contiguous or sorted by adapter, and no
// per-adapter dispatch exists, so the dispatch count is independent of how many
// distinct adapters the batch holds. Row `i` is a pure function of `(x_i, slot_i)`.
//
// Why a workgroup per (row, j) and not a thread per output: `t` is only `r_max`
// wide (8-32) while `k` is the model width (1k-12k), so the work is a long
// reduction behind a short output. One thread per `t[i, j]` would walk `k`
// serially (checklist C2: uncoalesced, a serial chain of `k` dependent loads).
// Here the 64 threads stride `k`, so both `x[i, ..]` and `a_bank[slot, j, ..]`
// are read contiguously; ONE barrier, then thread 0 folds the 64 partials (64
// adds, cheaper than a second barrier; the CPU JIT supports exactly one).
// `x[i, ..]` is re-read by the `r_max` workgroups of its row, which is L2
// traffic, not DRAM: `rows·k` floats is the whole activation.
//
// A workgroup whose row is NONE (or whose `j` is past the slot's rank) does NOT
// return before the barrier: the guard comes from a storage load, which WGSL's
// uniformity analysis treats as non-uniform, so an early return ahead of
// `workgroupBarrier()` would be rejected. It skips its loop, contributes zeros
// and writes nothing.
//
// The reduction folds `k` in a different order than the GEMM `lora_rows_off`
// uses, so the two agree to fp32 accumulation error, not bit for bit. The CPU
// JIT's split-at-barrier execution mis-executes cooperative reductions
// (`caps.workgroup_reductions` is false there), so `lora_xa_rows_serial` is the
// same contract for those devices - selected by `backend_api::select`'s
// `Op::LoraBankProject`.
//
// Dispatch: rows * r_max workgroups.

struct Params {
    rows: u32,
    k: u32,
    r_max: u32,
    slots: u32,
    row0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       a_bank: array<f32>;
@group(0) @binding(3) var<storage, read>       slot:   array<u32>;
@group(0) @binding(4) var<storage, read>       rank:   array<u32>;
@group(0) @binding(5) var<storage, read_write> t:      array<f32>;

var<workgroup> partial: array<f32, 64>;

// Whether `t[i, j]` is produced at all: the row is in range, its slot is a real
// one, and `j` is below that slot's rank.
fn is_live(i: u32, j: u32) -> bool {
    if (i >= p.rows) { return false; }
    let s = slot[p.row0 + i];
    return s < p.slots && j < min(rank[s], p.r_max);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = li.x;
    var acc = 0.0;
    {
        let item = wg.y * nwg.x + wg.x;
        let i = item / p.r_max;
        let j = item - i * p.r_max;
        if (is_live(i, j)) {
            let xbase = (p.row0 + i) * p.k;
            let abase = (slot[p.row0 + i] * p.r_max + j) * p.k;
            for (var c = tid; c < p.k; c = c + 64u) {
                acc = acc + x[xbase + c] * a_bank[abase + c];
            }
        }
    }
    partial[tid] = acc;
    workgroupBarrier();
    // The CPU JIT re-initialises per-invocation locals on each side of the
    // barrier, so nothing computed above is read below: the item is re-derived.
    if (tid == 0u) {
        let item = wg.y * nwg.x + wg.x;
        let i = item / p.r_max;
        let j = item - i * p.r_max;
        if (is_live(i, j)) {
            var sum = 0.0;
            for (var q = 0u; q < 64u; q = q + 1u) {
                sum = sum + partial[q];
            }
            t[i * p.r_max + j] = sum;
        }
    }
}
