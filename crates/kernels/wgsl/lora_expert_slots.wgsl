// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-tenant serving of fine-tuned mixture-of-experts models from one resident copy of
// their quantized expert banks for its clients. If your team needs expertise in per-request adapters over routed
// experts then you can procure our services by sending an email to info@swedishembedded.com.
// @what  Per-slot adapter index of a stacked-expert LoRA bank: eslot[s] = row_slot[s / top_k] * n_experts + ids[s], and the input row s / xdiv
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The index step of a stacked-expert adapter bank overlay. A routed expert product of a decode or prefill batch has
// one SLOT per (row, chosen expert): slot `s` belongs to row `s / top_k` and runs expert `ids[s]`. An adapter bank
// for a stacked expert projection holds `bank_slots * n_experts` sub-slots, adapter slot `a`'s expert `e` at
// `a * n_experts + e`; this kernel composes each expert slot's sub-slot from its row's adapter slot and its expert,
// so the per-row-adapter kernels (`lora_xa_rows`, `lora_delta_rows`) correct the `[slots, n]` bank product unchanged.
//
//   row_slot : [rows] u32     each row's adapter slot; a value >= `bank_slots` (the NONE sentinel is u32::MAX) means
//                             no adapter
//   ids      : [slots] u32    the expert each slot runs (the bank product's own ids); >= `n_experts` is no expert
//   eslot    : [slots] u32    out: the sub-slot, or u32::MAX (no adapter) for a row without one, so that slot's
//                             output keeps the bank product's bits
//   xrow     : [slots] u32    out: `s / xdiv`, the row of the product's input the slot read (`xdiv` = top_k for a
//                             projection of the token rows, 1 for one of per-slot rows), for the row gather in front
//                             of the projection
//   params   : slots, top_k, xdiv, n_experts, bank_slots
//
// Dispatch: slots threads.

struct Params {
    slots: u32,
    top_k: u32,
    xdiv: u32,
    n_experts: u32,
    bank_slots: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       row_slot: array<u32>;
@group(0) @binding(2) var<storage, read>       ids:      array<u32>;
@group(0) @binding(3) var<storage, read_write> eslot:    array<u32>;
@group(0) @binding(4) var<storage, read_write> xrow:     array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let s = gid.y * (nwg.x * 64u) + gid.x;
    if (s >= p.slots) { return; }
    let a = row_slot[s / p.top_k];
    let e = ids[s];
    if (a >= p.bank_slots || e >= p.n_experts) {
        eslot[s] = 0xFFFFFFFFu;
    } else {
        eslot[s] = a * p.n_experts + e;
    }
    xrow[s] = s / p.xdiv;
}
