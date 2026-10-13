// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row-adapter LoRA projection t = A_bank[slot[row]] · x[row], one thread per (row, rank index) - barrier-free twin of lora_xa_rows
// @how   one thread per output element, serial reduction over k
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The barrier-free twin of `lora_xa_rows.wgsl`: the SAME Params, bindings and
// contract (see that file for the layout), one invocation per `t[i, j]`
// walking `k` serially.
//
// It exists because the CPU JIT executes workgroups as split-at-barrier loops
// and mis-executes cooperative reductions (`caps.workgroup_reductions` is false
// there), so a device without that capability needs a kernel with no barrier.
// `backend_api::select`'s `Op::LoraBankProject` hands those devices this one;
// on a GPU it is the wrong kernel (one thread walks a whole `k` row, checklist
// C2) and `lora_xa_rows` is selected instead. The parity gate for the pair is
// `crates/model/tests/lora_bank_rows.rs`, which runs on both.
//
// Dispatch: rows * r_max threads.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.r_max) { return; }
    let i = idx / p.r_max;
    let j = idx - i * p.r_max;
    let s = slot[p.row0 + i];
    if (s >= p.slots || j >= min(rank[s], p.r_max)) { return; }
    let xbase = (p.row0 + i) * p.k;
    let abase = (s * p.r_max + j) * p.k;
    var acc = 0.0;
    for (var c = 0u; c < p.k; c = c + 1u) {
        acc = acc + x[xbase + c] * a_bank[abase + c];
    }
    t[idx] = acc;
}
