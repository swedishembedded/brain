// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Compact a dense MoE gate matrix into the top_k expert ids each row selected
// @how   one thread per row, single scan (no local array, no barrier)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// `router_gate.wgsl` (or `router_gate_train.wgsl`) already computes, per row,
// exactly `top_k` nonzero columns in its dense `[rows, n_experts]` gate output
// -- it just never writes WHICH columns those were, only their renormalised
// values. A dense forward (every model in this repo today) doesn't need to
// know: it loops every expert unconditionally and lets `moe_linear_gated
// .wgsl`'s per-row gate check discard the non-selected ones. That is exactly
// right when `rows` is large enough to amortise a GEMM's per-dispatch
// overhead, but at decode's `rows=1` the 256-expert loop pays that overhead
// 256 times to do 8 experts' worth of real work. The fix needs the host to
// know which `top_k` experts a row actually chose, cheaply -- this kernel is
// that lookup: `gate -> top_ids[rows*top_k]`, a compact per-row list of
// selected expert COLUMN INDICES, small enough to read back every decode
// step regardless of `n_experts` (`top_k` u32s per row, not `n_experts`
// f32s).
//
// A new, additive sibling kernel rather than a `router_gate.wgsl` output --
// changing that kernel's own bind group would break every existing caller
// (`omni::thinker`/`talker`, `moondream::decoder`, `crates/moe`), all of
// which only ever want the dense gate, never these ids.
//
// Deliberately NOT `router_gate.wgsl`'s own top-k selection re-run: the gate
// matrix already IS the selection (nonzero == selected, by that kernel's own
// construction), so this is a linear scan-and-collect, not a re-derivation --
// no `sel_idx`-style scratch array is needed at all, just a running scalar
// write cursor. `top_k` bounds the OUTPUT width, never a local array; nothing
// here is sized by `n_experts`.
//
// By `router_gate.wgsl`'s construction there are exactly `top_k` nonzero
// entries per row, so the scan should never come up short. This kernel does
// not trust that from the outside, though: any slot the scan doesn't fill
// (defensive only) is padded with the sentinel value `n_experts` -- an id no
// real expert has, which every host-side consumer of `top_ids` must treat as
// "no expert here, skip".
//
// Cost: O(n_experts) scalar reads per row (one pass, early-out once `top_k`
// ids are found) -- trivial next to the GEMMs this feeds into.

struct Params {
    rows: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate:    array<f32>;
@group(0) @binding(2) var<storage, read_write> top_ids: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let r = gidx;
    if (r >= p.rows) { return; }
    let e = p.n_experts;
    let k = p.top_k;
    let base_in = r * e;
    let base_out = r * k;

    var slot: u32 = 0u;
    for (var eidx: u32 = 0u; eidx < e; eidx = eidx + 1u) {
        if (slot >= k) { break; }
        if (gate[base_in + eidx] > 0.0) {
            top_ids[base_out + slot] = eidx;
            slot = slot + 1u;
        }
    }
    // Defensive pad -- see this file's header. Should never fire given
    // router_gate.wgsl's own contract, but a kernel this cheap should not
    // rely on an upstream invariant it cannot itself check.
    for (var s: u32 = slot; s < k; s = s + 1u) {
        top_ids[base_out + s] = e;
    }
}
