// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-synapse eligibility trace over a CSC column: e <- e*decay + x_pre*x_post
// @how   one thread per postsynaptic neuron over its edge range, no barrier
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The per-synapse half of a three-factor learning rule.
//
//   indptr : [n+1]  CSC column starts
//   pre    : [nnz]  presynaptic neuron of each edge
//   e      : [nnz]  eligibility trace, updated in place
//   x_pre  : [n]    presynaptic activity trace
//   x_post : [n]    postsynaptic activity trace
//   params : n, decay
//
// Dispatch: n invocations (one thread per postsynaptic neuron).
//
// Each edge writes only its own slot, so unlike `syn_gather_csc` there is no
// reduction and NO barrier at all. The per-NEURON shape is kept anyway,
// rather than one invocation per edge, because the column index is the only
// place CSC records an edge's POSTsynaptic neuron: a per-edge kernel would
// have to binary-search `indptr` for it, paying a log(n) search per edge.
//
// One thread per neuron rather than 64, for the reason `syn_gather_csc`
// documents at length and measures: 64 invocations per neuron over a column
// averaging 58 edges spends most of a dispatch on invocations that load
// `indptr` twice and then find their strided slice empty.
//
// Eligibility is what makes a delayed reward assignable to the synapses that
// earned it: the trace accumulates coincident pre/post activity now, and a
// neuromodulator arriving some ticks later multiplies whatever is left of it
// (`neuro_learn`). Nothing here is a weight update -- an eligible synapse has
// not yet learned anything.

struct Params {
    n: u32,
    decay: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       indptr: array<u32>;
@group(0) @binding(2) var<storage, read>       pre:    array<u32>;
@group(0) @binding(3) var<storage, read_write> e:      array<f32>;
@group(0) @binding(4) var<storage, read>       x_pre:  array<f32>;
@group(0) @binding(5) var<storage, read>       x_post: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Indexed off the FLAT invocation id rather than workgroup/local ids: the
    // CPU JIT compiles a barrier-free kernel per invocation and requires
    // `global_invocation_id`.
    let post = gid.y * nwg.x * 64u + gid.x;
    if (post >= p.n) { return; }
    let lo = indptr[post];
    let hi = indptr[post + 1u];
    let xp = x_post[post];
    for (var k = lo; k < hi; k = k + 1u) {
        e[k] = e[k] * p.decay + x_pre[pre[k]] * xp;
    }
}
