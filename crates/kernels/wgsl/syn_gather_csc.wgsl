// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse synaptic current: one WORKGROUP per postsynaptic neuron over its CSC column
// @how   64-thread workgroup tile over a contiguous edge range, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Propagate one tick of spikes through a connectome.
//
//   indptr : [n+1]  CSC column starts - edges of postsynaptic neuron `i` are
//                   `indptr[i] .. indptr[i+1]`, contiguous
//   pre    : [nnz]  presynaptic neuron index of each edge
//   w      : [nnz]  signed synaptic weight of each edge
//   spike  : [n]    1.0 where the presynaptic neuron fired this tick, else 0.0
//   isyn   : [n]    OUT: total synaptic current arriving at each neuron
//   params : n
//
// Dispatch: n * 64 invocations (one workgroup per postsynaptic neuron).
//
// GATHER, not scatter, and that is the whole design. The natural formulation
// of spike propagation is "for each spiking neuron, add its weight to every
// target", which is a scatter-add and wants an atomic. This engine has none:
// WGSL's integer atomics are portable but its compilers here are not (neither
// the CPU JIT nor the CUDA compiler lowers one), and an atomic kernel would
// forfeit the CPU backend that cross-backend parity is gated on. So the graph
// is stored transposed: each neuron reads its OWN incoming edges and writes
// its own output, and no two workgroups ever touch the same address. Nothing
// needs to be locked because nothing is shared.
//
// The edge range is contiguous, so the 64 threads of a workgroup walk it with
// stride 64 and every fetch of `pre`/`w` is fully used. One thread per neuron
// would instead give thread t a range starting at an arbitrary offset, and a
// warp's 32 loads would land in 32 unrelated places - the coalescing bug the
// `*_rows` family exists to avoid, here by construction rather than by
// retrofit.
//
// `spike[pre[k]]` is the one indirect read and it cannot be made contiguous:
// it IS the connectome's irregularity. It is also why the spike vector is
// f32 0.0/1.0 rather than a packed bitmask - a bitmask would save 32x on a
// vector that is already the smallest buffer here (n floats against nnz
// pairs, and nnz/n is ~226 on the fly's nerve cord), in exchange for bit
// arithmetic on every edge. Measure before changing that.
//
// One top-level barrier: the CPU JIT splits a body at one barrier and no
// more, so the 64 partials are folded redundantly by every thread (64 adds,
// cheaper than a second barrier) exactly as `rmsnorm_rows` does.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       indptr: array<u32>;
@group(0) @binding(2) var<storage, read>       pre:    array<u32>;
@group(0) @binding(3) var<storage, read>       w:      array<f32>;
@group(0) @binding(4) var<storage, read>       spike:  array<f32>;
@group(0) @binding(5) var<storage, read_write> isyn:   array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch): a
    // connectome has more neurons than a 1D grid is guaranteed to carry.
    let post = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (post >= p.n) { return; }

    let lo = indptr[post];
    let hi = indptr[post + 1u];
    var acc = 0.0;
    for (var k = lo + t; k < hi; k = k + 64u) {
        acc = acc + w[k] * spike[pre[k]];
    }
    partial[t] = acc;
    workgroupBarrier();

    var s = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        s = s + partial[i];
    }
    if (t == 0u) {
        isyn[post] = s;
    }
}
