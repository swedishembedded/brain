// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse synaptic current: one thread per postsynaptic neuron over its CSC column
// @how   scalar row walk, no work-group memory and no barrier
// @opt   2
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
//   syn    : [2n]   IN/OUT: the excitatory and inhibitory currents separately,
//                   interleaved as (e, i) per neuron, each carried across ticks
//                   by its own decay
//   params : n, decay_e, decay_i
//
// Dispatch: n invocations (one thread per postsynaptic neuron).
//
// GATHER, not scatter, and that is the whole design. The natural formulation
// of spike propagation is "for each spiking neuron, add its weight to every
// target", which is a scatter-add and wants an atomic. This engine has none:
// WGSL's integer atomics are portable but its compilers here are not (neither
// the CPU JIT nor the CUDA compiler lowers one), and an atomic kernel would
// forfeit the CPU backend that cross-backend parity is gated on. So the graph
// is stored transposed: each neuron reads its OWN incoming edges and writes
// its own output, and no two invocations ever touch the same address. Nothing
// needs to be locked because nothing is shared.
//
// ROW LENGTH is what picks the shape here, and it was measured rather than
// assumed. The obvious alternative - a 64-thread work-group per neuron,
// striding its edge range so a warp's loads coalesce - is what this kernel
// used to be, and it is the right shape for a matrix whose rows are thousands
// of elements long. A connectome's are not: the fly's nerve cord averages 226
// incoming edges per neuron unpruned and 58 at the synapse floor
// `fly::Wiring` actually runs, so the work-group's fixed per-neuron costs
// (a work-group memory allocation, a barrier, and a 128-element fold of the
// partials) dominated the handful of multiply-adds they existed to serve.
// Those costs scale with NEURONS, not with edges, which is the diagnostic: on
// the same cord the work-group shape cost 24.4 ms per tick over 1.37 M edges
// and 69.0 ms over 5.31 M, so pruning 74% of the connectome bought almost
// nothing. One neural tick of the MANC cord at the floor `fly::Wiring` runs
// (23,665 neurons / 1.37 M edges), measured by
// `crates/fly/examples/loop_profile.rs`:
//
//   shape                               Arc iGPU     22-thread CPU JIT
//   work-group per neuron, fold in all   24.4 ms          28.6 ms
//   work-group per neuron, fold in one   22.9 ms           5.7 ms
//   one thread per neuron (this)          5.2 ms           1.2 ms
//
// The middle row is there because the fold - every thread redundantly summing
// all 64 partials, the shape `rmsnorm_rows` uses - is the whole story on the
// CPU JIT (which runs a work-group as two 64-iteration lane loops) and almost
// none of it on the GPU. Fixing only that would have left the GPU where it
// was.
//
// The loss is coalescing: 32 neighbouring threads now walk 32 unrelated edge
// ranges instead of one contiguous one, which is exactly the pattern the
// `*_rows` kernel family exists to avoid. It is paid for many times over here
// because each thread's own walk is sequential, so a fetched cache line still
// serves that thread's next 16 edges. Re-measure before carrying this shape
// to a graph with long rows - the crossover is real, it is just far above any
// in-degree a nervous system has.
//
// `spike[pre[k]]` is the one indirect read and it cannot be made contiguous:
// it IS the connectome's irregularity. It is also why the spike vector is
// f32 0.0/1.0 rather than a packed bitmask - a bitmask would save 32x on a
// vector that is already the smallest buffer here (n floats against nnz
// pairs), in exchange for bit arithmetic on every edge. The whole vector is
// 92 KB on this cord, which is cache-resident on both backends. Measure
// before changing that.

struct Params {
    n: u32,
    // Fraction of last tick's current that survives into this one,
    // `exp(-dt / tau_syn)`, SEPARATELY for excitatory and inhibitory input.
    //
    // A spike is an impulse and a synapse is not: the postsynaptic current
    // from one release rises and decays over milliseconds. With no decay at
    // all a population's summed input is noise at the tick rate, and a
    // recurrent loop through three neurons closes in three ticks, so the only
    // oscillation such a network can hold has a period the integration step
    // chose rather than one the biology did.
    //
    // The two are separate because that is what makes a reciprocal-inhibition
    // oscillator oscillate. Such a circuit's period is set by how long the
    // inhibition takes to build and release relative to the excitation that
    // provokes it; give both the same time constant and the loop has no phase
    // lag to turn into a rhythm. Fast cholinergic excitation against slower
    // GABAergic inhibition is also simply what the animal has.
    //
    // Both at ZERO reproduces the instantaneous synapse this kernel had
    // before, up to the order the two partials are summed in.
    decay_e: f32,
    decay_i: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       indptr: array<u32>;
@group(0) @binding(2) var<storage, read>       pre:    array<u32>;
@group(0) @binding(3) var<storage, read>       w:      array<f32>;
@group(0) @binding(4) var<storage, read>       spike:  array<f32>;
@group(0) @binding(5) var<storage, read_write> isyn:   array<f32>;
@group(0) @binding(6) var<storage, read_write> syn:    array<f32>;
/// Presynaptic vesicle pool, one per neuron, maintained by `lif_step`. A
/// spike from a neuron that has been firing hard delivers less.
@group(0) @binding(7) var<storage, read>       depress: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear index (identity for a 1D dispatch): a connectome has
    // more neurons than a 1D grid is guaranteed to carry.
    let post = gid.y * nwg.x * 64u + gid.x;
    if (post >= p.n) { return; }

    let lo = indptr[post];
    let hi = indptr[post + 1u];
    var acc_e = 0.0;
    var acc_i = 0.0;
    for (var k = lo; k < hi; k = k + 1u) {
        let src = pre[k];
        let c = w[k] * spike[src] * depress[src];
        // Split by the SIGN OF THE WEIGHT, which is the presynaptic neuron's
        // transmitter, so excitation and inhibition can carry their own time
        // constants. Written as a clamp pair rather than as `if (w[k] < 0.0)`:
        // `spike` is 0.0 or 1.0 and never negative, so `c` has the weight's
        // sign whenever it is non-zero and is zero otherwise - which makes the
        // two forms produce the same two sums, including the case the branch
        // was written for (an edge whose source is silent contributes nothing
        // to either half rather than being filed as inhibitory).
        //
        // The branch was worth removing and not by a little: its condition is
        // a connectome's transmitter labels in edge order, which is as close
        // to unpredictable as data gets, and at one mispredict per edge the
        // CPU JIT spent more time recovering from it than reading the graph.
        // Measured on the MANC cord at the synapse floor, per neural tick:
        // 1.70 ms branched against 0.88 ms here on 22 CPU threads. On the Arc
        // iGPU it made no measurable difference (8.9 ms either way) - a GPU
        // predicates a two-sided branch instead of predicting it.
        acc_e = acc_e + max(c, 0.0);
        acc_i = acc_i + min(c, 0.0);
    }
    let e = syn[2u * post] * p.decay_e + acc_e;
    let inh = syn[2u * post + 1u] * p.decay_i + acc_i;
    syn[2u * post] = e;
    syn[2u * post + 1u] = inh;
    isyn[post] = e + inh;
}
