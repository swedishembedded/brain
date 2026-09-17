// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-compartment neuromodulator from its own neurons' spikes
// @how   one thread per compartment over its source list, no barrier
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The third factor, computed INSIDE the network instead of handed in.
//
//   m      : [c]      modulator level per compartment, updated in place
//   indptr : [c+1]    source-list starts, CSC-style over compartments
//   source : [nsrc]   neuron index of each modulatory source
//   gain   : [c]      signed per-compartment gain on the summed drive
//   spike  : [n]      1.0 where a neuron fired this tick
//   params : c, decay
//
// Dispatch: c invocations (one thread per compartment).
//
// A reward-modulated rule usually gets its third factor from the experimenter:
// a scalar broadcast to every synapse, set by host code that already knows
// whether the animal did well. That is a perfectly good instrument and it is
// not a nervous system -- it puts the evaluation outside the animal, and every
// synapse hears the same thing, so two different reinforcers are the same
// signal with a different sign.
//
// In a fly the third factor is the firing of identified dopaminergic neurons,
// and they are wired: each innervates one compartment of the mushroom body and
// modulates only the synapses there. Sugar and shock are then different
// signals because they recruit different cells, which is why a fly can learn
// to approach one odour and avoid another rather than merely to do more or
// less of everything.
//
// `gain` is signed and carries the rule's direction. Dopamine paired with
// presynaptic activity DEPRESSES the Kenyon-cell output synapse rather than
// strengthening it, so the caller that builds a mushroom body passes a
// negative gain. It is not hard-coded here: the sign of a modulatory effect is
// a property of the receptor and the compartment, not of arithmetic.
//
// A compartment with NO sources is left exactly as it is found, which is what
// makes the experimenter-driven case a special case of this one rather than a
// separate path: the host writes the slot, this kernel does not touch it, and
// the learning kernel cannot tell the difference. Compartment 0 is reserved as
// permanently inert and never has sources, so it is covered by the same rule.

struct Params {
    n_comp: u32,
    decay: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> m:      array<f32>;
@group(0) @binding(2) var<storage, read>       indptr: array<u32>;
@group(0) @binding(3) var<storage, read>       source: array<u32>;
@group(0) @binding(4) var<storage, read>       gain:   array<f32>;
@group(0) @binding(5) var<storage, read>       spike:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let c = gid.y * nwg.x * 64u + gid.x;
    if (c >= p.n_comp) { return; }
    let lo = indptr[c];
    let hi = indptr[c + 1u];
    // Host-driven compartment: its level is owned by whoever wrote it.
    if (hi == lo) { return; }
    var s = 0.0;
    for (var k = lo; k < hi; k = k + 1u) {
        s = s + spike[source[k]];
    }
    m[c] = m[c] * p.decay + gain[c] * s;
}
