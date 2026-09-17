// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Three-factor weight update at sited synapses: w <- clamp(w + eta*m[site]*e, lo, hi)
// @how   one thread per synapse
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The third factor lands here, and this is the only kernel that moves a
// weight.
//
//   w      : [nnz]  synaptic weights, updated in place
//   e      : [nnz]  eligibility trace
//   site   : [nnz]  compartment modulating each synapse; 0 = does not learn
//   m      : [c]    modulator level per compartment (`neuro_modulate`)
//   params : nnz, eta, w_min, w_max
//
// Dispatch: nnz invocations.
//
// Every synapse being eligible is a modelling choice, and in an animal it is
// the wrong one. A connectome is 16 million edges; the population where
// *Drosophila* associative learning is actually known to happen is about
// 18,000 of them. A rule that updates all of them is not a stronger version of
// that, it is a different claim, and it is the claim that makes a search
// stop needing the connectome. `site` is what restricts it, and compartment 0
// is reserved so that "not plastic" is an early return rather than a small
// update.
//
// A zero modulator leaves weights BIT-IDENTICAL rather than merely close,
// because `w + eta*0*e` is exactly `w`. That is what lets "plasticity off",
// "reward shuffled" and "unpaired control" be tested as exact no-ops instead
// of as differences small enough to argue about.
//
// The clamp is not decoration. An unbounded reward-modulated rule is
// positively unstable: a synapse that helps earn reward is strengthened,
// which makes it more likely to fire together with its target, which makes it
// more eligible next time. Real synapses have a finite conductance; so does
// this one.
//
// Sign is NOT preserved across the clamp. A connectome's neurotransmitter
// prediction gives a prior on whether a synapse is excitatory or inhibitory,
// not a certainty, so a caller that wants Dale's law enforced sets `w_min`/
// `w_max` per population to bracket zero on one side. Baking a sign rule in
// here would assert something the data does not support.

struct Params {
    nnz: u32,
    eta: f32,
    w_min: f32,
    w_max: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> w:    array<f32>;
@group(0) @binding(2) var<storage, read>       e:    array<f32>;
@group(0) @binding(3) var<storage, read>       site: array<u32>;
@group(0) @binding(4) var<storage, read>       m:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid.y * nwg.x * 64u + gid.x;
    if (k >= p.nnz) { return; }
    let c = site[k];
    if (c == 0u) { return; }
    w[k] = clamp(w[k] + p.eta * m[c] * e[k], p.w_min, p.w_max);
}
