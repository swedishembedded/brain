// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Three-factor weight update: w <- clamp(w + eta*e*delta, w_min, w_max)
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
//   params : nnz, eta*delta (premultiplied), w_min, w_max
//
// Dispatch: nnz invocations.
//
// `eta` and the neuromodulator `delta` arrive premultiplied as one scalar
// because that is how they are used: two uniforms that are only ever
// multiplied together invite a call site that sets one and forgets the other.
// A `delta` of zero therefore leaves every weight BIT-IDENTICAL rather than
// merely close, which is what lets "reward shuffled" and "plasticity off" be
// tested as exact no-ops instead of as small differences.
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
    eta_delta: f32,
    w_min: f32,
    w_max: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read>       e: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid.y * nwg.x * 64u + gid.x;
    if (k >= p.nnz) { return; }
    w[k] = clamp(w[k] + p.eta_delta * e[k], p.w_min, p.w_max);
}
