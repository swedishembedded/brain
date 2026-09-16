// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Exponentially-decaying activity trace: x <- x*decay + spike
// @how   one thread per neuron
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The per-neuron half of a three-factor learning rule.
//
//   x      : [n]  trace, updated in place
//   spike  : [n]  1.0 where the neuron fired this tick
//   params : n, decay
//
// Dispatch: n invocations.
//
// A synapse cannot see whether its pre- and postsynaptic neurons fired "close
// together in time" -- it only ever sees now. The standard resolution is to
// give each neuron a leaky memory of its own recent firing, so that
// coincidence becomes a product of two numbers available at the instant the
// eligibility trace is updated. `decay` is `exp(-dt/tau_trace)` and is
// computed by the host: it is a constant of the configuration, and
// recomputing an exponential per neuron per tick would be the same value
// every time.
//
// One kernel serves both the pre- and post-synaptic traces; they differ only
// in their decay constant and in which buffer they are handed.

struct Params {
    n: u32,
    decay: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x:     array<f32>;
@group(0) @binding(2) var<storage, read>       spike: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * nwg.x * 64u + gid.x;
    if (i >= p.n) { return; }
    x[i] = x[i] * p.decay + spike[i];
}
