// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Leaky integrate-and-fire membrane update, threshold, reset and refractory countdown
// @how   one thread per neuron
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One tick of a leaky integrate-and-fire population.
//
//   v      : [n]  membrane potential, updated in place
//   refrac : [n]  refractory ticks REMAINING, updated in place
//   isyn   : [n]  synaptic current from `syn_gather_csc`
//   drive  : [n]  external/sensory current injected this tick
//   spike  : [n]  OUT: 1.0 where the neuron fired, else 0.0
//   adapt  : [n]  spike-frequency adaptation current, updated in place
//   params : n, refrac_ticks, a (= dt/tau_m), v_rest, v_reset, v_th, r,
//            adapt_decay, adapt_increment
//
// Dispatch: n invocations.
//
// The integration is forward Euler on `tau dv/dt = (v_rest - v) + R*I`:
//
//     v <- v + a * (v_rest - v + r * (isyn + drive))
//
// which is worth writing down because it has an EXACT discrete solution for a
// constant input, not merely an approximate one:
//
//     v_k = v_inf + (v_0 - v_inf) * (1 - a)^k,   v_inf = v_rest + r*I
//
// That is what `neuro`'s analytic gate compares against, to the fp32 noise
// floor rather than to a tolerance. Gradcheck does not apply to a runtime
// whose learning rule is local rather than differentiated, so an exact
// closed form for the forward dynamics is the substitute, and it only exists
// because the discretisation was chosen to have one.
//
// ADAPTATION is the second state variable, and it is what lets a network of
// these oscillate at all. `adapt` decays geometrically every tick and jumps by
// `adapt_increment` on every spike, and it is SUBTRACTED from the input
// current - so a cell under a constant drive fires fast, slows, and settles at
// a lower rate. Two populations inhibiting each other cannot take turns
// without something like it: with only a membrane and a threshold, whichever
// side wins the first tick wins every tick, and the alternation a half-centre
// oscillator lives on never starts. That is not a hypothesis about this code,
// it is what `fly`'s gait analysis measured - coordination that appears on
// being driven and decays, never becoming periodic.
//
// The increment defaults to zero and zero is EXACTLY inert: nothing enters the
// variable, so nothing leaves it and `- adapt` is `- 0.0`. Every measurement
// taken before this existed reproduces bit-for-bit.
//
// Refractory neurons are CLAMPED to v_reset rather than merely barred from
// firing: a neuron that kept integrating while refractory would fire
// immediately on release and the absolute refractory period would set only
// the spike's timing, not the cell's state. The countdown is stored per
// neuron rather than as a "last spike tick" so a restored snapshot needs no
// notion of absolute time.

struct Params {
    n: u32,
    refrac_ticks: u32,
    a: f32,
    v_rest: f32,
    v_reset: f32,
    v_th: f32,
    r: f32,
    /// Fraction of the adaptation current that survives a tick,
    /// `exp(-dt / tau_w)`.
    adapt_decay: f32,
    /// How much of it one spike adds.
    adapt_increment: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> v:      array<f32>;
@group(0) @binding(2) var<storage, read_write> refrac: array<u32>;
@group(0) @binding(3) var<storage, read>       isyn:   array<f32>;
@group(0) @binding(4) var<storage, read>       drive:  array<f32>;
@group(0) @binding(5) var<storage, read_write> spike:  array<f32>;
@group(0) @binding(6) var<storage, read_write> adapt:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear index (identity for a 1D dispatch).
    let i = gid.y * nwg.x * 64u + gid.x;
    if (i >= p.n) { return; }

    // Decays whether or not the cell is refractory: it is a current in the
    // membrane, not a property of the spike that raised it, and freezing it
    // through the refractory period would make the adapted rate depend on the
    // refractory period twice over.
    let w = adapt[i] * p.adapt_decay;

    let left = refrac[i];
    if (left > 0u) {
        v[i] = p.v_reset;
        refrac[i] = left - 1u;
        spike[i] = 0.0;
        adapt[i] = w;
        return;
    }

    let vi = v[i] + p.a * (p.v_rest - v[i] + p.r * (isyn[i] + drive[i] - w));
    if (vi >= p.v_th) {
        v[i] = p.v_reset;
        refrac[i] = p.refrac_ticks;
        spike[i] = 1.0;
        adapt[i] = w + p.adapt_increment;
    } else {
        v[i] = vi;
        spike[i] = 0.0;
        adapt[i] = w;
    }
}
