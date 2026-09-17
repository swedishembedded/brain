// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Leaky integrate-and-fire membrane update, with per-neuron physiology, threshold, reset and refractory countdown
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
    /// Excitatory and inhibitory REVERSAL potentials.
    ///
    /// A synapse is a conductance, not a current source. It opens a channel
    /// whose ions have an equilibrium potential, and the current it passes is
    /// `g * (E_rev - v)`: proportional to how far the membrane is from that
    /// equilibrium, and zero when it arrives. Two consequences, and this model
    /// had neither.
    ///
    /// Inhibition cannot hyperpolarise past its reversal potential. In a fly
    /// GABA-A reverses at about the resting potential, so inhibition at rest
    /// passes NO current at all and can only oppose depolarisation. With
    /// current-based synapses it instead subtracts a fixed amount however
    /// negative the cell already is, and in a balanced network the integrated
    /// inhibition scales with in-degree: measured on this fly's cord, the leg
    /// muscle with the most input sat 274 threshold-gaps below rest and could
    /// never fire again.
    ///
    /// And inhibition becomes DIVISIVE rather than subtractive: raising `g_i`
    /// raises the total conductance, which shrinks the cell's response to
    /// everything else. That is the gain control a mushroom body uses to keep
    /// its odour code sparse, and it is not expressible with current-based
    /// synapses at all.
    e_exc: f32,
    e_inh: f32,
    /// Inhibitory reversal potential: the floor a membrane cannot be pushed
    /// below.
    ///
    /// Not a numerical guard. Inhibition in a real neuron opens channels whose
    /// reversal potential is a little below rest, so however much of it
    /// arrives the membrane approaches that value and stops; it cannot be
    /// driven arbitrarily negative. Without the floor, a cell in a balanced
    /// network with a large in-degree is pushed hundreds of threshold-gaps
    /// below rest and never fires again - measured on this fly's own cord,
    /// where the leg muscle with the MOST input sat at -274 against a
    /// threshold of +1 while the one with the least fired at 119 Hz. The
    /// silence looked like weak excitation and was unrecoverable
    /// hyperpolarisation.
    v_min: f32,
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
// Excitatory and inhibitory conductance, interleaved `(e, i)` per neuron, as
// `syn_gather_csc` accumulates them. The inhibitory half arrives NEGATIVE
// because the gather splits on the sign of the weight; a conductance is a
// magnitude, so it is negated here.
@group(0) @binding(3) var<storage, read>       syn:    array<f32>;
@group(0) @binding(4) var<storage, read>       drive:  array<f32>;
@group(0) @binding(5) var<storage, read_write> spike:  array<f32>;
@group(0) @binding(6) var<storage, read_write> adapt:  array<f32>;
// Per-neuron MULTIPLIERS on the two parameters that are uniform above, so a
// population can differ in the way real cells differ without a kernel per
// cell type. Both are 1.0 by default, and `p.a * 1.0` is exactly `p.a`, so a
// network that does not use them is bit-identical to one compiled before they
// existed - which is what lets "the physiology is uniform" stand as a control
// rather than as an approximation.
//
// `tau_scale` multiplies `dt/tau`, so it is the RECIPROCAL of a time-constant
// multiplier: 2.0 means a membrane twice as fast. `gain_scale` multiplies the
// input resistance, which is how excitable the cell is per unit of synaptic
// current. These two and a tonic bias (which needs no kernel support, since
// the drive port already carries per-neuron current) are the physiological
// parameters a connectome does not contain and a fitted model has to supply.
@group(0) @binding(7) var<storage, read>       tau_scale:  array<f32>;
@group(0) @binding(8) var<storage, read>       gain_scale: array<f32>;

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

    let a = p.a * tau_scale[i];
    let r = p.r * gain_scale[i];
    let g_e = max(syn[2u * i], 0.0) * r;
    let g_i = max(-syn[2u * i + 1u], 0.0) * r;

    // Exponential Euler on the conductance-based membrane, which is the only
    // integrator that stays stable here. Collecting
    //
    //   tau dv/dt = (v_rest - v) + g_e (e_exc - v) + g_i (e_inh - v) + I
    //
    // into `tau dv/dt = A - B v` gives a leak that GROWS with the total
    // conductance, so a forward step of fixed size diverges exactly when the
    // input is strongest. Solving the linear equation over the tick instead is
    // unconditionally stable and is the exact answer for constant input, which
    // is also what keeps `LifParams::analytic_v` an oracle rather than an
    // approximation.
    let bb = 1.0 + g_e + g_i;
    let aa = p.v_rest + g_e * p.e_exc + g_i * p.e_inh + r * (drive[i] - w);
    let v_inf = aa / bb;
    let vi = max(v_inf + (v[i] - v_inf) * exp(-a * bb), p.v_min);
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
