// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seams a digital creature is built on.
//!
//! These are deliberately NOT `model::Model`. That trait is a static-graph,
//! token-batch, differentiable surface -- `vocab()`, `set_batch(Batch)`,
//! `forward() -> f32`, gradients into a `ParamStore` -- and its documentation
//! promises every implementor is gradient-checkable by construction. A
//! creature has no vocabulary, no scalar loss, and state that must survive
//! across calls; implementing `Model` for one would make `vocab()` a lie and
//! break the invariant that makes `Model` worth having.
//!
//! brain's established pattern for a new KIND of model is a sibling trait
//! unified at the capability layer, which is what `forecast::ForecastModel`,
//! `wm_core::WorldModel` and `model::serve::PagedDecoder` all are. This is
//! that, for systems whose defining property is that they have a state and a
//! clock.

/// A named input or output channel of a running system.
///
/// Ports are how a body reaches a nervous system without either one knowing
/// the other's internals: a sensory organ writes a port, a motor system reads
/// one. Which neurons back a port is the model's business, not the caller's.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Port {
    /// External current injected into every neuron this tick (write).
    Drive,
    /// 1.0 where a neuron fired on the last tick, else 0.0 (read).
    Spike,
    /// Membrane potential (read). For instrumentation and for the analytic
    /// gate; a body should be reading [`Port::Spike`].
    Membrane,
    /// Total synaptic current arriving at each neuron on the last tick
    /// (read). Exposed because it is the only direct view of what the gather
    /// computed: inferring it from the membrane requires a configuration in
    /// which nothing spikes, and a gather test that never sees a spike is a
    /// test of nothing.
    Current,
}

/// What one tick did. Cheap counters only: anything that needs a device
/// readback belongs in an explicit [`DynamicalSystem::read`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    /// Ticks elapsed since the last [`DynamicalSystem::reset`].
    pub tick: u64,
}

/// A stateful system with a clock: reset it, drive it, step it, read it.
///
/// There is no `dt` parameter on [`Self::step`] on purpose. A discretisation
/// constant that is baked into the coefficients at construction cannot be
/// varied per call, and a `dt` argument that is silently ignored is worse
/// than no argument at all. The tick length is a property of the system.
pub trait DynamicalSystem {
    /// Return to the initial state, deterministically for a fixed `seed`.
    fn reset(&mut self, seed: u64);

    /// Advance exactly one tick.
    fn step(&mut self) -> StepStats;

    /// Write a port's worth of input for the NEXT tick. `values.len()` must
    /// match [`Self::port_len`].
    fn drive(&mut self, port: Port, values: &[f32]) -> Result<(), String>;

    /// Read a port. `out.len()` must match [`Self::port_len`].
    fn read(&self, port: Port, out: &mut [f32]) -> Result<(), String>;

    /// How many values a port carries.
    fn port_len(&self, port: Port) -> usize;

    /// Everything needed to resume exactly where this call left off.
    fn snapshot(&self) -> State;

    /// Resume from a [`State`]. Restoring and replaying the same inputs must
    /// reproduce the same spikes bit-for-bit -- that is the property that
    /// makes "restart it and it still knows how to walk" a checkable claim
    /// rather than an impression.
    fn restore(&mut self, state: &State) -> Result<(), String>;
}

/// A system that learns from its own activity plus a neuromodulator, rather
/// than from a gradient computed elsewhere.
///
/// [`Self::set_plasticity`] is on the trait rather than in whatever experiment
/// happens to need it because "plasticity disabled" is one of the required
/// controls: a creature that cannot be frozen cannot be shown to have learned
/// anything.
pub trait Plastic {
    /// Turn weight updates on or off. Frozen weights still produce behaviour;
    /// they just stop changing.
    fn set_plasticity(&mut self, on: bool);

    /// Whether weight updates are currently applied.
    fn plasticity(&self) -> bool;

    /// Deliver a neuromodulator signal (the third factor) for this tick.
    fn modulate(&mut self, delta: f32);
}

/// A resumable snapshot of a running system.
///
/// Per-neuron and per-edge state travel together because they are only
/// meaningful together: weights without the membrane state they were learned
/// against describe a different animal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct State {
    /// Membrane potential, one per neuron.
    pub v: Vec<f32>,
    /// Refractory ticks remaining, one per neuron.
    pub refrac: Vec<u32>,
    /// Last tick's spikes, one per neuron.
    pub spike: Vec<f32>,
    /// Synaptic weights, one per edge. Empty when the system is not plastic
    /// and the weights are therefore still the connectome's own.
    pub w: Vec<f32>,
    /// Ticks elapsed at the moment of the snapshot.
    pub tick: u64,
}
