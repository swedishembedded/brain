// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A leaky integrate-and-fire population over a CSC connectome.

use gpu_core::{BufUsage, DeviceBuffer, Gpu};

use crate::csc::Csc;
use crate::seam::{DynamicalSystem, Plastic, Port, State, StepStats};

/// Kernel indices into [`crate::KERNELS`]. The order is this crate's own;
/// `Gpu::step` takes the index the device was built with.
const K_GATHER: usize = 0;
const K_LIF: usize = 1;
const K_TRACE: usize = 2;
const K_ELIG: usize = 3;
const K_MODULATE: usize = 4;
const K_LEARN: usize = 5;

/// Membrane dynamics, in the discretised form the kernel actually integrates.
///
/// `dt_over_tau` rather than `dt` and `tau` separately because that is the
/// only combination the update uses, and carrying both would let a caller set
/// a pair the kernel cannot honour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LifParams {
    /// `dt / tau_m`. Must be in `(0, 1]`: at `a > 1` the Euler update
    /// overshoots the resting potential and the membrane oscillates, which is
    /// a numerical artefact rather than a neuron.
    pub dt_over_tau: f32,
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_th: f32,
    /// Input resistance: the scale from current to membrane volts.
    pub r: f32,
    /// Absolute refractory period, in ticks.
    pub refrac_ticks: u32,
    /// `dt / tau_syn` for EXCITATORY input. `1.0` is an instantaneous synapse.
    ///
    /// A spike is an impulse and a synapse is not. Without a synaptic time
    /// constant the summed input to a population is noise at the tick rate,
    /// and a recurrent loop through three neurons closes in three ticks, so
    /// the only rhythm such a network can hold is one the integration step
    /// chose. This is what puts a network oscillator's period in the range an
    /// animal moves at.
    pub dt_over_tau_syn: f32,
    /// Fraction of the SPIKE-FREQUENCY ADAPTATION current that survives a
    /// tick, `exp(-dt / tau_w)`.
    ///
    /// The slow variable. A cell's own recent firing subtracts from its input,
    /// so a constant drive produces a fast onset that settles to a lower rate.
    /// This is the ingredient that lets a pair of mutually inhibiting
    /// populations take turns: without a slow variable the side that wins the
    /// first tick wins every tick, and the network can coordinate but never
    /// oscillate. See `crates/kernels/wgsl/lif_step.wgsl`.
    pub adapt_decay: f32,
    /// How much adaptation current one spike adds. `0.0` - the default -
    /// disables adaptation EXACTLY: nothing ever enters the variable, so
    /// nothing leaves it and the membrane update is unchanged bit for bit.
    pub adapt_increment: f32,
    /// The same for INHIBITORY input, which is deliberately a separate number.
    ///
    /// A reciprocal-inhibition oscillator's period is set by how long the
    /// inhibition takes to build and release relative to the excitation that
    /// provoked it. Give both the same time constant and the loop has no phase
    /// lag to turn into a rhythm. Fast cholinergic excitation against slower
    /// GABAergic inhibition is also simply what a fly has.
    pub dt_over_tau_inh: f32,
}

impl Default for LifParams {
    /// Dimensionless defaults in the usual millivolt-shaped range, chosen so
    /// a unit-current input is comfortably suprathreshold.
    fn default() -> Self {
        LifParams {
            dt_over_tau: 0.1,
            v_rest: 0.0,
            v_reset: 0.0,
            v_th: 1.0,
            r: 1.0,
            refrac_ticks: 0,
            // Instantaneous, so the default reproduces every measurement
            // taken before synapses had a time constant.
            dt_over_tau_syn: 1.0,
            dt_over_tau_inh: 1.0,
            // Off, for the same reason: the default is the model this runtime
            // had before it had a slow variable.
            adapt_decay: 0.9,
            adapt_increment: 0.0,
        }
    }
}

impl LifParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.dt_over_tau > 0.0 && self.dt_over_tau <= 1.0) {
            return Err(format!("dt_over_tau must be in (0, 1], got {}", self.dt_over_tau));
        }
        for (name, a) in [("dt_over_tau_syn", self.dt_over_tau_syn), ("dt_over_tau_inh", self.dt_over_tau_inh)] {
            if !(a > 0.0 && a <= 1.0) {
                return Err(format!("{name} must be in (0, 1], got {a}"));
            }
        }
        if !(0.0..1.0).contains(&self.adapt_decay) {
            return Err(format!("adapt_decay must be in [0, 1), got {} - a current that never decays is not adaptation, it is a ramp to silence", self.adapt_decay));
        }
        if self.adapt_increment < 0.0 {
            return Err(format!("adapt_increment must not be negative, got {} - a negative one is positive feedback", self.adapt_increment));
        }
        if self.v_th <= self.v_reset {
            return Err(format!("v_th ({}) must exceed v_reset ({})", self.v_th, self.v_reset));
        }
        Ok(())
    }

    /// The exact discrete membrane trajectory for a CONSTANT input current,
    /// starting from `v0`, ignoring threshold.
    ///
    /// The Euler update `v <- v + a(v_rest - v + r*I)` is a geometric
    /// recurrence, so it has a closed form rather than merely an
    /// approximation:
    ///
    /// ```text
    /// v_k = v_inf + (v0 - v_inf)(1 - a)^k,   v_inf = v_rest + r*I
    /// ```
    ///
    /// This is the runtime's correctness oracle. `gradcheck` cannot gate a
    /// system whose learning rule is local rather than differentiated, so the
    /// substitute is an independently derived closed form for the forward
    /// dynamics -- and it exists only because the discretisation was chosen to
    /// have one.
    pub fn analytic_v(&self, v0: f32, current: f32, ticks: u32) -> f32 {
        let v_inf = self.v_rest + self.r * current;
        v_inf + (v0 - v_inf) * (1.0 - self.dt_over_tau).powi(ticks as i32)
    }
}

/// Three-factor plasticity: eligibility accumulates locally, a neuromodulator
/// decides whether any of it becomes learning.
///
/// Not a gradient. Nothing here differentiates a loss; the weight change at a
/// synapse is a product of three quantities that synapse can actually see -
/// its own recent pre/post coincidence, and a scalar broadcast to the whole
/// population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticityParams {
    /// Presynaptic activity-trace decay per tick, `exp(-dt/tau_pre)`.
    pub pre_decay: f32,
    /// Postsynaptic activity-trace decay per tick.
    pub post_decay: f32,
    /// Eligibility-trace decay per tick. Usually the slowest of the three:
    /// it sets how long after acting a creature can still be credited for it.
    pub elig_decay: f32,
    /// Learning rate.
    pub eta: f32,
    /// Weight bounds. An unbounded reward-modulated rule is positively
    /// unstable, so these are required rather than optional.
    pub w_min: f32,
    pub w_max: f32,
}

impl Default for PlasticityParams {
    fn default() -> Self {
        PlasticityParams { pre_decay: 0.9, post_decay: 0.9, elig_decay: 0.95, eta: 0.01, w_min: -1.0, w_max: 1.0 }
    }
}

impl PlasticityParams {
    pub fn validate(&self) -> Result<(), String> {
        for (name, d) in [("pre_decay", self.pre_decay), ("post_decay", self.post_decay), ("elig_decay", self.elig_decay)] {
            if !(0.0..1.0).contains(&d) {
                return Err(format!("{name} must be in [0, 1), got {d} - a decay of 1 never forgets"));
            }
        }
        if self.w_min >= self.w_max {
            return Err(format!("w_min ({}) must be below w_max ({})", self.w_min, self.w_max));
        }
        Ok(())
    }
}

/// A spiking network: a connectome plus the state that makes it run.
pub struct SpikingNet {
    gpu: Gpu,
    n: u32,
    params: LifParams,
    tick: u64,

    // Graph (immutable for as long as the connectome is).
    indptr: DeviceBuffer,
    pre: DeviceBuffer,
    w: DeviceBuffer,

    // State.
    v: DeviceBuffer,
    refrac: DeviceBuffer,
    spike: DeviceBuffer,
    isyn: DeviceBuffer,
    /// Spike-frequency adaptation current, one per neuron.
    adapt: DeviceBuffer,
    /// Per-neuron multipliers on `dt/tau` and on the input resistance. Both
    /// are 1.0 unless [`SpikingNet::set_cell_scales`] says otherwise, and at
    /// 1.0 the arithmetic is exactly the uniform model's.
    tau_scale: DeviceBuffer,
    gain_scale: DeviceBuffer,
    /// Excitatory and inhibitory current, interleaved `(e, i)` per neuron, so
    /// each can carry across ticks on its own time constant.
    syn: DeviceBuffer,
    drive: DeviceBuffer,

    // The connectome's own weights, kept so `reset` can restore them after
    // plasticity has moved them.
    w0: Vec<f32>,

    // Plasticity. `None` until `enable_plasticity` allocates the traces: a
    // forward-only run should not pay for an eligibility buffer the size of
    // the edge list.
    plast: Option<Plasticity>,
}

/// The buffers and settings plasticity needs, allocated only when it is on.
struct Plasticity {
    params: PlasticityParams,
    on: bool,
    /// This tick's neuromodulator for the host-driven compartments. Consumed
    /// by `step` and cleared, so a reward delivered once is applied once.
    delta: f32,
    /// Whether `delta` has been written since it was last consumed. Without
    /// it, clearing on every tick would mean rewriting the whole modulator
    /// buffer 500 times a second to write zeros over zeros.
    delta_pending: bool,
    x_pre: DeviceBuffer,
    x_post: DeviceBuffer,
    elig: DeviceBuffer,
    /// Where learning is allowed and what modulates it.
    sites: crate::Sites,
    /// Compartment of each edge, on the device.
    site_of_edge: DeviceBuffer,
    /// The compartments' source lists, CSC-style.
    site_indptr: DeviceBuffer,
    site_source: DeviceBuffer,
    site_gain: DeviceBuffer,
    /// Modulator level, one per compartment.
    modulator: DeviceBuffer,
}

impl SpikingNet {
    /// Build on an existing device. The caller owns device creation because
    /// this workspace allows ONE device per process (`Gpu::share` /
    /// `gpu_core::testgpu` for tests) and a crate that built its own would be
    /// the second one.
    pub fn new(gpu: Gpu, csc: &Csc, params: LifParams) -> Result<SpikingNet, String> {
        csc.validate()?;
        params.validate()?;
        let n = csc.n;
        if n == 0 {
            return Err("a connectome with no neurons cannot be stepped".to_string());
        }
        let bytes = |k: usize| (k.max(1) * 4) as u64;
        // The graph is written once and never read back; the state is both.
        let fixed = BufUsage::STORAGE | BufUsage::COPY_DST;
        let live = BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC;

        let indptr = gpu.buffer("neuro.indptr", bytes(csc.indptr.len()), fixed);
        let pre = gpu.buffer("neuro.pre", bytes(csc.pre.len()), fixed);
        let w = gpu.buffer("neuro.w", bytes(csc.w.len()), live);
        gpu.write(&indptr, &csc.indptr);
        gpu.write(&pre, &csc.pre);
        gpu.write_f32(&w, &csc.w);

        let nb = bytes(n as usize);
        let v = gpu.buffer("neuro.v", nb, live);
        let refrac = gpu.buffer("neuro.refrac", nb, live);
        let spike = gpu.buffer("neuro.spike", nb, live);
        let isyn = gpu.buffer("neuro.isyn", nb, live);
        let adapt = gpu.buffer("neuro.adapt", nb, live);
        let syn = gpu.buffer("neuro.syn", bytes(2 * n as usize), live);
        let drive = gpu.buffer("neuro.drive", nb, live);
        let tau_scale = gpu.buffer("neuro.tau_scale", nb, live);
        let gain_scale = gpu.buffer("neuro.gain_scale", nb, live);
        gpu.write_f32(&tau_scale, &vec![1.0; n as usize]);
        gpu.write_f32(&gain_scale, &vec![1.0; n as usize]);

        let mut net = SpikingNet {
            gpu,
            n,
            params,
            tick: 0,
            indptr,
            pre,
            w,
            v,
            refrac,
            spike,
            isyn,
            adapt,
            tau_scale,
            gain_scale,
            syn,
            drive,
            w0: csc.w.clone(),
            plast: None,
        };
        net.reset(0);
        Ok(net)
    }

    pub fn neurons(&self) -> u32 {
        self.n
    }

    pub fn params(&self) -> LifParams {
        self.params
    }

    /// Change the membrane and synapse parameters of a running network.
    ///
    /// They are uniform across the population and are passed to the kernel
    /// every tick, so this is a struct assignment rather than a rebuild -
    /// which is what lets a search over dynamics cost one episode per
    /// candidate instead of one graph upload per candidate. The DYNAMICAL
    /// state is left alone: a caller changing the time constants mid-episode
    /// gets exactly that, and one that wants a fresh start calls
    /// [`SpikingNet::reset_state`].
    pub fn set_params(&mut self, params: LifParams) -> Result<(), String> {
        params.validate()?;
        self.params = params;
        Ok(())
    }

    /// Allocate the traces and turn three-factor plasticity on.
    ///
    /// Separate from `new` because a forward-only run - a parity check, a
    /// replay, a frozen creature being evaluated - should not allocate an
    /// eligibility buffer the size of the edge list, which is the largest
    /// array in the system.
    pub fn enable_plasticity(&mut self, params: PlasticityParams) -> Result<(), String> {
        self.enable_plasticity_at(params, crate::Sites::everywhere(self.w0.len()))
    }

    /// Turn plasticity on only where `sites` allows it.
    ///
    /// The difference from [`Self::enable_plasticity`] is the whole point of
    /// the exercise rather than an optimisation. A rule free to change every
    /// synapse can solve a task by walking away from the measured wiring, so
    /// a result obtained that way says nothing about the connectome. A rule
    /// confined to one identified population, driven by identified cells,
    /// cannot: whatever it achieves, it achieved through the graph.
    pub fn enable_plasticity_at(&mut self, params: PlasticityParams, sites: crate::Sites) -> Result<(), String> {
        params.validate()?;
        sites.validate(self.w0.len(), self.n)?;
        let live = BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC;
        let n = self.n as usize;
        let nnz = self.w0.len();
        let x_pre = self.gpu.buffer("neuro.x_pre", (n.max(1) * 4) as u64, live);
        let x_post = self.gpu.buffer("neuro.x_post", (n.max(1) * 4) as u64, live);
        let elig = self.gpu.buffer("neuro.elig", (nnz.max(1) * 4) as u64, live);
        self.gpu.write_f32(&x_pre, &vec![0.0; n]);
        self.gpu.write_f32(&x_post, &vec![0.0; n]);
        self.gpu.write_f32(&elig, &vec![0.0; nnz]);

        let (of_edge, indptr, source, gain) = sites.parts();
        let nc = gain.len();
        let site_of_edge = self.gpu.buffer("neuro.site", (of_edge.len().max(1) * 4) as u64, live);
        let site_indptr = self.gpu.buffer("neuro.site_indptr", (indptr.len().max(1) * 4) as u64, live);
        let site_source = self.gpu.buffer("neuro.site_source", (source.len().max(1) * 4) as u64, live);
        let site_gain = self.gpu.buffer("neuro.site_gain", (nc.max(1) * 4) as u64, live);
        let modulator = self.gpu.buffer("neuro.modulator", (nc.max(1) * 4) as u64, live);
        self.gpu.write(&site_of_edge, of_edge);
        self.gpu.write(&site_indptr, indptr);
        self.gpu.write(&site_source, if source.is_empty() { &[0u32][..] } else { source });
        self.gpu.write_f32(&site_gain, gain);
        self.gpu.write_f32(&modulator, &vec![0.0; nc]);

        self.plast = Some(Plasticity {
            params,
            on: true,
            delta: 0.0,
            delta_pending: false,
            x_pre,
            x_post,
            elig,
            sites,
            site_of_edge,
            site_indptr,
            site_source,
            site_gain,
            modulator,
        });
        Ok(())
    }

    /// The modulator level of every compartment, index 0 first.
    ///
    /// The instrument that says a reinforcer actually reached the synapses it
    /// was meant to reach, which is otherwise only visible as a weight change
    /// that could have come from anywhere.
    pub fn modulator(&self) -> Vec<f32> {
        match &self.plast {
            Some(pl) => self.gpu.read(&pl.modulator, pl.sites.len()),
            None => Vec::new(),
        }
    }

    /// Where this network is allowed to learn.
    pub fn sites(&self) -> Option<&crate::Sites> {
        self.plast.as_ref().map(|pl| &pl.sites)
    }

    /// The eligibility trace, one entry per edge. Empty when plasticity was
    /// never enabled.
    pub fn eligibility(&self) -> Vec<f32> {
        match &self.plast {
            Some(pl) => self.gpu.read(&pl.elig, self.w0.len()),
            None => Vec::new(),
        }
    }

    /// The synaptic weights as they stand now.
    pub fn weights(&self) -> Vec<f32> {
        self.gpu.read(&self.w, self.w0.len())
    }

    /// Put the host's `modulate` value into every compartment it owns.
    ///
    /// "Owns" means a compartment with no modulatory neurons of its own: an
    /// experimenter's reinforcer, as against dopamine the network produced.
    /// Both kinds live in one buffer and the learning kernel cannot tell them
    /// apart, which is deliberate - it is the same rule either way, and the
    /// difference is only in who decided.
    fn publish_delta(&mut self) {
        let Some(pl) = &mut self.plast else { return };
        if !pl.delta_pending {
            return;
        }
        pl.delta_pending = false;
        let mut level = self.gpu.read(&pl.modulator, pl.sites.len());
        for (c, slot) in level.iter_mut().enumerate() {
            if pl.sites.host_driven(c as u32) {
                *slot = pl.delta;
            }
        }
        self.gpu.write_f32(&pl.modulator, &level);
    }

    /// The learning half of a tick, appended to the forward steps.
    ///
    /// Order matters and is the reason this is not three separate submissions:
    /// the traces must be updated with THIS tick's spikes before the
    /// eligibility trace reads them, and the weight update must see the
    /// eligibility the same tick produced.
    fn learn_steps(&self, steps: &mut Vec<gpu_core::Step>) {
        let Some(pl) = &self.plast else { return };
        let p = pl.params;
        steps.push(self.gpu.step(K_TRACE, &[&pl.x_pre, &self.spike], &[self.n, p.pre_decay.to_bits()], self.n));
        steps.push(self.gpu.step(K_TRACE, &[&pl.x_post, &self.spike], &[self.n, p.post_decay.to_bits()], self.n));
        steps.push(self.gpu.step(
            K_ELIG,
            &[&self.indptr, &self.pre, &pl.elig, &pl.x_pre, &pl.x_post],
            &[self.n, p.elig_decay.to_bits()],
            self.n,
        ));
        // The third factor, computed from the network's own dopaminergic
        // cells. Runs whether or not `on`, so that turning learning off
        // freezes the weights without also blinding the instrument that says
        // what the reinforcement was doing.
        steps.push(self.gpu.step(
            K_MODULATE,
            &[&pl.modulator, &pl.site_indptr, &pl.site_source, &pl.site_gain, &self.spike],
            &[pl.sites.len() as u32, pl.sites.decay.to_bits()],
            pl.sites.len() as u32,
        ));
        if pl.on {
            // A modulator of 0 makes the update bit-identical rather than
            // small, and an edge in the inert compartment is not written at
            // all.
            steps.push(self.gpu.step(
                K_LEARN,
                &[&self.w, &pl.elig, &pl.site_of_edge, &pl.modulator],
                &[self.w0.len() as u32, p.eta.to_bits(), p.w_min.to_bits(), p.w_max.to_bits()],
                self.w0.len() as u32,
            ));
        }
    }

    /// The LIF kernel's `Params` block: two counts then five f32 bit patterns,
    /// in declaration order. A mismatched param list here is silently wrong
    /// rather than a crash, which is why the order is written once.
    fn lif_params(&self) -> [u32; 9] {
        [
            self.n,
            self.params.refrac_ticks,
            self.params.dt_over_tau.to_bits(),
            self.params.v_rest.to_bits(),
            self.params.v_reset.to_bits(),
            self.params.v_th.to_bits(),
            self.params.r.to_bits(),
            self.params.adapt_decay.to_bits(),
            self.params.adapt_increment.to_bits(),
        ]
    }
}

impl SpikingNet {
    /// Clear the dynamical state and start again, KEEPING whatever the weights
    /// have become.
    ///
    /// Distinct from [`DynamicalSystem::reset`], which also restores the
    /// connectome's original weights. Both are needed and confusing them is
    /// expensive: an episode loop that calls the full reset unlearns between
    /// every episode, so every episode is byte-identical and the experiment
    /// reports a perfectly reproducible failure to learn. That is exactly what
    /// happened here before this existed.
    pub fn reset_state(&mut self) {
        let n = self.n as usize;
        self.gpu.write_f32(&self.v, &vec![self.params.v_rest; n]);
        self.gpu.write(&self.refrac, &vec![0u32; n]);
        self.gpu.write_f32(&self.spike, &vec![0.0; n]);
        self.gpu.write_f32(&self.isyn, &vec![0.0; n]);
        self.gpu.write_f32(&self.adapt, &vec![0.0; n]);
        self.gpu.write_f32(&self.syn, &vec![0.0; 2 * n]);
        self.gpu.write_f32(&self.drive, &vec![0.0; n]);
        if let Some(pl) = &mut self.plast {
            pl.delta = 0.0;
            pl.delta_pending = false;
            self.gpu.write_f32(&pl.x_pre, &vec![0.0; n]);
            self.gpu.write_f32(&pl.x_post, &vec![0.0; n]);
            self.gpu.write_f32(&pl.elig, &vec![0.0; self.w0.len()]);
            self.gpu.write_f32(&pl.modulator, &vec![0.0; pl.sites.len()]);
        }
        self.tick = 0;
    }

    /// Overwrite the synaptic weights.
    ///
    /// For a search over the weight space that is not plasticity: an
    /// optimiser choosing weights from outside, to answer what the structure
    /// could do under a stronger method than a local rule. Does NOT change
    /// what `reset` restores, so the connectome's own weights remain the
    /// baseline a run can always return to.
    pub fn set_weights(&mut self, w: &[f32]) -> Result<(), String> {
        if w.len() != self.w0.len() {
            return Err(format!("this connectome has {} edges, got {} weights", self.w0.len(), w.len()));
        }
        self.gpu.write_f32(&self.w, w);
        Ok(())
    }

    /// Give each neuron its own time constant and excitability.
    ///
    /// Both are MULTIPLIERS on the uniform [`LifParams`] values, so `1.0`
    /// everywhere is exactly the uniform model and is the control this is
    /// measured against. `tau_scale` multiplies `dt/tau`: a value of 2.0 is a
    /// membrane twice as fast, not twice as slow.
    ///
    /// This exists because a connectome does not contain physiology, and the
    /// physiology is what decides whether a circuit oscillates or rings. The
    /// parameters are not per neuron in practice - they are shared across
    /// cells of the same published class, which is what keeps the count in the
    /// hundreds instead of the hundreds of thousands and keeps the anatomy
    /// rather than the optimiser in charge of what the network is.
    pub fn set_cell_scales(&mut self, tau_scale: &[f32], gain_scale: &[f32]) -> Result<(), String> {
        let n = self.n as usize;
        for (name, v) in [("tau_scale", tau_scale), ("gain_scale", gain_scale)] {
            if v.len() != n {
                return Err(format!("{name} expects {n} values, got {}", v.len()));
            }
            if let Some(bad) = v.iter().find(|x| !x.is_finite() || **x <= 0.0) {
                return Err(format!("{name} must be positive and finite, got {bad}"));
            }
        }
        self.gpu.write_f32(&self.tau_scale, tau_scale);
        self.gpu.write_f32(&self.gain_scale, gain_scale);
        Ok(())
    }

    /// The connectome's weights as they were loaded, before any plasticity.
    pub fn initial_weights(&self) -> &[f32] {
        &self.w0
    }
}

impl DynamicalSystem for SpikingNet {
    fn reset(&mut self, _seed: u64) {
        // Deterministic and seed-independent today: every neuron starts at
        // rest with no spike history. The seed is in the signature because a
        // stochastic initialisation is the obvious next variant and callers
        // should already be passing one.
        let n = self.n as usize;
        self.gpu.write_f32(&self.v, &vec![self.params.v_rest; n]);
        self.gpu.write(&self.refrac, &vec![0u32; n]);
        self.gpu.write_f32(&self.spike, &vec![0.0; n]);
        self.gpu.write_f32(&self.isyn, &vec![0.0; n]);
        self.gpu.write_f32(&self.adapt, &vec![0.0; n]);
        self.gpu.write_f32(&self.syn, &vec![0.0; 2 * n]);
        self.gpu.write_f32(&self.drive, &vec![0.0; n]);
        self.gpu.write_f32(&self.w, &self.w0.clone());
        if let Some(pl) = &mut self.plast {
            // Traces are state, not configuration: a reset that left them
            // running would leak one episode's activity into the next.
            pl.delta = 0.0;
            pl.delta_pending = false;
            self.gpu.write_f32(&pl.x_pre, &vec![0.0; n]);
            self.gpu.write_f32(&pl.x_post, &vec![0.0; n]);
            self.gpu.write_f32(&pl.elig, &vec![0.0; self.w0.len()]);
            self.gpu.write_f32(&pl.modulator, &vec![0.0; pl.sites.len()]);
        }
        self.tick = 0;
    }

    fn step(&mut self) -> StepStats {
        let gather = self.gpu.step(
            K_GATHER,
            &[&self.indptr, &self.pre, &self.w, &self.spike, &self.isyn, &self.syn],
            &[
                self.n,
                (1.0 - self.params.dt_over_tau_syn).to_bits(),
                (1.0 - self.params.dt_over_tau_inh).to_bits(),
            ],
            self.n,
        );
        let lif = self.gpu.step(
            K_LIF,
            &[
                &self.v,
                &self.refrac,
                &self.isyn,
                &self.drive,
                &self.spike,
                &self.adapt,
                &self.tau_scale,
                &self.gain_scale,
            ],
            &self.lif_params(),
            self.n,
        );
        // One submission, in order: the gather reads last tick's spikes, the
        // LIF overwrites them, and the learning steps read what the LIF just
        // wrote. None of these may be reordered.
        let mut steps = vec![gather, lif];
        self.publish_delta();
        self.learn_steps(&mut steps);
        self.gpu.submit(&[], &steps);
        if let Some(pl) = &mut self.plast {
            // A reward delivered once is applied once. Leaving delta set
            // would silently turn a single reinforcement into a standing one.
            pl.delta = 0.0;
        }
        self.tick += 1;
        StepStats { tick: self.tick }
    }

    fn drive(&mut self, port: Port, values: &[f32]) -> Result<(), String> {
        if port != Port::Drive {
            return Err(format!("{port:?} is not writable"));
        }
        if values.len() != self.n as usize {
            return Err(format!("drive expects {} values, got {}", self.n, values.len()));
        }
        self.gpu.write_f32(&self.drive, values);
        Ok(())
    }

    fn read(&self, port: Port, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.n as usize {
            return Err(format!("read expects {} values, got {}", self.n, out.len()));
        }
        let buf = match port {
            Port::Spike => &self.spike,
            Port::Membrane => &self.v,
            Port::Current => &self.isyn,
            Port::Drive => &self.drive,
        };
        out.copy_from_slice(&self.gpu.read(buf, self.n as usize));
        Ok(())
    }

    fn port_len(&self, _port: Port) -> usize {
        self.n as usize
    }

    fn snapshot(&self) -> State {
        let n = self.n as usize;
        State {
            v: self.gpu.read(&self.v, n),
            // `Gpu::read` returns f32 words; a u32 buffer's values are those
            // words' BIT PATTERNS, not their numeric value (`model::paged`
            // reads its own u32 buffers the same way). `as u32` here would
            // silently round a refractory counter to garbage.
            refrac: self.gpu.read(&self.refrac, n).iter().map(|x| x.to_bits()).collect(),
            spike: self.gpu.read(&self.spike, n),
            w: self.gpu.read(&self.w, self.w0.len()),
            tick: self.tick,
        }
    }

    fn restore(&mut self, state: &State) -> Result<(), String> {
        let n = self.n as usize;
        if state.v.len() != n || state.refrac.len() != n || state.spike.len() != n {
            return Err(format!("snapshot is for {} neurons, this network has {n}", state.v.len()));
        }
        if !state.w.is_empty() && state.w.len() != self.w0.len() {
            return Err(format!("snapshot has {} weights, this connectome has {}", state.w.len(), self.w0.len()));
        }
        self.gpu.write_f32(&self.v, &state.v);
        self.gpu.write(&self.refrac, &state.refrac);
        self.gpu.write_f32(&self.spike, &state.spike);
        if !state.w.is_empty() {
            self.gpu.write_f32(&self.w, &state.w);
        }
        self.tick = state.tick;
        Ok(())
    }
}

impl Plastic for SpikingNet {
    fn set_plasticity(&mut self, on: bool) {
        if let Some(pl) = &mut self.plast {
            pl.on = on;
        }
    }

    fn plasticity(&self) -> bool {
        self.plast.as_ref().is_some_and(|pl| pl.on)
    }

    fn modulate(&mut self, delta: f32) {
        if let Some(pl) = &mut self.plast {
            pl.delta = delta;
            pl.delta_pending = true;
        }
    }
}
