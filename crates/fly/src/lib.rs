// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composed animal.
//!
//! A connectome running as a spiking network, driving a MuJoCo body through
//! its own motor neurons, and reading that body back through its own
//! proprioceptors. Nothing here learns: this is the loop that has to exist and
//! run in real time before learning can be attached to it.
//!
//! ```text
//!   descending drive -> [ spiking VNC ] -> motor neurons -> actuators
//!                             ^                                 |
//!                             |                                 v
//!                       proprioceptors <------------------ joint state
//! ```
//!
//! Swedish Embedded AB implements closed-loop neural control against real and
//! simulated hardware, including the timing discipline that makes such a loop
//! hold its rate. If your team needs a controller closed around a physical
//! system, you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod learn;
pub mod reference;
pub mod sense;

use connectome::Connectome;
use flybody::MotorMap;
use gpu_core::Gpu;
use mujoco::{Data, Model, StateSpec};
use neuro::{DynamicalSystem, LifParams, Plastic, Port, SpikingNet};

pub use reference::{ImitationReward, Reference};
pub use sense::{Modality, Sensor};

/// How the three clocks in this loop relate.
///
/// They are genuinely different rates and pretending otherwise is how a
/// sensorimotor loop ends up silently running the body faster than the brain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    /// Neural ticks per control tick.
    pub neural_per_control: u32,
    /// Physics steps per control tick. flybody's own timestep is 1e-4 s, so
    /// 20 of them is the 2 ms (500 Hz) control period its walking tasks use.
    pub physics_per_control: u32,
}

impl Default for Timing {
    fn default() -> Self {
        Timing { neural_per_control: 1, physics_per_control: 20 }
    }
}

// Why `neural_per_control` defaults to 1, measured rather than assumed:
//
//   neural step, no readback : 0.068 ms
//   spike readback alone     : 0.138 ms
//   step THEN readback       : 0.807 ms
//
// The first two do not add up to the third, and that gap is the whole story.
// A step without a readback only SUBMITS work; the readback is what forces the
// sync that pays for it. So the cost of a neural tick is not the kernel time,
// it is one GPU round trip, and every extra neural tick inside a control tick
// buys another one. At 500 Hz the entire budget is 2.00 ms, so two neural
// ticks would spend 1.6 ms of it before the body moved at all.
//
// Raising this is legitimate and costs ~0.8 ms per extra tick. It is not
// something to raise without re-measuring.

/// How body state becomes current, and spikes become torque.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coupling {
    /// Current injected per radian of joint deflection.
    pub angle_gain: f32,
    /// Current injected per radian/second, for the load proxy.
    pub load_gain: f32,
    /// Muscle activation decay per control tick. A spike is an impulse; a
    /// muscle is not, so motor output is low-pass filtered rather than fed
    /// through raw.
    pub activation_decay: f32,
    /// Activation added per polarity-weighted motor spike.
    pub activation_gain: f32,
}

impl Default for Coupling {
    fn default() -> Self {
        Coupling { angle_gain: 10.0, load_gain: 0.05, activation_decay: 0.8, activation_gain: 0.05 }
    }
}

// `angle_gain` is not a taste parameter. A proprioceptor is an afferent: it
// has almost no incoming synapses, so the ONLY thing that can make it fire is
// the current injected here. Below threshold it injects current every tick,
// spikes never, and transmits nothing - while every other measurement in the
// loop looks healthy. Swept, at v_th = 1.0:
//
//   gain  max sensor current  proprioceptor spikes  effect of lesioning
//    2.0                0.45                     0             none
//    5.0                1.20                    23            0.105
//   10.0                1.99                   638            0.183
//   20.0                3.90                  3641            0.217
//
// The transition is exactly at threshold, and 2.0 was this crate's first
// default: the loop looked closed, the cord fired, the body moved, and the
// sensory channel was carrying nothing at all. 10.0 puts a few of the 304 leg
// proprioceptors over threshold per tick, which is the sparse regime wanted.

/// One control tick's worth of accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tick {
    pub control_tick: u64,
    /// Motor neurons that fired during this control tick.
    pub motor_spikes: u32,
    /// Neurons that fired anywhere in the cord.
    pub total_spikes: u32,
}

/// A connectome, a body, and the wiring between them.
pub struct Fly {
    net: SpikingNet,
    map: MotorMap,
    sensors: Vec<Sensor>,
    model: Model,
    data: Data,
    timing: Timing,
    coupling: Coupling,

    /// Indices of the descending neurons, the command input.
    descending: Vec<u32>,
    /// Per-actuator muscle activation, carried across ticks.
    activation: Vec<f32>,
    /// Per-actuator strength, 1.0 for an intact muscle. The body perturbation.
    muscle: Vec<f32>,
    /// Which generalized coordinate each leg actuator moves. Established by
    /// measurement at construction, not by reading another mjModel field.
    actuator_qpos: Vec<usize>,
    actuator_names: Vec<String>,

    proprioception: bool,
    last_proprio_spikes: u32,
    /// The standing descending command, one per descending neuron. Persists
    /// across ticks until a caller changes it.
    command: Vec<f32>,
    /// Per-neuron current for the next tick, rebuilt from `command` plus
    /// sensing every step so a lesioned channel leaves nothing behind.
    drive: Vec<f32>,
    control_tick: u64,
}

impl Fly {
    /// Compose a connectome and a body.
    ///
    /// `weight_scale` converts raw synapse counts to membrane current; see
    /// [`connectome::Connectome::signed_csc`] for why it is the caller's dial
    /// and not something this crate can pick. `shuffle_seed` replaces the real
    /// wiring with a degree-matched shuffle of it, which is the structural
    /// control: if a creature does as well on that, the connectome was not
    /// what mattered.
    pub fn new(
        gpu: Gpu,
        c: &Connectome,
        model: Model,
        lif: LifParams,
        // `weight_scale`: raw synapse count to membrane current. See
        // `Connectome::signed_csc`; nothing here fits it.
        weight_scale: f32,
        // `shuffle_seed`: `Some` runs the structural control on a
        // degree-matched shuffle of this connectome instead of the real one.
        shuffle_seed: Option<u64>,
        timing: Timing,
        coupling: Coupling,
    ) -> Result<Fly, String> {
        let mut data = Data::new(&model)?;
        let actuator_names: Vec<String> =
            model.actuator_names().into_iter().map(|n| n.unwrap_or_default()).collect();
        let map = flybody::build(c, &actuator_names);
        if map.mapped() == 0 {
            return Err("no motor neuron attached to any actuator; the body cannot be driven".to_string());
        }
        let sensors = sense::proprioceptors(c);
        let descending = c.population(|n| n.super_class == "descending");
        let actuator_qpos = probe_actuator_joints(&model, &mut data);

        // Signed and scaled: `Connectome::csc` carries raw synapse counts,
        // which are unsigned, and a network in which every synapse excites has
        // no inhibition and saturates on the first tick.
        let mut graph = c.signed_csc(weight_scale);
        if let Some(seed) = shuffle_seed {
            // The structural control: same in-degrees, same weights, sources
            // randomly reassigned. Applied AFTER signing so the sign
            // distribution is identical too - shuffling first would also
            // shuffle which neurons are inhibitory and confound two variables.
            graph = graph.shuffled_sources(seed);
        }
        let net = SpikingNet::new(gpu, &graph, lif)?;
        let n = c.neurons.len();
        let n_desc = descending.len();
        Ok(Fly {
            net,
            map,
            sensors,
            model,
            data,
            timing,
            coupling,
            descending,
            activation: vec![0.0; actuator_names.len()],
            muscle: vec![1.0; actuator_names.len()],
            actuator_qpos,
            actuator_names,
            proprioception: true,
            last_proprio_spikes: 0,
            command: vec![0.0; n_desc],
            drive: vec![0.0; n],
            control_tick: 0,
        })
    }

    /// Weaken or restore one muscle, `1.0` being intact and `0.0` severed.
    ///
    /// This is the BODY perturbation the control matrix asks for, and it is
    /// deliberately peripheral: nothing about the nervous system changes, the
    /// same spikes arrive at the same actuator, and less force comes out. That
    /// is what a weakened or damaged muscle is, and it is the manipulation
    /// insect locomotion work actually performs. Changing a coupling constant
    /// or a motor-map polarity instead would perturb the CONTROLLER and then
    /// call the recovery re-adaptation, which would be a different claim
    /// wearing this one's name.
    pub fn set_muscle_strength(&mut self, actuator: usize, strength: f32) -> Result<(), String> {
        let n = self.muscle.len();
        if actuator >= n {
            return Err(format!("actuator {actuator} is out of range; this body has {n}"));
        }
        self.muscle[actuator] = strength;
        Ok(())
    }

    /// Every muscle's current strength.
    pub fn muscle_strengths(&self) -> &[f32] {
        &self.muscle
    }

    /// Weaken every actuator whose name contains `pattern`, returning how many
    /// were affected. Zero is an error rather than a silent no-op: a lesion
    /// that hit nothing looks exactly like one the animal recovered from.
    pub fn lesion_matching(&mut self, pattern: &str, strength: f32) -> Result<usize, String> {
        let hit: Vec<usize> = self
            .actuator_names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.contains(pattern))
            .map(|(i, _)| i)
            .collect();
        if hit.is_empty() {
            return Err(format!("no actuator name contains {pattern:?}"));
        }
        for i in &hit {
            self.muscle[*i] = strength;
        }
        Ok(hit.len())
    }

    /// Turn the proprioceptive channel on or off.
    ///
    /// This is a lesion, and it is here rather than in a test because "does
    /// the feedback do anything" is a question the loop has to be able to
    /// answer about itself. A loop whose behaviour is identical with sensing
    /// disabled is an open loop wearing a closed loop's clothes.
    pub fn set_proprioception(&mut self, on: bool) {
        self.proprioception = on;
    }

    pub fn proprioception(&self) -> bool {
        self.proprioception
    }

    /// Command input: one value per descending neuron.
    pub fn set_descending(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() != self.descending.len() {
            return Err(format!("expected {} descending values, got {}", self.descending.len(), values.len()));
        }
        self.command.copy_from_slice(values);
        Ok(())
    }

    pub fn descending_count(&self) -> usize {
        self.descending.len()
    }
    pub fn proprioceptor_count(&self) -> usize {
        self.sensors.len()
    }
    pub fn motor_map(&self) -> &MotorMap {
        &self.map
    }
    /// The body, for a renderer.
    ///
    /// Handed out as a pair rather than separately because MuJoCo's rendering
    /// calls take both and they must describe the same simulation; two
    /// accessors would let a caller pair a model with another fly's data,
    /// which reads out as a body frozen in its initial pose.
    pub fn body(&self) -> (&Model, &Data) {
        (&self.model, &self.data)
    }

    pub fn actuator_names(&self) -> &[String] {
        &self.actuator_names
    }

    /// Generalized coordinates of the body.
    pub fn qpos(&self) -> Vec<f64> {
        self.data.get(&self.model, StateSpec::QPOS)
    }

    /// Place the body at a given pose and velocity.
    ///
    /// Reference-state initialisation, which is standard for imitation: a
    /// creature that always starts from rest has to learn to reach the
    /// reference's starting posture before it can track anything, and that
    /// detour is not what is being measured. Starting on the trajectory means
    /// the reward is about following it.
    pub fn set_pose(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<(), String> {
        self.data.set(&self.model, StateSpec::QPOS, qpos)?;
        self.data.set(&self.model, StateSpec::QVEL, qvel)?;
        self.data.forward(&self.model);
        Ok(())
    }

    /// Generalized velocities of the body.
    pub fn qvel(&self) -> Vec<f64> {
        self.data.get(&self.model, StateSpec::QVEL)
    }

    /// `(nq, nv)` of the body, for checking a reference describes it.
    pub fn dims(&self) -> (usize, usize) {
        (self.model.nq(), self.model.nv())
    }

    /// Start a new episode: clear the dynamical state and put the body back,
    /// KEEPING whatever the weights have learned.
    ///
    /// Not `DynamicalSystem::reset`, which also restores the connectome's
    /// original weights. An episode loop built on that unlearns between every
    /// episode and reports a perfectly reproducible failure to learn.
    /// Put the body back where it started and silence the cord, leaving
    /// anything learned - and any muscle lesion - in place.
    ///
    /// Muscle strength deliberately survives a reset. A body perturbation that
    /// undid itself at every episode boundary could never be re-adapted to,
    /// and the control that asks for re-adaptation would be measuring nothing.
    pub fn reset(&mut self) {
        self.net.reset_state();
        self.data.reset(&self.model);
        for a in self.activation.iter_mut() {
            *a = 0.0;
        }
        for c in self.command.iter_mut() {
            *c = 0.0;
        }
        self.last_proprio_spikes = 0;
        self.control_tick = 0;
    }

    /// Turn weight updates on or off. See `neuro::Plastic`.
    pub fn set_plasticity(&mut self, on: bool) {
        self.net.set_plasticity(on);
    }

    /// Deliver this tick's neuromodulator.
    pub fn modulate(&mut self, delta: f32) {
        self.net.modulate(delta);
    }

    /// Enable three-factor plasticity on the cord.
    pub fn enable_plasticity(&mut self, p: neuro::PlasticityParams) -> Result<(), String> {
        self.net.enable_plasticity(p)
    }

    /// The cord's current synaptic weights.
    pub fn weights(&self) -> Vec<f32> {
        self.net.weights()
    }

    /// Overwrite the cord's synaptic weights. See
    /// [`neuro::SpikingNet::set_weights`].
    pub fn set_weights(&mut self, w: &[f32]) -> Result<(), String> {
        self.net.set_weights(w)
    }

    /// The largest absolute weight the connectome started with.
    ///
    /// The number a plasticity clamp has to be sized against. Weights here are
    /// scaled synapse counts, so they run to tens; a clamp picked without
    /// looking at them squashes the entire connectome on the first update.
    pub fn initial_weight_scale(&self) -> f32 {
        self.net.initial_weights().iter().fold(0.0f32, |m, w| m.max(w.abs()))
    }

    /// Range of the sensory current injected on the last tick, for tracing a
    /// loop that is not doing what it should.
    pub fn sensor_current_range(&self) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for s in &self.sensors {
            let v = self.drive[s.neuron as usize];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if self.sensors.is_empty() { (0.0, 0.0) } else { (lo, hi) }
    }

    /// Proprioceptors that fired on the last tick.
    ///
    /// The number that distinguishes a sensory channel which is CONNECTED from
    /// one which is TRANSMITTING. A proprioceptor held below threshold injects
    /// current every tick, spikes never, and changes nothing downstream - and
    /// every other measurement in this loop looks healthy while it happens.
    pub fn proprioceptor_spikes(&self) -> u32 {
        self.last_proprio_spikes
    }

    /// Largest absolute muscle activation, same purpose.
    pub fn activation_range(&self) -> f32 {
        self.activation.iter().fold(0.0f32, |m, a| m.max(a.abs()))
    }

    /// One control tick: sense, think, act, integrate.
    pub fn step(&mut self) -> Result<Tick, String> {
        let qpos = self.data.get(&self.model, StateSpec::QPOS);
        let qvel = self.data.get(&self.model, StateSpec::QVEL);

        // --- sense -------------------------------------------------------
        // Rebuilt every tick from the descending command, so a stale sensory
        // current cannot persist after the channel is lesioned.
        for v in self.drive.iter_mut() {
            *v = 0.0;
        }
        for (slot, v) in self.descending.iter().zip(&self.command) {
            self.drive[*slot as usize] = *v;
        }
        if self.proprioception {
            for s in &self.sensors {
                let current = self.sensor_current(s, &qpos, &qvel);
                self.drive[s.neuron as usize] += current;
            }
        }

        // --- think -------------------------------------------------------
        self.net.drive(Port::Drive, &self.drive)?;
        let mut spike = vec![0.0f32; self.drive.len()];
        let mut motor_spikes = 0u32;
        let mut total_spikes = 0u32;
        // Activation decays once per CONTROL tick, not once per neural tick:
        // it is a muscle's time constant, not the nervous system's.
        for a in self.activation.iter_mut() {
            *a *= self.coupling.activation_decay;
        }
        for _ in 0..self.timing.neural_per_control {
            self.net.step();
            self.net.read(Port::Spike, &mut spike)?;
            total_spikes += spike.iter().filter(|&&s| s > 0.5).count() as u32;
            self.last_proprio_spikes =
                self.sensors.iter().filter(|s| spike[s.neuron as usize] > 0.5).count() as u32;
            for d in &self.map.drives {
                if spike[d.neuron as usize] > 0.5 {
                    self.activation[d.actuator] += self.coupling.activation_gain * d.polarity;
                    motor_spikes += 1;
                }
            }
        }

        // --- act ---------------------------------------------------------
        let ctrl: Vec<f64> = self
            .activation
            .iter()
            .zip(&self.muscle)
            .map(|(a, m)| (a * m).clamp(-1.0, 1.0) as f64)
            .collect();
        self.data.set(&self.model, StateSpec::CTRL, &ctrl)?;

        // --- integrate ---------------------------------------------------
        for _ in 0..self.timing.physics_per_control {
            self.data.step(&self.model);
        }

        self.control_tick += 1;
        Ok(Tick { control_tick: self.control_tick, motor_spikes, total_spikes })
    }

    fn sensor_current(&self, s: &Sensor, qpos: &[f64], qvel: &[f64]) -> f32 {
        match s.modality {
            Modality::Angle(dof) => {
                let want = dof.actuator(s.segment, s.side);
                match self.actuator_names.iter().position(|n| *n == want) {
                    Some(i) => {
                        let q = self.actuator_qpos[i];
                        qpos.get(q).copied().unwrap_or(0.0) as f32 * self.coupling.angle_gain
                    }
                    None => 0.0,
                }
            }
            Modality::LoadProxy => {
                // Mean joint speed of this leg. See `Modality::LoadProxy` for
                // why this is a proxy and not the quantity itself.
                let suffix = format!("_{}_{}", seg_str(s.segment), side_str(s.side));
                let leg: Vec<usize> = self
                    .actuator_names
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.ends_with(&suffix))
                    .map(|(i, _)| i)
                    .collect();
                if leg.is_empty() {
                    return 0.0;
                }
                let sum: f64 = leg
                    .iter()
                    .filter_map(|&i| self.actuator_qpos.get(i))
                    .filter_map(|&q| qvel.get(q))
                    .map(|v| v.abs())
                    .sum();
                (sum / leg.len() as f64) as f32 * self.coupling.load_gain
            }
        }
    }
}

fn seg_str(s: flybody::Segment) -> &'static str {
    match s {
        flybody::Segment::T1 => "T1",
        flybody::Segment::T2 => "T2",
        flybody::Segment::T3 => "T3",
    }
}

fn side_str(s: flybody::Side) -> &'static str {
    match s {
        flybody::Side::Left => "left",
        flybody::Side::Right => "right",
    }
}

/// Which generalized coordinate each actuator moves, found by driving each one
/// alone and taking the argmax displacement.
///
/// Measured rather than read out of `actuator_trnid`/`jnt_qposadr`, which
/// would mean mirroring two more mjModel arrays whose offsets this binding
/// deliberately does not depend on. For a single actuator driven in isolation
/// from rest, the coordinate that moves IS its joint.
fn probe_actuator_joints(model: &Model, data: &mut Data) -> Vec<usize> {
    let nu = model.nu();
    let mut out = vec![0usize; nu];
    for (i, slot) in out.iter_mut().enumerate() {
        data.reset(model);
        data.forward(model);
        let base = data.get(model, StateSpec::QPOS);
        let mut ctrl = vec![0.0f64; nu];
        ctrl[i] = 1.0;
        if data.set(model, StateSpec::CTRL, &ctrl).is_err() {
            continue;
        }
        for _ in 0..25 {
            data.step(model);
        }
        let now = data.get(model, StateSpec::QPOS);
        *slot = base
            .iter()
            .zip(&now)
            .enumerate()
            .max_by(|a, b| (a.1 .1 - a.1 .0).abs().partial_cmp(&(b.1 .1 - b.1 .0).abs()).unwrap())
            .map(|(k, _)| k)
            .unwrap_or(0);
    }
    data.reset(model);
    out
}
