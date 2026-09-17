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

pub mod conditioning;
pub mod gait;
pub mod learn;
pub mod reference;
pub mod cns;
pub mod physiology;
pub mod rhythm;
pub mod search;
pub mod sense;
pub mod tuning;
pub mod wing;

use connectome::Connectome;
use flybody::MotorMap;
use gpu_core::Gpu;
use mujoco::{Data, Model, StateSpec};
use neuro::{DynamicalSystem, LifParams, Plastic, Port, SpikingNet};

pub use gait::{analyse as analyse_gait, Gait, Trace};
pub use wing::{WingCommand, Wingbeat};
pub use reference::{ImitationReward, Reference};
pub use cns::Cns;
pub use tuning::Tuning;
pub use sense::{Antennae, Modality, Sensor};

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
    /// The model's own physics timestep, in seconds. Only the wingbeat needs
    /// it, and it needs it exactly: the stroke's phase advances in real time,
    /// so a value that disagrees with the MJCF produces a wingbeat at the
    /// wrong frequency while every other reading stays correct.
    pub physics_dt: f64,
}

impl Default for Timing {
    fn default() -> Self {
        Timing { neural_per_control: 1, physics_per_control: 20, physics_dt: 1e-4 }
    }
}

/// A control tick is 2 ms of body time: 500 Hz, which is the rate flybody's
/// own walking tasks control at.
pub const CONTROL_PERIOD: f64 = 2e-3;

impl Timing {
    /// How many integrator steps of `dt` fill one control period.
    ///
    /// Derived rather than written down beside the timestep, because the two
    /// disagreeing is not a crash: it is a body that runs faster or slower
    /// than the nervous system driving it, and everything downstream - the
    /// gait frequency, the real-time ratio, the reward per episode - reads as
    /// a behaviour change instead of as a misconfiguration.
    pub fn substeps(dt: f64) -> Result<u32, String> {
        if !dt.is_finite() || dt <= 0.0 || dt > CONTROL_PERIOD {
            return Err(format!(
                "a timestep of {dt} s cannot fill a {CONTROL_PERIOD} s control period; it must be positive and no longer than one"
            ));
        }
        let n = (CONTROL_PERIOD / dt).round();
        // A step that does not divide the control period leaves the body
        // ahead of or behind the cord by a fraction of a tick, every tick.
        if (n * dt - CONTROL_PERIOD).abs() > 1e-12 {
            return Err(format!(
                "a timestep of {dt} s does not divide the {CONTROL_PERIOD} s control period ({n} steps would be {} s)",
                n * dt
            ));
        }
        Ok(n as u32)
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

/// The membrane and synapse parameters a nerve cord is run at here.
///
/// A function rather than a constant because it is a MEASURED operating point,
/// not a convention, and the measurement should live next to the number. On the
/// front-leg neuropil driven through one descending cell type, sweeping the
/// synaptic time constants over a decade each:
///
/// | inhibitory tau | rhythmicity | cord active | motor spikes/tick |
/// |---|---|---|---|
/// | 5 ms | 0.04 to 0.30 | 13 to 15% | 13 to 18 |
/// | **20 ms** | **0.14 to 0.46** | **3 to 6%** | **7 to 19** |
/// | 40 ms | 0.11 to 0.40 | 2 to 5% | 1 to 6 |
///
/// Twenty milliseconds is a peak rather than an end of a range. Below it the
/// cord fires at a seizure's rate and the rhythm is buried; above it the rhythm
/// survives but the inhibition swallows the motor output, and a cord that
/// oscillates beautifully while sending nothing to the muscles is not a
/// controller. It is also the asymmetry a reciprocal-inhibition oscillator
/// needs and roughly what the animal has: fast cholinergic excitation against
/// slower GABAergic inhibition.
///
/// The membrane constant is 10 ms and the refractory period one control tick.
pub fn cord_lif() -> LifParams {
    let dt_ms = 2.0;
    LifParams {
        dt_over_tau: dt_ms / 10.0,
        v_th: 1.0,
        r: 1.0,
        refrac_ticks: 1,
        dt_over_tau_syn: dt_ms / 5.0,
        dt_over_tau_inh: dt_ms / 20.0,
        ..LifParams::default()
    }
}

/// How the connectome becomes a network.
///
/// A struct rather than three more positional arguments, because two of these
/// are controls and one is a modelling decision, and a call site that reads
/// `..., 3e-2, None, Some(10.0), ...` tells a reader none of that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wiring {
    /// Raw synapse count to membrane current. Nothing in the data fixes it.
    pub weight_scale: f32,
    /// `Some` runs the structural control on a degree-matched shuffle of the
    /// connectome instead of the real one.
    pub shuffle_seed: Option<u64>,
    /// Scale each neuron's input by its own reconstructed membrane area,
    /// clamped to this factor either way. See
    /// [`connectome::Connectome::excitability`] - a uniform threshold makes the
    /// largest cells in the cord hundreds of times more excitable than the
    /// smallest, and that is the difference between a network that oscillates
    /// and one that saturates.
    ///
    /// `None` keeps the uniform model, which is the CONTROL for whether the
    /// normalisation is doing anything.
    pub size_limit: Option<f32>,
    /// Drop every pair connected by fewer than this many synapses. See
    /// [`connectome::Connectome::network`]. `1` keeps everything.
    pub min_synapses: u32,
}

impl Default for Wiring {
    fn default() -> Self {
        // 0.6: swept against the gait criterion with the command delivered to
        // ONE descending cell type. This crate's earlier default of 0.03 was
        // swept too, but under a simultaneous barrage of all 1,328 descending
        // neurons - and under a single-cell-type command it leaves the cord at
        // 0.007% active, which is to say the command dies before it arrives.
        //
        // 5: a reconstruction assigns a great many one- and two-synapse pairs
        // at the edge of what the imaging resolves, numerous enough to dominate
        // a neuron's input count while carrying almost none of its drive.
        //
        // 10.0: the reconstruction also leaves a long tail of fragments and
        // giant cells, and an unclamped size factor turns one badly
        // reconstructed neuron into a silent one or a runaway one.
        Wiring { weight_scale: 0.6, shuffle_seed: None, size_limit: Some(10.0), min_synapses: 5 }
    }
}

/// How body state becomes current, and spikes become torque.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coupling {
    /// Current injected per radian of joint deflection.
    pub angle_gain: f32,
    /// Current injected into each olfactory receptor neuron per unit of odour
    /// concentration at its antenna.
    ///
    /// A receptor neuron has almost no incoming synapses - it is an afferent,
    /// so injected current is the ONLY thing that can fire it, exactly as the
    /// proprioceptors are. That is the failure this crate has already had
    /// once: a sensory channel that was connected, carried current every tick,
    /// never reached threshold, and changed nothing when it was lesioned while
    /// every other reading stayed healthy. `Fly::antenna_spikes` exists so a
    /// caller can assert the channel CARRIES something before asserting that
    /// it matters.
    pub odour_gain: f32,
    /// Current injected per radian/second, for the load proxy.
    pub load_gain: f32,
    /// Muscle activation decay per control tick. A spike is an impulse; a
    /// muscle is not, so motor output is low-pass filtered rather than fed
    /// through raw.
    pub activation_decay: f32,
    /// Activation added per polarity-weighted motor spike.
    pub activation_gain: f32,
    /// Wingbeat power added per power-motor-neuron spike, before the same
    /// decay the leg activation uses. The power muscles are asynchronous, so
    /// this is a RATE-to-amplitude conversion and not a per-spike impulse on a
    /// joint - see [`wing`].
    pub wing_power_gain: f32,
    /// Stroke-amplitude bias added per steering-motor-neuron spike.
    pub wing_steer_gain: f32,
}

impl Default for Coupling {
    fn default() -> Self {
        Coupling {
            angle_gain: 10.0,
            // The same order as `angle_gain`, and for the same reason: a
            // receptor's threshold is the cord's, and the concentrations a
            // caller supplies are normalised to 1.0 at the source.
            odour_gain: 10.0,
            load_gain: 0.05,
            activation_decay: 0.8,
            activation_gain: 0.05,
            // 24 power motor neurons firing at up to one spike per control
            // tick reach full power in a few ticks at this gain, which is the
            // right order for a thorax that spins up over a handful of
            // wingbeats rather than instantly or over a second.
            wing_power_gain: 0.05,
            wing_steer_gain: 0.02,
        }
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
    /// Wall time inside the nervous system: the drive upload, the neural
    /// ticks, and the spike readback each one waits for.
    ///
    /// Reported per tick rather than left to a profiling build because this
    /// loop has a HARD deadline - 2 ms at 500 Hz - and which half is over it
    /// is not guessable from the outside. The two are also fixed by different
    /// things (the cord by the device the network is on, the body by MuJoCo's
    /// own solver), so a single "slow" number sends the reader to the wrong
    /// one as often as not. Two `Instant::now()` pairs per control tick is
    /// about 40 ns against a 2 ms budget.
    pub cord: std::time::Duration,
    /// Wall time inside MuJoCo: `physics_per_control` steps of the body.
    pub body: std::time::Duration,
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
    /// Each descending neuron's published cell type, so a command can name one
    /// rather than address the whole population.
    descending_types: Vec<String>,
    /// Per-neuron input scaling, `1.0` everywhere when size normalisation is
    /// off. Applied to INJECTED current as well as to synaptic weights, or the
    /// two input paths would end up on different scales.
    excite: Vec<f32>,
    /// Per-actuator muscle activation, carried across ticks.
    activation: Vec<f32>,
    /// Motor spikes on each actuator THIS tick, split by polarity:
    /// `[agonist, antagonist]`.
    ///
    /// Kept separately from `activation` because the net signal cannot tell
    /// two silent muscles from two loud ones pulling against each other, and
    /// those are different failures. A cord that produces a clean rhythm on
    /// both sides of a joint at the same phase drives that joint nowhere, and
    /// every summary that looks at the net output reports it as no rhythm.
    opposed: Vec<[f32; 2]>,
    /// Per-actuator strength, 1.0 for an intact muscle. The body perturbation.
    muscle: Vec<f32>,
    /// `Some` once flight is enabled: the wingbeat this body is flying on.
    wingbeat: Option<Wingbeat>,
    /// Wing motor neurons and what each one does.
    wing_drives: Vec<flybody::WingDrive>,
    /// `[wing][dof]` actuator indices, left then right, yaw/roll/pitch.
    wing_actuators: [[Option<usize>; 3]; 2],
    /// Where each of those six joints sits in `qpos`.
    ///
    /// Taken from the joint's NAME rather than from the actuator probe the
    /// legs use. The probe drives one actuator and records which coordinate
    /// moved most, and the three wing degrees of freedom are mechanically
    /// coupled hard enough that driving the stroke moves the feathering more -
    /// so the probe reports the wrong joint, and a stroke amplitude read that
    /// way exceeds the stroke joint's own range without anything complaining.
    wing_qpos: [[Option<usize>; 3]; 2],
    /// The wingbeat command the cord is currently producing, low-passed the
    /// same way leg activation is.
    wing_cmd: WingCommand,
    /// When set, the cord's wing output is IGNORED and this is flown instead.
    wing_hold: Option<WingCommand>,
    /// The `coxa` actuator of each leg, in `flybody::LEGS` order: the
    /// fore-aft swing, which is the signal a stepping rhythm shows up in.
    leg_coxa: [Option<usize>; 6],
    actuator_names: Vec<String>,
    /// Which generalized coordinates each sensor reads, resolved once against
    /// the body rather than looked up by name every tick.
    sensor_target: Vec<SensorTarget>,

    /// The olfactory receptor neurons, by side. Empty on a cord-only
    /// nervous system: a nerve cord has no nose.
    antennae: Antennae,
    /// This tick's odour concentration at each antenna, `[left, right]`.
    /// Persists until a caller changes it, the same way the descending
    /// command does.
    odour: [f32; 2],
    /// Where the food is, if the world has any. When set, the odour is
    /// recomputed from the body's own pose every tick rather than being
    /// supplied from outside - which is what lets an EPISODE smell, without
    /// the loop that runs it having to know what a plume is.
    food: Option<[f64; 3]>,
    /// Receptor neurons that fired on the last neural tick, `(left, right)`.
    last_antenna_spikes: (u32, u32),
    proprioception: bool,
    last_proprio_spikes: u32,
    /// The standing descending command, one per descending neuron. Persists
    /// across ticks until a caller changes it.
    command: Vec<f32>,
    /// Per-neuron current for the next tick, rebuilt from `command` plus
    /// sensing every step so a lesioned channel leaves nothing behind.
    drive: Vec<f32>,
    /// This tick's spikes, read back from the device. A field rather than a
    /// local because `step` runs on a 2 ms deadline and this is one allocation
    /// of a neuron-sized buffer per neural tick - 92 KB on the fly's cord.
    spike: Vec<f32>,
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
        wiring: Wiring,
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
        // Which generalized coordinate each leg actuator moves, established by
        // measurement rather than by reading another mjModel field. Consumed
        // here and not kept: the only thing that ever wanted it was the
        // sensors, and they now carry their own answer.
        let actuator_qpos = probe_actuator_joints(&model, &mut data);
        let sensor_target = resolve_sensors(&sensors, &actuator_names, &actuator_qpos);
        let mut wing_qpos: [[Option<usize>; 3]; 2] = [[None; 3]; 2];
        for (w, side) in [flybody::Side::Left, flybody::Side::Right].into_iter().enumerate() {
            for (d, dof) in [flybody::WingDof::Yaw, flybody::WingDof::Roll, flybody::WingDof::Pitch]
                .into_iter()
                .enumerate()
            {
                wing_qpos[w][d] = model.joint_qpos(&dof.actuator(side));
            }
        }
        let mut leg_coxa: [Option<usize>; 6] = [None; 6];
        for (i, (seg, side, _)) in flybody::LEGS.iter().enumerate() {
            let want = flybody::LegDof::Coxa.actuator(*seg, *side);
            leg_coxa[i] = actuator_names.iter().position(|n| *n == want);
        }

        // Signed and scaled: `Connectome::csc` carries raw synapse counts,
        // which are unsigned, and a network in which every synapse excites has
        // no inhibition and saturates on the first tick.
        // Excitability first, shuffle second. A shuffle reassigns SOURCES and
        // leaves each neuron's own column intact, so the two commute - but
        // stating the order matters, because a normalisation applied after a
        // shuffle would be normalising by the shuffled graph's degrees rather
        // than by the real cell's size.
        let excite = match wiring.size_limit {
            Some(limit) => c.excitability(limit),
            None => vec![1.0; c.neurons.len()],
        };
        let mut graph = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
        if let Some(seed) = wiring.shuffle_seed {
            // The structural control: same in-degrees, same weights, sources
            // randomly reassigned. Applied AFTER signing so the sign
            // distribution is identical too - shuffling first would also
            // shuffle which neurons are inhibitory and confound two variables.
            graph = graph.shuffled_sources(seed);
        }
        let net = SpikingNet::new(gpu, &graph, lif)?;
        let n = c.neurons.len();
        let n_desc = descending.len();
        let descending_types: Vec<String> =
            descending.iter().map(|i| c.neurons[*i as usize].cell_type.clone()).collect();
        Ok(Fly {
            net,
            map,
            sensors,
            model,
            data,
            timing,
            coupling,
            descending,
            descending_types,
            excite,
            activation: vec![0.0; actuator_names.len()],
            opposed: vec![[0.0; 2]; actuator_names.len()],
            muscle: vec![1.0; actuator_names.len()],
            wingbeat: None,
            wing_drives: flybody::wing_map(c),
            wing_actuators: {
                let mut a = [[None; 3]; 2];
                for (w, side) in [flybody::Side::Left, flybody::Side::Right].into_iter().enumerate() {
                    for (d, dof) in [flybody::WingDof::Yaw, flybody::WingDof::Roll, flybody::WingDof::Pitch]
                        .into_iter()
                        .enumerate()
                    {
                        let want = dof.actuator(side);
                        a[w][d] = actuator_names.iter().position(|n| *n == want);
                    }
                }
                a
            },
            wing_qpos,
            wing_cmd: WingCommand::default(),
            wing_hold: None,
            leg_coxa,
            sensor_target,
            actuator_names,
            antennae: Antennae::of(c),
            odour: [0.0; 2],
            food: None,
            last_antenna_spikes: (0, 0),
            proprioception: true,
            last_proprio_spikes: 0,
            command: vec![0.0; n_desc],
            drive: vec![0.0; n],
            spike: vec![0.0; n],
            control_tick: 0,
        })
    }

    /// Start the wingbeat.
    ///
    /// Flight is opt-in because it changes what a control tick costs: the
    /// stroke has to be written inside the physics loop, so every physics step
    /// now carries a control write. A walking run should not pay for that.
    ///
    /// Fails when the body has no wing actuators or the connectome has no wing
    /// motor neurons, rather than flying a body nothing is attached to.
    pub fn enable_flight(&mut self, beat: Wingbeat) -> Result<(), String> {
        if self.wing_actuators.iter().flatten().any(|a| a.is_none()) {
            return Err("this body does not expose all six wing actuators".to_string());
        }
        if self.wing_drives.is_empty() {
            return Err("this connectome has no wing motor neurons to fly with".to_string());
        }
        self.wingbeat = Some(beat);
        Ok(())
    }

    pub fn wingbeat(&self) -> Option<Wingbeat> {
        self.wingbeat
    }

    /// Set the wingbeat frequency without disturbing its phase.
    pub fn set_wingbeat_hz(&mut self, hz: f64) {
        if let Some(b) = &mut self.wingbeat {
            b.hz = hz;
        }
    }

    /// Fly an imposed wing command, ignoring what the cord produces.
    ///
    /// This is a calibration control, not a shortcut: it is how the AIRFRAME
    /// is characterised - what stroke amplitude a given drive reaches, at what
    /// frequency it resonates, whether it lifts at all - separately from
    /// whether a connectome can command it. Measuring the two together makes
    /// a fly that does not take off uninterpretable.
    pub fn hold_wing_command(&mut self, cmd: Option<WingCommand>) {
        self.wing_hold = cmd;
    }

    /// Each wing's `[yaw, roll, pitch]` joint angle, left then right.
    pub fn wing_angles(&self) -> [[f64; 3]; 2] {
        let qpos = self.qpos();
        let mut out = [[0.0; 3]; 2];
        for (wing, slots) in out.iter_mut().zip(self.wing_qpos) {
            for (angle, slot) in wing.iter_mut().zip(slots) {
                if let Some(i) = slot {
                    *angle = qpos.get(i).copied().unwrap_or(0.0);
                }
            }
        }
        out
    }

    /// What the cord is currently telling the wings to do.
    pub fn wing_command(&self) -> WingCommand {
        self.wing_cmd
    }

    /// How many wing motor neurons attached, by role.
    pub fn wing_summary(&self) -> String {
        let (mut power, mut amp, mut aoa) = (0, 0, 0);
        for d in &self.wing_drives {
            match d.action {
                flybody::WingAction::Power => power += 1,
                flybody::WingAction::Amplitude(_) => amp += 1,
                flybody::WingAction::AngleOfAttack(_) => aoa += 1,
            }
        }
        format!("{power} power, {amp} amplitude, {aoa} angle-of-attack wing motor neurons")
    }

    /// Each leg's fore-aft swing command, in `flybody::LEGS` order.
    ///
    /// The coxa's activation rather than a sum over the leg's actuators,
    /// because a stepping rhythm is a matter of PHASE and a sum of magnitudes
    /// discards it: a leg swinging forward and one swinging back would look
    /// identical. This is the same signal a scripted tripod gait drives.
    pub fn leg_swing(&self) -> [f32; 6] {
        let mut out = [0.0f32; 6];
        for (o, slot) in out.iter_mut().zip(self.leg_coxa) {
            if let Some(i) = slot {
                *o = self.activation[i] * self.muscle[i];
            }
        }
        out
    }

    /// Each leg's coxa drive this tick as `[agonist, antagonist]` spike
    /// counts, in `flybody::LEGS` order.
    ///
    /// The diagnostic [`Self::leg_swing`] cannot give: a joint whose two
    /// muscles both fire rhythmically IN PHASE produces no movement, and the
    /// net activation that drives the body reports that as no rhythm at all.
    /// Looking at one side alone says whether the rhythm is absent or merely
    /// cancelled, and those need different fixes.
    pub fn leg_opposed(&self) -> [[f32; 2]; 6] {
        let mut out = [[0.0f32; 2]; 6];
        for (o, slot) in out.iter_mut().zip(self.leg_coxa) {
            if let Some(i) = slot {
                *o = self.opposed[i];
            }
        }
        out
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
    /// How many actuators the body has, so a caller can lesion all of them
    /// without knowing the model.
    pub fn actuator_count(&self) -> usize {
        self.muscle.len()
    }

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

    /// Drive only the descending neurons whose published cell type is `name`,
    /// silencing every other one.
    ///
    /// Addressing the whole descending population at once, which is the
    /// obvious thing to do, is a category error: a fly has over a thousand
    /// descending neurons and they command DIFFERENT behaviours, several of
    /// which oppose each other. Driving them all together is not "go", it is
    /// every command at once, and what reaches the muscles is whatever
    /// survives the collision.
    ///
    /// Returns how many neurons were driven. Zero is an error rather than a
    /// silent no-op, because a command that reached nobody looks exactly like
    /// one the cord ignored.
    pub fn drive_cell_type(&mut self, name: &str, current: f32) -> Result<usize, String> {
        let mut hit = 0;
        for (slot, ty) in self.descending_types.iter().enumerate() {
            if ty == name {
                self.command[slot] = current;
                hit += 1;
            } else {
                self.command[slot] = 0.0;
            }
        }
        if hit == 0 {
            return Err(format!("no descending neuron has cell type {name:?}"));
        }
        Ok(hit)
    }

    /// Every distinct descending cell type, with how many neurons carry it.
    pub fn descending_types(&self) -> Vec<(String, usize)> {
        let mut v: Vec<String> = self.descending_types.clone();
        v.sort();
        let mut out: Vec<(String, usize)> = Vec::new();
        for t in v {
            match out.last_mut() {
                Some((last, n)) if *last == t => *n += 1,
                _ => out.push((t, 1)),
            }
        }
        out
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
    /// Put the animal back where it started.
    ///
    /// EVERY piece of dynamical state, and the list is exhaustive on purpose.
    /// This used to reset the network, the body, the muscle activation and the
    /// descending command, and to leave the wingbeat's PHASE, the cord's wing
    /// command and the per-joint opposition counts running - so two identical
    /// episodes back to back were not identical. Measured before the fix, the
    /// same 400-tick flight episode run four times: 53, 30, 36 and 30 ticks
    /// airborne. An experiment comparing two parameter sets on numbers like
    /// those is comparing its own leftovers, and the search built on top of it
    /// would have been fitting them.
    ///
    /// What is NOT reset is CONFIGURATION: the weights, the wing hold, the
    /// muscle strengths, the plasticity switch. A caller set those and a reset
    /// that silently undid them would be the opposite bug - which this crate
    /// has also had, when `reset` restored the connectome's original weights
    /// and every episode in a learning run unlearned.
    pub fn reset(&mut self) {
        self.net.reset_state();
        self.data.reset(&self.model);
        for a in self.activation.iter_mut() {
            *a = 0.0;
        }
        for o in self.opposed.iter_mut() {
            *o = [0.0; 2];
        }
        for c in self.command.iter_mut() {
            *c = 0.0;
        }
        for d in self.drive.iter_mut() {
            *d = 0.0;
        }
        for s in self.spike.iter_mut() {
            *s = 0.0;
        }
        self.odour = [0.0; 2];
        self.wing_cmd = WingCommand::default();
        if let Some(beat) = self.wingbeat.as_mut() {
            beat.phase = 0.0;
        }
        self.last_proprio_spikes = 0;
        self.last_antenna_spikes = (0, 0);
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

    /// Change the cord's membrane and synapse parameters in place.
    ///
    /// A search over dynamics is a search over THESE - the synaptic time
    /// constants and the adaptation current are what decide whether a network
    /// of these cells can oscillate - and they are uniform across the
    /// population, so changing them costs nothing. See
    /// `neuro::SpikingNet::set_params`.
    pub fn set_lif(&mut self, params: LifParams) -> Result<(), String> {
        self.net.set_params(params)
    }

    pub fn lif(&self) -> LifParams {
        self.net.params()
    }

    /// Change how body state becomes current and spikes become torque.
    pub fn set_coupling(&mut self, coupling: Coupling) {
        self.coupling = coupling;
    }

    pub fn coupling(&self) -> Coupling {
        self.coupling
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

    /// Set the odour concentration at each antenna, `[left, right]`.
    ///
    /// A stimulus, not a command. See [`Antennae`]: the difference between the
    /// two is a fraction of a percent at any useful distance and what to do
    /// about it is the brain's problem, not this function's.
    pub fn smell(&mut self, left: f32, right: f32) {
        self.food = None;
        self.odour = [left, right];
    }

    /// Put something in the world to smell, and let the animal's own pose
    /// decide what reaches each antenna.
    ///
    /// `None` removes it, which is the CONTROL: the same animal in the same
    /// world with nothing to smell. Without that control, "it approached the
    /// food" is equally satisfied by an animal that walks in one direction.
    pub fn set_food(&mut self, at: Option<[f64; 3]>) {
        self.food = at;
        self.odour = [0.0; 2];
    }

    pub fn food(&self) -> Option<[f64; 3]> {
        self.food
    }

    /// How far the food is, or `None` if there is none.
    pub fn food_range(&self) -> Option<f64> {
        self.food.map(|f| sense::yaw_and_range(&self.qpos(), f).1)
    }

    /// The whole cord's spikes from the last neural tick, as the device last
    /// reported them.
    ///
    /// Already in hand - `step` reads this vector back every tick to find the
    /// motor neurons - so an experiment measuring some other population costs
    /// a slice rather than a second readback.
    pub fn spikes(&self) -> &[f32] {
        &self.spike
    }

    /// How many receptor neurons there are, by side.
    pub fn antenna_counts(&self) -> (usize, usize) {
        (self.antennae.left.len(), self.antennae.right.len())
    }

    /// How many of them fired on the last neural tick, by side.
    ///
    /// The instrument that tells a connected-and-silent channel from a
    /// working one, which is a distinction this crate has already paid for
    /// once.
    pub fn antenna_spikes(&self) -> (u32, u32) {
        self.last_antenna_spikes
    }

    /// Largest absolute muscle activation, same purpose.
    pub fn activation_range(&self) -> f32 {
        self.activation.iter().fold(0.0f32, |m, a| m.max(a.abs()))
    }

    /// One control tick: sense, think, act, integrate.
    pub fn step(&mut self) -> Result<Tick, String> {
        let began = std::time::Instant::now();
        let qpos = self.data.get(&self.model, StateSpec::QPOS);
        let qvel = self.data.get(&self.model, StateSpec::QVEL);

        // --- sense -------------------------------------------------------
        // Rebuilt every tick from the descending command, so a stale sensory
        // current cannot persist after the channel is lesioned.
        for v in self.drive.iter_mut() {
            *v = 0.0;
        }
        for (slot, v) in self.descending.iter().zip(&self.command) {
            self.drive[*slot as usize] = *v * self.excite[*slot as usize];
        }
        if self.proprioception {
            for i in 0..self.sensors.len() {
                let current = self.sensor_current(i, &qpos, &qvel);
                let n = self.sensors[i].neuron as usize;
                self.drive[n] += current * self.excite[n];
            }
        }
        // Smell, if this animal has a nose. Rebuilt every tick like the rest
        // of the sensory drive, so an odour that stops arriving stops being
        // smelled rather than lingering as a standing current.
        if let Some(food) = self.food {
            let (yaw, _) = sense::yaw_and_range(&qpos, food);
            let p = [qpos.first().copied().unwrap_or(0.0), qpos.get(1).copied().unwrap_or(0.0), qpos.get(2).copied().unwrap_or(0.0)];
            let (l, r) = sense::plume(food, p, yaw);
            self.odour = [l, r];
        }
        for (side, cells) in [(0usize, &self.antennae.left), (1, &self.antennae.right)] {
            let current = self.odour[side] * self.coupling.odour_gain;
            if current == 0.0 {
                continue;
            }
            for &i in cells {
                self.drive[i as usize] += current * self.excite[i as usize];
            }
        }

        // --- think -------------------------------------------------------
        self.net.drive(Port::Drive, &self.drive)?;
        let mut motor_spikes = 0u32;
        let mut total_spikes = 0u32;
        // Activation decays once per CONTROL tick, not once per neural tick:
        // it is a muscle's time constant, not the nervous system's.
        for a in self.activation.iter_mut() {
            *a *= self.coupling.activation_decay;
        }
        for o in self.opposed.iter_mut() {
            *o = [0.0; 2];
        }
        if self.wingbeat.is_some() && self.wing_hold.is_none() {
            // The same leak as the muscles, for the same reason: a spike is an
            // impulse and a thorax is not. Power decays toward zero when the
            // power motor neurons stop, which is what makes a fly stop flying.
            self.wing_cmd.power *= self.coupling.activation_decay;
            for v in self.wing_cmd.amplitude.iter_mut().chain(self.wing_cmd.aoa.iter_mut()) {
                *v *= self.coupling.activation_decay;
            }
        }
        for _ in 0..self.timing.neural_per_control {
            self.net.step();
            self.net.read(Port::Spike, &mut self.spike)?;
            let spike = &self.spike;
            total_spikes += spike.iter().filter(|&&s| s > 0.5).count() as u32;
            self.last_proprio_spikes =
                self.sensors.iter().filter(|s| spike[s.neuron as usize] > 0.5).count() as u32;
            let fired = |cells: &[u32]| cells.iter().filter(|&&i| spike[i as usize] > 0.5).count() as u32;
            self.last_antenna_spikes = (fired(&self.antennae.left), fired(&self.antennae.right));
            for d in &self.map.drives {
                if spike[d.neuron as usize] > 0.5 {
                    self.activation[d.actuator] += self.coupling.activation_gain * d.polarity;
                    self.opposed[d.actuator][usize::from(d.polarity < 0.0)] += 1.0;
                    motor_spikes += 1;
                }
            }
            if self.wingbeat.is_some() && self.wing_hold.is_none() {
                for d in &self.wing_drives {
                    if spike[d.neuron as usize] <= 0.5 {
                        continue;
                    }
                    motor_spikes += 1;
                    let w = match d.side {
                        Some(flybody::Side::Left) => 0,
                        Some(flybody::Side::Right) => 1,
                        None => usize::MAX,
                    };
                    match d.action {
                        // A power motor neuron works the whole thorax, so it
                        // has no side even though its soma does.
                        flybody::WingAction::Power => {
                            self.wing_cmd.power += self.coupling.wing_power_gain;
                        }
                        flybody::WingAction::Amplitude(p) => {
                            self.wing_cmd.amplitude[w] += self.coupling.wing_steer_gain * p;
                        }
                        flybody::WingAction::AngleOfAttack(p) => {
                            self.wing_cmd.aoa[w] += self.coupling.wing_steer_gain * p;
                        }
                    }
                }
                self.wing_cmd.power = self.wing_cmd.power.clamp(0.0, 1.0);
                for v in self.wing_cmd.amplitude.iter_mut().chain(self.wing_cmd.aoa.iter_mut()) {
                    *v = v.clamp(-1.0, 1.0);
                }
            }
        }

        let cord = began.elapsed();

        // --- act ---------------------------------------------------------
        let ctrl: Vec<f64> = self
            .activation
            .iter()
            .zip(&self.muscle)
            .map(|(a, m)| (a * m).clamp(-1.0, 1.0) as f64)
            .collect();
        self.data.set(&self.model, StateSpec::CTRL, &ctrl)?;

        // --- integrate ---------------------------------------------------
        let began = std::time::Instant::now();
        match self.wingbeat {
            // The stroke is written EVERY physics step. At 218 Hz a wingbeat
            // lasts 4.6 ms and a control tick is 2 ms, so writing it once per
            // control tick would sample the stroke barely twice and alias it
            // into a slow wobble - the wings would move, the fly would not fly,
            // and every other reading would look healthy.
            Some(mut beat) => {
                let dt = self.timing.physics_dt;
                let mut ctrl = ctrl;
                for _ in 0..self.timing.physics_per_control {
                    beat.advance(dt);
                    let cmd = self.wing_hold.unwrap_or(self.wing_cmd);
                    for w in 0..2 {
                        let t = beat.torques(w, &cmd);
                        for (d, value) in t.iter().enumerate() {
                            if let Some(i) = self.wing_actuators[w][d] {
                                ctrl[i] = value * self.muscle[i] as f64;
                            }
                        }
                    }
                    self.data.set(&self.model, StateSpec::CTRL, &ctrl)?;
                    self.data.step(&self.model);
                }
                self.wingbeat = Some(beat);
            }
            None => {
                for _ in 0..self.timing.physics_per_control {
                    self.data.step(&self.model);
                }
            }
        }

        let body = began.elapsed();

        self.control_tick += 1;
        Ok(Tick { control_tick: self.control_tick, motor_spikes, total_spikes, cord, body })
    }

    /// This sensor's reading, from the generalized coordinates it was RESOLVED
    /// to at construction.
    ///
    /// The resolution happens once because it used to happen here, every
    /// sensor, every control tick: an angle sensor formatted its actuator's
    /// name with `format!` and then linear-searched 44 strings for it, and a
    /// load sensor formatted a suffix, scanned the same 44 with `ends_with`
    /// and collected the matches into a fresh `Vec`. Three hundred and four
    /// sensors at 500 Hz is 150,000 string allocations a second to answer a
    /// question whose answer cannot change: which joint a fixed sensor watches
    /// is a property of the body, fixed when the body is loaded.
    fn sensor_current(&self, i: usize, qpos: &[f64], qvel: &[f64]) -> f32 {
        match &self.sensor_target[i] {
            SensorTarget::Angle(Some(q)) => {
                qpos.get(*q).copied().unwrap_or(0.0) as f32 * self.coupling.angle_gain
            }
            SensorTarget::Angle(None) => 0.0,
            SensorTarget::Load(qs) if !qs.is_empty() => {
                // Mean joint speed of this leg. See `Modality::LoadProxy` for
                // why this is a proxy and not the quantity itself.
                let sum: f64 = qs.iter().filter_map(|&q| qvel.get(q)).map(|v| v.abs()).sum();
                (sum / qs.len() as f64) as f32 * self.coupling.load_gain
            }
            SensorTarget::Load(_) => 0.0,
        }
    }
}

/// Which generalized coordinates a sensor reads, resolved once.
enum SensorTarget {
    /// One joint's angle, or `None` where this body has no such actuator.
    Angle(Option<usize>),
    /// Every joint of one leg, for the load proxy.
    Load(Vec<usize>),
}

/// Resolve every sensor against a body's actuator list, once.
fn resolve_sensors(sensors: &[Sensor], names: &[String], qpos_of: &[usize]) -> Vec<SensorTarget> {
    sensors
        .iter()
        .map(|s| match s.modality {
            Modality::Angle(dof) => {
                let want = dof.actuator(s.segment, s.side);
                SensorTarget::Angle(names.iter().position(|n| *n == want).and_then(|i| qpos_of.get(i).copied()))
            }
            Modality::LoadProxy => {
                let suffix = format!("_{}_{}", seg_str(s.segment), side_str(s.side));
                SensorTarget::Load(
                    names
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| n.ends_with(&suffix))
                        .filter_map(|(i, _)| qpos_of.get(i).copied())
                        .collect(),
                )
            }
        })
        .collect()
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
