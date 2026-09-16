// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`Creature`]: a connectome running a body.
//!
//! The rest of this SDK loads a static function and calls it. A creature is
//! not that: it has state that survives a call, it is coupled to a physics
//! simulation, and its parameters change while it runs. So the shape here is
//! a builder and a stepping loop rather than `from_pretrained` + `generate`,
//! and the surface is deliberately small - build it, drive it, step it, look
//! at it, and switch off one part at a time to see what that part was doing.
//!
//! ```no_run
//! let mut fly = brain::Creature::fruit_fly()
//!     .connectome("resources/connectome")
//!     .body("flybody/floor.xml")
//!     .build()?;
//! fly.drive(1.5);
//! for _ in 0..500 {
//!     fly.step()?;
//! }
//! println!("{:?} after one second", fly.position());
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! ## What the controls are for
//!
//! [`Creature::set_plasticity`], [`Creature::set_proprioception`] and
//! [`CreatureBuilder::shuffled_connectome`] are on the public surface rather
//! than buried in an experiment binary, because they are the only way to tell
//! a result from a coincidence. A creature that behaves identically with its
//! sensing lesioned was never using it; one that does as well on a
//! degree-matched shuffle of its wiring was not using the connectome. Anyone
//! embedding this should be able to run those checks without reaching past
//! the SDK.
//!
//! Swedish Embedded AB implements embodied neural simulation for its clients -
//! a real connectome, a real body, and the controls that tell you which of
//! them produced the result. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use crate::{Device, Error};

fn backend(e: String) -> Error {
    Error::Backend(e)
}

/// A required builder argument was never set -- see [`Error::MissingArgument`]'s
/// own doc for why this is a distinct variant from [`backend`].
fn missing_argument(e: String) -> Error {
    Error::MissingArgument(e)
}

/// How a creature is assembled. Values, not features: a step count and a
/// weight scale are configuration, and configuration belongs in a builder.
pub struct CreatureBuilder {
    connectome: Option<PathBuf>,
    cns: fly::Cns,
    body: Option<PathBuf>,
    dataset: String,
    weight_scale: f32,
    shuffle: Option<u64>,
    plasticity: bool,
    arena: Arena,
    timestep: Option<f64>,
    food: Option<[f64; 3]>,
    device: Device,
}

pub use flybody::Arena;

impl CreatureBuilder {
    /// Where the connectome export is.
    ///
    /// Either the directory holding `neurons.csv.gz` and
    /// `connections_princeton.csv.gz` directly, or a parent containing a
    /// `manc/` or `manc-codex/` beside them. All three layouts are what a
    /// download of this data actually produces.
    pub fn connectome(mut self, dir: impl AsRef<Path>) -> CreatureBuilder {
        self.connectome = Some(dir.as_ref().to_path_buf());
        self
    }

    /// The fruit-fly MJCF - the BODY model, not a scene.
    ///
    /// The scene is generated: floor, sky, light, and whatever else the arena
    /// calls for. That is not convenience. A body file with no floor simulates
    /// a fly falling forever while every other reading looks healthy, and the
    /// flight arena needs four changes to the published model before a
    /// wingbeat produces any lift at all. Neither is something a caller should
    /// have to know to get right.
    pub fn body(mut self, xml: impl AsRef<Path>) -> CreatureBuilder {
        self.body = Some(xml.as_ref().to_path_buf());
        self
    }

    /// Ground to walk on, or air to fly through. See [`Arena`].
    pub fn arena(mut self, arena: Arena) -> CreatureBuilder {
        self.arena = arena;
        self
    }

    /// Which GPU/backend the spiking network runs on. Defaults to
    /// [`Device::default`] ("auto" -- whatever hardware the machine has),
    /// the same default [`crate::ImagePipelineBuilder::device`] uses.
    pub fn device(mut self, device: Device) -> CreatureBuilder {
        self.device = device;
        self
    }

    /// Integrate the body at `dt` seconds instead of the published 1e-4.
    ///
    /// The one dial that decides whether a walking fly can be watched at
    /// natural speed, and it buys that by giving up contact accuracy. Physics
    /// cost per simulated second is `1/dt` steps at a fixed cost each, so
    /// MuJoCo's own measured 0.30x real time for this model at 1e-4 becomes
    /// 0.62x at 2e-4 and 1.5x at 4e-4 - and the floor softens in step, because
    /// its contact time constant has to stay at twice the timestep to stay
    /// solvable. The control rate is unchanged: a control tick is 2 ms of body
    /// time whatever the integrator does inside it.
    ///
    /// Leave it alone for anything being measured. It is here so that a person
    /// watching the animal can choose to watch it move at its own speed, which
    /// is a different purpose from measuring its gait.
    pub fn timestep(mut self, dt: f64) -> CreatureBuilder {
        self.timestep = Some(dt);
        self
    }

    /// Put something in the world to go towards, in the model's own
    /// centimetres. The fly's body is about 0.25 cm long.
    pub fn food(mut self, at: [f64; 3]) -> CreatureBuilder {
        self.food = Some(at);
        self
    }

    /// Give the animal a BRAIN.
    ///
    /// The default is the ventral nerve cord alone, which is what MANC is:
    /// everything below the neck, with no eyes and nothing that decides where
    /// to go. The descending command then has to come from the caller, which
    /// is the caller standing in for the missing half of the animal.
    ///
    /// With this on, BANC's brain is joined to that cord at the 3,530
    /// published crossing cells (`connectome::bridge`), and the descending
    /// population is driven by the brain instead. It is eight times the
    /// neurons and twice the edges, and it needs `banc/` beside `manc/` under
    /// the connectome directory - see `tools/convert/banc_codex.py`.
    pub fn brain(mut self, on: bool) -> CreatureBuilder {
        self.cns = if on { fly::Cns::BrainAndCord } else { fly::Cns::Cord };
        self
    }

    /// Raw synapse count to membrane current. The connectome carries counts,
    /// not strengths, and nothing in the data fixes this conversion.
    pub fn weight_scale(mut self, scale: f32) -> CreatureBuilder {
        self.weight_scale = scale;
        self
    }

    /// Replace the wiring with a degree-matched shuffle of it: same in-degrees,
    /// same weights, sources randomly reassigned. The structural control.
    pub fn shuffled_connectome(mut self, seed: u64) -> CreatureBuilder {
        self.shuffle = Some(seed);
        self
    }

    /// Whether synapses may change. Off by default, because a creature that
    /// learns while you are measuring something else is not a measurement.
    pub fn plasticity(mut self, on: bool) -> CreatureBuilder {
        self.plasticity = on;
        self
    }

    /// Load the connectome and the body and wire them together.
    ///
    /// This is the expensive call: it reads the connectome from disk, uploads
    /// several million synapses to the device, and compiles the MJCF.
    pub fn build(self) -> Result<Creature, Error> {
        let dir = self.connectome.ok_or_else(|| missing_argument("no connectome directory set; call .connectome(dir)".into()))?;
        let body = self.body.ok_or_else(|| missing_argument("no body set; call .body(scene.xml)".into()))?;
        // Tolerant of how the export was unpacked - see `connectome::find`.
        // A caller should not have to know whether their download put the two
        // CSVs in `manc/`, in `manc-codex/`, or at the root.
        let c = match self.cns {
            // `dataset` still selects the cord, so a caller with a differently
            // named export is not forced through the joined path to use it.
            fly::Cns::Cord => {
                let (neurons, edges) = connectome::find(&dir, &self.dataset).map_err(backend)?;
                connectome::load(&self.dataset, &neurons, &edges).map_err(backend)?
            }
            fly::Cns::BrainAndCord => fly::cns::load(&dir, fly::Cns::BrainAndCord).map_err(backend)?,
        };

        let mj = mujoco::MuJoCo::load().map_err(backend)?;
        // The generated scene lives in a scratch directory the creature keeps
        // alive: MuJoCo reads the file at load and never again, but a caller
        // who wants to look at what was generated should find it still there.
        let scratch = tempfile::tempdir().map_err(|e| backend(format!("no scratch directory: {e}")))?;
        let world =
            flybody::World { arena: self.arena, food: self.food, timestep: self.timestep, ..flybody::World::default() };
        let scene = flybody::world(&body, scratch.path(), world).map_err(backend)?;
        let model = mujoco::Model::from_xml(&mj, &scene).map_err(backend)?;
        let lif = fly::cord_lif();
        // Flight integrates ten times finer than walking and needs it: a
        // 218 Hz wingbeat resolved at the walking timestep integrates the
        // stroke adequately and the fluid forces on a reversing wing badly.
        // A control tick is 2 ms of BODY time whatever the integrator does
        // inside it, so the substep count is derived from the timestep rather
        // than written next to it: the two disagreeing is a loop whose
        // nervous system and body run at different speeds, which looks like a
        // behaviour change rather than like a bug.
        let timing = match self.arena {
            Arena::Ground => match self.timestep {
                None => fly::Timing::default(),
                Some(dt) => fly::Timing {
                    neural_per_control: 1,
                    physics_per_control: fly::Timing::substeps(dt).map_err(backend)?,
                    physics_dt: dt,
                },
            },
            Arena::Air => fly::Timing {
                neural_per_control: 1,
                physics_per_control: fly::Timing::substeps(world.flight.timestep).map_err(backend)?,
                physics_dt: world.flight.timestep,
            },
        };
        crate::device::resolve(&self.device)?;
        let gpu = gpu_core::Gpu::new(&neuro::KERNELS);
        let mut inner = fly::Fly::new(
            gpu,
            &c,
            model,
            lif,
            fly::Wiring { weight_scale: self.weight_scale, shuffle_seed: self.shuffle, ..fly::Wiring::default() },
            timing,
            fly::Coupling::default(),
        )
        .map_err(backend)?;
        if self.plasticity {
            // Clamp sized from the connectome's own weight range rather than
            // picked: the weights are scaled synapse counts running to tens,
            // so a fixed small bound squashes every one of them on the first
            // update and destroys the graph.
            let bound = 1.5 * inner.initial_weight_scale();
            inner
                .enable_plasticity(neuro::PlasticityParams { eta: 0.02, w_min: -bound, w_max: bound, ..Default::default() })
                .map_err(backend)?;
        }
        if self.arena == Arena::Air {
            // 180 Hz rather than the animal's 218: this airframe's wing hinge
            // resonates lower than a real thorax does, measured by sweeping
            // the drive frequency rather than assumed from the biology.
            //
            // Feathering at 0.4 rather than the default 0.6, and the actuator
            // gain left at the published 18. Both are measurements: sweeping
            // the gain to 30 and 45 makes the stroke bigger and the flight
            // WORSE - 90.8% of body weight supported at 18, 73.2% at 30, 59.4%
            // at 45 - because the wing over-rotates past a useful angle of
            // attack and stalls. The published value is well chosen and a
            // bigger number is not a better one.
            inner
                .enable_flight(fly::Wingbeat { hz: 180.0, feather: 0.4, ..fly::Wingbeat::default() })
                .map_err(backend)?;
            // Retract the legs, which is what a flying fly does and what the
            // published flight tasks do to this model. Leaving them driven
            // means six limbs being flailed by a cord that has no idea it is
            // airborne: they add drag, they catch the floor on the way past,
            // and the difference reads as the wings underperforming.
            let legs: Vec<usize> = inner
                .actuator_names()
                .iter()
                .enumerate()
                .filter(|(_, n)| !n.starts_with("wing_"))
                .map(|(i, _)| i)
                .collect();
            for i in legs {
                // Cannot fail: the index came from this body's own list.
                let _ = inner.set_muscle_strength(i, 0.0);
            }
        }
        let neurons = c.neurons.len();
        // An air arena starts the fly ALOFT. Left on the floor with its wings
        // beating, what gets measured is a fly skimming a surface, which looks
        // like flight in the numbers and is not.
        let mut creature = Creature {
            inner,
            mj,
            neurons,
            drive: 0.0,
            turn: 0.0,
            food: self.food,
            extent: flybody::world_extent(world),
            // Twenty centimetres, which is eighty body lengths and an
            // ordinary height for a fly. It has to be this high because the
            // wings carry about 90% of the body's weight and not 100%: the fly
            // sinks at roughly 14 cm/s, so altitude is flight TIME. A
            // centimetre sounds like plenty and buys 45 milliseconds.
            start_height: match self.arena {
                Arena::Air => 20.0,
                Arena::Ground => 0.0,
            },
            _scratch: scratch,
        };
        creature.reset();
        Ok(creature)
    }
}

/// How far the odour carries, in centimetres. A plume nobody can smell from
/// across the arena makes the sense useless; one that saturates everywhere
/// carries no gradient. Two centimetres is eight body lengths, which is the
/// scale the arena is built at.
const DECAY_CM: f64 = 2.0;
/// Where the antennae sit relative to the root, in centimetres: a little ahead
/// of it and to either side of the midline, on a body 0.25 cm long.
const HEAD_AHEAD_CM: f64 = 0.10;
const ANTENNA_HALF_BASE_CM: f64 = 0.012;

/// A connectome driving a body.
pub struct Creature {
    inner: fly::Fly,
    mj: std::sync::Arc<mujoco::MuJoCo>,
    neurons: usize,
    drive: f32,
    turn: f32,
    food: Option<[f64; 3]>,
    /// How far above its resting pose the body starts, in centimetres.
    start_height: f64,
    /// The scene's `<statistic extent>`, for anything pointing a camera.
    extent: f64,
    // Holds the generated scene on disk for as long as the creature lives.
    _scratch: tempfile::TempDir,
}

impl Creature {
    /// A *Drosophila melanogaster* ventral nerve cord driving the flybody
    /// model. Nothing is loaded until [`CreatureBuilder::build`].
    pub fn fruit_fly() -> CreatureBuilder {
        CreatureBuilder {
            connectome: None,
            cns: fly::Cns::Cord,
            body: None,
            dataset: "manc".to_string(),
            weight_scale: fly::Wiring::default().weight_scale,
            shuffle: None,
            plasticity: false,
            arena: Arena::Ground,
            timestep: None,
            food: None,
            device: Device::default(),
        }
    }

    /// Neurons in the connectome.
    pub fn neurons(&self) -> usize {
        self.neurons
    }

    /// How the motor neurons attached to the body's actuators, in one line.
    pub fn wiring(&self) -> String {
        self.inner.motor_map().summary()
    }

    /// Set the standing descending command: how hard the brain is telling the
    /// cord to go.
    pub fn drive(&mut self, forward: f32) {
        self.drive = forward;
        self.apply_command();
    }

    /// Bias the descending command to turn: `turn` in `[-1, 1]` weights the
    /// two halves of the descending population against each other, so a turn
    /// has to come out of the cord's own circuitry rather than from anything
    /// steering the body directly.
    pub fn turn(&mut self, turn: f32) {
        self.turn = turn.clamp(-1.0, 1.0);
        self.apply_command();
    }

    fn apply_command(&mut self) {
        let n = self.inner.descending_count();
        let command: Vec<f32> = (0..n)
            .map(|i| self.drive * if i * 2 < n { 1.0 + self.turn * 0.5 } else { 1.0 - self.turn * 0.5 })
            .collect();
        // Cannot fail: the length is taken from the creature itself.
        let _ = self.inner.set_descending(&command);
    }

    /// The scene's declared extent, which is what MuJoCo scales every camera
    /// gesture by.
    pub fn scene_extent(&self) -> f64 {
        self.extent
    }

    /// Where the food is, if this world has any.
    pub fn food(&self) -> Option<[f64; 3]> {
        self.food
    }

    /// Distance and bearing to the food, in the FLY's own frame: `(range in
    /// centimetres, bearing in radians, positive to the left)`.
    ///
    /// Computed from the body's position and orientation alone, both of which
    /// are in the state vector - no MuJoCo struct is read and no site is
    /// queried, because the food does not move and its position is something
    /// this creature was told rather than something it has to look up.
    pub fn bearing_to_food(&self) -> Option<(f64, f64)> {
        let food = self.food?;
        let q = self.inner.qpos();
        let (px, py) = (q.first().copied().unwrap_or(0.0), q.get(1).copied().unwrap_or(0.0));
        let (dx, dy) = (food[0] - px, food[1] - py);
        let range = (dx * dx + dy * dy + (food[2] - q.get(2).copied().unwrap_or(0.0)).powi(2)).sqrt();
        // Yaw from the root quaternion, MuJoCo's (w, x, y, z) order. Only the
        // heading matters for a bearing, so this is the standard yaw
        // extraction rather than a full rotation.
        let (w, x, y, z) = (
            q.get(3).copied().unwrap_or(1.0),
            q.get(4).copied().unwrap_or(0.0),
            q.get(5).copied().unwrap_or(0.0),
            q.get(6).copied().unwrap_or(0.0),
        );
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        // Wrapped to (-pi, pi], so a target just behind the left shoulder is a
        // small left turn rather than an almost full circle to the right.
        let mut bearing = dy.atan2(dx) - yaw;
        while bearing > std::f64::consts::PI {
            bearing -= 2.0 * std::f64::consts::PI;
        }
        while bearing <= -std::f64::consts::PI {
            bearing += 2.0 * std::f64::consts::PI;
        }
        Some((range, bearing))
    }

    /// Steer towards the food, returning the range left.
    ///
    /// The bearing is computed here and delivered through the DESCENDING
    /// command, which is where it belongs anatomically rather than as a
    /// convenience. This connectome is the ventral nerve cord; a fly's
    /// navigation happens in its brain, and the only thing the brain sends
    /// down is a descending command. So a controller that works out which way
    /// to go and pushes it into the descending population is standing in for
    /// the missing half of the animal, in exactly the place the missing half
    /// would have connected.
    ///
    /// `None` when there is nothing to seek.
    pub fn seek_food(&mut self, forward: f32) -> Option<f64> {
        let (range, bearing) = self.bearing_to_food()?;
        // Saturating rather than proportional: a fly does not turn twice as
        // hard for a target twice as far off to the side, and a linear law
        // makes the approach oscillate as the bearing crosses zero.
        let turn = (bearing / std::f64::consts::FRAC_PI_4).clamp(-1.0, 1.0) as f32;
        self.turn(turn);
        // Slow down when nearly there, so arriving does not mean overshooting.
        self.drive(forward * (range as f32 / 1.0).clamp(0.25, 1.0));
        Some(range)
    }

    /// Let the animal SMELL the food, and report the two concentrations.
    ///
    /// The other half of `seek_food`, and the honest one. `seek_food` decides
    /// where to go and pushes the answer into the descending population: it is
    /// the missing brain, written by hand. This instead puts an odour on the
    /// antennae and leaves the deciding to the nervous system - which is only
    /// meaningful on a creature that HAS a brain
    /// ([`CreatureBuilder::brain`]), since a nerve cord has no nose.
    ///
    /// The field is `exp(-r / DECAY_CM)` from the source, evaluated at each
    /// antenna: a diffusive plume without wind, normalised to 1.0 at the food
    /// itself. No wind, because flybody's arena has none and a plume model
    /// with a wind direction nothing else in the simulation knows about would
    /// be inventing physics to sense.
    ///
    /// The bilateral difference this produces is SMALL - the antennae are
    /// about a tenth of a body length apart and the field is smooth - which is
    /// a fact about fly chemotaxis rather than a shortcoming: a real fly turns
    /// on a difference of a few percent and supplements it by casting, which
    /// is a behaviour and not a sensor. Returns `(left, right)`, or `None`
    /// when there is nothing to smell.
    pub fn smell_food(&mut self) -> Option<(f32, f32)> {
        let food = self.food?;
        let (yaw, _) = self.attitude();
        let [x, y, z] = self.position();
        // Forward is +x at yaw 0 and left is +y, matching
        // `bearing_to_food`'s own convention; the antennae sit ahead of the
        // root and to either side of the midline. Derived from the root pose
        // rather than from the antenna BODIES, which would need mjData's
        // xpos: at this baseline the two differ by far less than the plume
        // varies over one body length.
        let (c, s) = (yaw.cos(), yaw.sin());
        let at = |ahead: f64, left: f64| {
            let (px, py) = (x + c * ahead - s * left, y + s * ahead + c * left);
            let r = ((food[0] - px).powi(2) + (food[1] - py).powi(2) + (food[2] - z).powi(2)).sqrt();
            (-r / DECAY_CM).exp() as f32
        };
        let pair = (at(HEAD_AHEAD_CM, ANTENNA_HALF_BASE_CM), at(HEAD_AHEAD_CM, -ANTENNA_HALF_BASE_CM));
        self.inner.smell(pair.0, pair.1);
        Some(pair)
    }

    /// How many olfactory receptor neurons fired on the last tick, by side.
    ///
    /// A sensory channel that is connected and silent looks exactly like one
    /// that is working, from every other reading. This is how to tell.
    pub fn antenna_spikes(&self) -> (u32, u32) {
        self.inner.antenna_spikes()
    }

    /// How many receptor neurons the animal has, by side. `(0, 0)` without a
    /// brain.
    pub fn antenna_counts(&self) -> (usize, usize) {
        self.inner.antenna_counts()
    }

    /// Whether the fly has reached the food, within one body length.
    pub fn reached_food(&self) -> bool {
        // 0.25 cm is a fruit fly's body length; anything tighter is asking the
        // physics for a precision the contact solver does not have.
        self.bearing_to_food().is_some_and(|(range, _)| range < 0.25)
    }

    /// Whether this creature has wings running at all.
    pub fn flying(&self) -> bool {
        self.inner.wingbeat().is_some()
    }

    /// Drive the wings directly, `0.0` for stopped and `1.0` for full power.
    ///
    /// Overrides what the wing motor neurons are producing. That is an honest
    /// override rather than a hidden one: the cord's wing output is measured
    /// and reported by [`Creature::wing_command`], and until something trains
    /// it there is nothing to fly on. Pass `None` to hand the wings back.
    pub fn set_wing_power(&mut self, power: Option<f32>) {
        self.inner.hold_wing_command(power.map(|p| fly::WingCommand {
            power: p.clamp(0.0, 1.0),
            ..fly::WingCommand::default()
        }));
    }

    /// What the cord is currently telling the wings to do, whatever is
    /// actually being flown.
    pub fn wing_command(&self) -> (f32, [f32; 2]) {
        let c = self.inner.wing_command();
        (c.power, c.amplitude)
    }

    /// Each wing's `[yaw, roll, pitch]` joint angle, left then right - the
    /// stroke as it actually came out of the physics.
    pub fn wing_angles(&self) -> [[f64; 3]; 2] {
        self.inner.wing_angles()
    }

    /// How many motor neurons attached to the wings, by role.
    pub fn wing_wiring(&self) -> String {
        self.inner.wing_summary()
    }

    /// Heading and body pitch, in radians. Pitch is positive nose-up.
    ///
    /// A fly's attitude is most of what its trajectory means: the same forward
    /// speed at two different body pitches is two different behaviours, and
    /// neither is visible in a position log alone.
    pub fn attitude(&self) -> (f64, f64) {
        let q = self.inner.qpos();
        let (w, x, y, z) = (
            q.get(3).copied().unwrap_or(1.0),
            q.get(4).copied().unwrap_or(0.0),
            q.get(5).copied().unwrap_or(0.0),
            q.get(6).copied().unwrap_or(0.0),
        );
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        // Clamped before the arcsine: a unit quaternion's sine term is in
        // range by construction and a denormalised one is not, and NaN in a
        // trajectory log is worse than a saturated angle.
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        (yaw, pitch)
    }

    /// How fast the body is moving, in centimetres per second.
    pub fn velocity(&self) -> [f64; 3] {
        let v = self.inner.qvel();
        [
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
        ]
    }

    /// Whether the body is resting on the ground rather than in the air.
    ///
    /// Height above the floor rather than a contact query: a contact set says
    /// what is TOUCHING, and a fly with one tarsus brushing the ground on its
    /// way past is not landed.
    ///
    /// The threshold has to clear the standing pose rather than the floor. A
    /// fly stands with its body about a tenth of a centimetre up, and a flying
    /// one with its legs retracted lies LOWER than that when it comes down -
    /// so a threshold set at the floor plane calls a landed fly airborne.
    pub fn grounded(&self) -> bool {
        self.position()[2] < -0.05 && self.velocity()[2].abs() < 5.0
    }

    /// Advance one control tick: 2 ms of body time.
    pub fn step(&mut self) -> Result<Beat, Error> {
        let t = self.inner.step().map_err(backend)?;
        Ok(Beat {
            tick: t.control_tick,
            spikes: t.total_spikes,
            motor_spikes: t.motor_spikes,
            cord: t.cord,
            body: t.body,
        })
    }

    /// Advance `ticks` control ticks, returning their totals.
    pub fn step_for(&mut self, ticks: u32) -> Result<Beat, Error> {
        let mut out = Beat::default();
        for _ in 0..ticks {
            let b = self.step()?;
            out.tick = b.tick;
            out.spikes += b.spikes;
            out.motor_spikes += b.motor_spikes;
            out.cord += b.cord;
            out.body += b.body;
        }
        Ok(out)
    }

    /// Where the body is, in the model's own length units.
    pub fn position(&self) -> [f64; 3] {
        let q = self.inner.qpos();
        [
            q.first().copied().unwrap_or(0.0),
            q.get(1).copied().unwrap_or(0.0),
            q.get(2).copied().unwrap_or(0.0),
        ]
    }

    /// Whether synapses may change.
    pub fn set_plasticity(&mut self, on: bool) {
        self.inner.set_plasticity(on);
    }

    /// Deliver a neuromodulator pulse: the third factor. Positive potentiates
    /// what was recently eligible, negative depresses it.
    pub fn reward(&mut self, delta: f32) {
        self.inner.modulate(delta);
    }

    /// Turn the proprioceptive channel on or off. A control, not a setting.
    pub fn set_proprioception(&mut self, on: bool) {
        self.inner.set_proprioception(on);
    }

    pub fn proprioception(&self) -> bool {
        self.inner.proprioception()
    }

    /// Put the body back where it started, leaving anything learned in place.
    pub fn reset(&mut self) {
        self.inner.reset();
        if self.start_height > 0.0 {
            let mut qpos = self.inner.qpos();
            if let Some(z) = qpos.get_mut(2) {
                *z += self.start_height;
            }
            let qvel = vec![0.0; self.inner.dims().1];
            // Cannot fail: both vectors came from this body's own dimensions.
            let _ = self.inner.set_pose(&qpos, &qvel);
        }
        self.apply_command();
    }

    pub(crate) fn body_handles(&self) -> (&mujoco::Model, &mujoco::Data) {
        self.inner.body()
    }

    pub(crate) fn mujoco(&self) -> &std::sync::Arc<mujoco::MuJoCo> {
        &self.mj
    }
}

/// One control tick's accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Beat {
    /// Control ticks since the creature was built.
    pub tick: u64,
    /// Neurons that fired anywhere in the cord.
    pub spikes: u32,
    /// Of those, motor neurons.
    pub motor_spikes: u32,
    /// Wall time spent in the nervous system, summed over the ticks this beat
    /// covers. See [`Creature::step_for`] - a loop that misses real time needs
    /// to know WHICH half of the animal is over budget, and the two are fixed
    /// by different things.
    pub cord: std::time::Duration,
    /// Wall time spent in the body's physics, over the same ticks.
    pub body: std::time::Duration,
}
