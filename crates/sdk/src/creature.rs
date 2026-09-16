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

use crate::Error;

fn backend(e: String) -> Error {
    Error::Backend(e)
}

/// How a creature is assembled. Values, not features: a step count and a
/// weight scale are configuration, and configuration belongs in a builder.
pub struct CreatureBuilder {
    connectome: Option<PathBuf>,
    body: Option<PathBuf>,
    dataset: String,
    weight_scale: f32,
    shuffle: Option<u64>,
    plasticity: bool,
}

impl CreatureBuilder {
    /// Directory holding the connectome export - for the fruit fly, the one
    /// containing `manc-codex/`.
    pub fn connectome(mut self, dir: impl AsRef<Path>) -> CreatureBuilder {
        self.connectome = Some(dir.as_ref().to_path_buf());
        self
    }

    /// The MJCF scene. Use the one WITH a ground plane: a body file with no
    /// floor simulates a fly falling forever, and every other measurement
    /// still looks healthy while it does.
    pub fn body(mut self, xml: impl AsRef<Path>) -> CreatureBuilder {
        self.body = Some(xml.as_ref().to_path_buf());
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
        let dir = self.connectome.ok_or_else(|| backend("no connectome directory set; call .connectome(dir)".into()))?;
        let body = self.body.ok_or_else(|| backend("no body set; call .body(scene.xml)".into()))?;
        let export = dir.join(format!("{}-codex", self.dataset));
        let c = connectome::load(
            &self.dataset,
            &export.join("neurons.csv.gz"),
            &export.join("connections_princeton.csv.gz"),
        )
        .map_err(backend)?;

        let mj = mujoco::MuJoCo::load().map_err(backend)?;
        let model = mujoco::Model::from_xml(&mj, &body).map_err(backend)?;
        let lif = neuro::LifParams {
            dt_over_tau: 0.2,
            v_th: 1.0,
            r: 1.0,
            refrac_ticks: 1,
            ..neuro::LifParams::default()
        };
        let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
        let mut inner = fly::Fly::new(
            gpu,
            &c,
            model,
            lif,
            self.weight_scale,
            self.shuffle,
            fly::Timing::default(),
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
        let neurons = c.neurons.len();
        Ok(Creature { inner, mj, neurons, drive: 0.0, turn: 0.0 })
    }
}

/// A connectome driving a body.
pub struct Creature {
    inner: fly::Fly,
    mj: std::sync::Arc<mujoco::MuJoCo>,
    neurons: usize,
    drive: f32,
    turn: f32,
}

impl Creature {
    /// A *Drosophila melanogaster* ventral nerve cord driving the flybody
    /// model. Nothing is loaded until [`CreatureBuilder::build`].
    pub fn fruit_fly() -> CreatureBuilder {
        CreatureBuilder {
            connectome: None,
            body: None,
            dataset: "manc".to_string(),
            weight_scale: 3e-2,
            shuffle: None,
            plasticity: false,
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
    /// cord to go, and which way to lean.
    ///
    /// `turn` in `[-1, 1]` biases the two halves of the descending population
    /// against each other, so a turn has to come out of the cord's own
    /// circuitry rather than from anything steering the body directly.
    pub fn drive(&mut self, forward: f32) {
        self.drive = forward;
        self.apply_command();
    }

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

    /// Advance one control tick: 2 ms of body time.
    pub fn step(&mut self) -> Result<Beat, Error> {
        let t = self.inner.step().map_err(backend)?;
        Ok(Beat { tick: t.control_tick, spikes: t.total_spikes, motor_spikes: t.motor_spikes })
    }

    /// Advance `ticks` control ticks, returning their totals.
    pub fn step_for(&mut self, ticks: u32) -> Result<Beat, Error> {
        let mut out = Beat::default();
        for _ in 0..ticks {
            let b = self.step()?;
            out.tick = b.tick;
            out.spikes += b.spikes;
            out.motor_spikes += b.motor_spikes;
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
}
