// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The wingbeat, and why it cannot come out of the motor neurons directly.
//!
//! A leg motor neuron's spike train IS the leg's command: it fires, the muscle
//! pulls, the joint moves. A wing's power muscles do not work that way. They
//! are stretch-activated - asynchronous - and contract many times per motor
//! spike, driving a mechanically resonant thorax at a frequency the thorax
//! sets, not the nervous system. A *Drosophila* wingbeat runs near 218 Hz while
//! its power motor neurons fire at a few tens of hertz. Wiring a wing joint to
//! a power motor neuron's spikes would therefore produce a wingbeat one or two
//! orders of magnitude too slow, and it would look like a modelling detail
//! rather than the category error it is.
//!
//! So the wingbeat is generated here, as an oscillation the nervous system
//! MODULATES rather than produces: the power motor neurons set how hard it is
//! driven, and the steering motor neurons bias each wing's stroke and angle of
//! attack. That division is the anatomy, not a convenience.
//!
//! ## The stroke, and why it is written this way
//!
//! Three angles per wing, driven as TORQUE - which is what flybody's wing
//! actuators take - so what the wing actually does is the response of a sprung,
//! damped, inertial hinge to a periodic drive, exactly as in the animal.
//!
//! * **Stroke** (yaw) is sinusoidal. It is the sweep that does the work.
//! * **Deviation** (roll) runs at TWICE the stroke frequency, which is what
//!   turns a flat sweep into the shallow figure-of-eight a real wing traces.
//! * **Feathering** (pitch) is `tanh(k cos phi)`, not a sinusoid. A real wing
//!   holds a nearly constant angle of attack through each half-stroke and then
//!   flips it fast at the reversal; `tanh` of a cosine is flat over most of the
//!   cycle and sweeps through zero quickly exactly where `cos` crosses it,
//!   which reproduces that shape with one parameter and no lookup table.
//!
//! Swedish Embedded AB implements biomechanical pattern generators for clients
//! simulating animals and animal-like machines. If your team needs a controller
//! whose structure follows the anatomy rather than fighting it, you can procure
//! our services by sending an email to info@swedishembedded.com.

/// A wingbeat pattern generator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wingbeat {
    /// Nominal wingbeat frequency in hertz. A fruit fly runs near 218.
    pub hz: f64,
    /// Peak stroke torque at full power.
    pub stroke: f32,
    /// Peak deviation torque, as a fraction of `stroke`.
    pub deviation: f32,
    /// Peak feathering torque.
    pub feather: f32,
    /// Sharpness of the feathering flip. Larger holds the angle of attack
    /// flatter through the half-stroke and reverses it faster.
    pub flip: f32,
    /// Current phase, radians. Advanced by [`Self::advance`].
    pub phase: f64,
}

impl Default for Wingbeat {
    fn default() -> Self {
        Wingbeat { hz: 218.0, stroke: 1.0, deviation: 0.15, feather: 0.6, flip: 3.0, phase: 0.0 }
    }
}

/// What the nervous system is doing to the wingbeat this instant.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WingCommand {
    /// Power-muscle drive in `[0, 1]`: how hard the thorax is being worked.
    /// Zero means the wings are not being driven at all.
    pub power: f32,
    /// Per-wing stroke amplitude bias, left then right. Asymmetry is how a fly
    /// turns.
    pub amplitude: [f32; 2],
    /// Per-wing angle-of-attack bias, left then right.
    pub aoa: [f32; 2],
}

impl Wingbeat {
    /// Advance the phase by `dt` seconds.
    ///
    /// Kept as a separate call rather than folded into [`Self::torques`] so the
    /// generator can be stepped at the PHYSICS rate while the nervous system
    /// updates its command at the control rate. At 218 Hz a wingbeat lasts
    /// 4.6 ms, so a 2 ms control tick samples it barely twice - the stroke has
    /// to be written inside the physics loop or it is aliased into nonsense.
    pub fn advance(&mut self, dt: f64) {
        self.phase = (self.phase + 2.0 * std::f64::consts::PI * self.hz * dt) % (2.0 * std::f64::consts::PI);
    }

    /// Torques for one wing: `[yaw, roll, pitch]`, each in `[-1, 1]`.
    ///
    /// `wing` is 0 for left and 1 for right. The two wings are driven in
    /// PHASE, not antiphase: a fly's wings sweep together, and the stroke plane
    /// is mirrored by the body's own geometry rather than by the command.
    pub fn torques(&self, wing: usize, cmd: &WingCommand) -> [f64; 3] {
        let amp = (cmd.power * (1.0 + cmd.amplitude.get(wing).copied().unwrap_or(0.0))).clamp(0.0, 2.0);
        let (s, c) = (self.phase.sin(), self.phase.cos());
        let yaw = (self.stroke * amp) as f64 * s;
        // Twice the stroke frequency: sin(2p) = 2 sin p cos p, written that way
        // to avoid a second transcendental call per wing per physics step.
        let roll = (self.stroke * self.deviation * amp) as f64 * 2.0 * s * c;
        let flip = (self.flip as f64 * c).tanh();
        let pitch = (self.feather * amp) as f64 * flip + cmd.aoa.get(wing).copied().unwrap_or(0.0) as f64;
        [yaw.clamp(-1.0, 1.0), roll.clamp(-1.0, 1.0), pitch.clamp(-1.0, 1.0)]
    }
}
