// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning a body's state into current for sensory neurons.

use connectome::{Connectome, Neuron};
use flybody::{LegDof, Segment, Side};

/// What a proprioceptor reports.
///
/// Assigned from the sensillum CLASS, which is real anatomy rather than a
/// convenience:
///
/// * a chordotonal organ is a stretch receptor spanning a joint - the femoral
///   one is the fly's femur-tibia angle sensor;
/// * a hair plate is a field of bristles at a joint's limit, deflected as the
///   segments close, so it reports proximal joint angle;
/// * campaniform sensilla are cuticular strain gauges: they report LOAD, not
///   position;
/// * strand receptors span the femur.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    /// Angle of a named joint on this neuron's own leg.
    Angle(LegDof),
    /// A stand-in for cuticular load.
    ///
    /// Campaniform sensilla measure strain, and strain is not reachable
    /// through MuJoCo's flat state API - `mjSTATE_QFRC_APPLIED` is the force a
    /// caller applied, not the constraint force a leg is bearing. Driven from
    /// the leg's joint speed instead, which correlates with load during stance
    /// but is NOT the same quantity. Named `LoadProxy` rather than `Load` so
    /// nothing downstream mistakes it for the real thing.
    LoadProxy,
}

/// One sensory neuron's attachment to the body.
#[derive(Clone, Copy, Debug)]
pub struct Sensor {
    /// Index into the connectome's neuron list.
    pub neuron: u32,
    pub segment: Segment,
    pub side: Side,
    pub modality: Modality,
}

/// Parse a sensory `Sub Class` such as `SN-prothoracic_leg-LegNpT1_L`.
fn parse_leg(sub: &str) -> Option<(Segment, Side)> {
    let (_, np) = sub.rsplit_once('-')?;
    let (np, side) = np.rsplit_once('_')?;
    let seg = match np {
        "LegNpT1" => Segment::T1,
        "LegNpT2" => Segment::T2,
        "LegNpT3" => Segment::T3,
        _ => return None,
    };
    let side = match side {
        "L" => Side::Left,
        "R" => Side::Right,
        _ => return None,
    };
    Some((seg, side))
}

fn modality_of(class: &str) -> Option<Modality> {
    Some(match class {
        // The femoral chordotonal organ is the femur-tibia angle sensor.
        "chordotonal_organ" => Modality::Angle(LegDof::Tibia),
        // Hair plates sit at the thorax-coxa and coxa-trochanter joints.
        "hair_plate" => Modality::Angle(LegDof::Coxa),
        "strand_receptor" => Modality::Angle(LegDof::Femur),
        "campaniform_sensilla" => Modality::LoadProxy,
        _ => return None,
    })
}

/// Every leg proprioceptor in `c`, attached to a leg and a modality.
///
/// Tactile bristles and taste bristles are deliberately excluded: they are
/// sensory, but they report contact and chemistry rather than the body's own
/// configuration, and a walking loop has nothing to drive them with.
pub fn proprioceptors(c: &Connectome) -> Vec<Sensor> {
    let mut out = Vec::new();
    for (i, n) in c.neurons.iter().enumerate() {
        if !is_sensory(n) {
            continue;
        }
        let Some(modality) = modality_of(&n.class) else { continue };
        let Some((segment, side)) = parse_leg(&n.sub_class) else { continue };
        out.push(Sensor { neuron: i as u32, segment, side, modality });
    }
    out
}

fn is_sensory(n: &Neuron) -> bool {
    matches!(n.super_class.as_str(), "sensory" | "sensory_ascending")
}

/// The fly's antennae, as two populations of olfactory receptor neurons.
///
/// ## Why smell and not sight
///
/// A fly finds food by smell, and this connectome supports it far better than
/// it supports vision. BANC carries 3,007 olfactory receptor neurons across 56
/// receptor types, split 1,617 left and 1,390 right - a bilateral pair, which
/// is what a chemotactic animal steers on. Its photoreceptors are 1,846 cells
/// of which 1,597 are on the RIGHT and 249 on the left, and they are R7 and R8
/// only: the colour-sensitive inner pair, without the R1-R6 that carry motion.
/// Steering on an eye that is reconstructed on one side is not a measurement of
/// anything, so the exteroceptive channel here is the nose.
///
/// ## What a caller supplies, and what it does NOT
///
/// Two concentrations, one per antenna. That is a STIMULUS, not a command: the
/// difference between them is a fraction of a percent at any useful distance,
/// and what to do about it is the brain's problem. Nothing here turns the
/// animal - `Fly::smell` injects current into the receptor neurons and the
/// consequence, if any, has to come out of the wiring.
#[derive(Clone, Debug, Default)]
pub struct Antennae {
    pub left: Vec<u32>,
    pub right: Vec<u32>,
}

impl Antennae {
    /// Select them out of a connectome, by the export's own annotation.
    ///
    /// Maxillary palp receptors come along with the antennal ones: they are
    /// the fly's second olfactory organ, they are sided the same way, and
    /// dropping them would be a modelling choice made by accident.
    pub fn of(c: &Connectome) -> Antennae {
        let mut a = Antennae::default();
        for (i, n) in c.neurons.iter().enumerate() {
            if n.class != "olfactory_receptor_neuron" {
                continue;
            }
            match n.soma_side.as_str() {
                "left" => a.left.push(i as u32),
                "right" => a.right.push(i as u32),
                // A receptor nobody could side cannot contribute to a
                // bilateral comparison, and averaging it into both sides
                // would dilute the very difference being measured.
                _ => {}
            }
        }
        a
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }
}


/// How far the odour carries, in centimetres.
///
/// A plume nobody can smell from across the arena makes the sense useless; one
/// that saturates everywhere carries no gradient. Two centimetres is eight
/// body lengths, which is the scale the arena is built at.
pub const PLUME_DECAY_CM: f64 = 2.0;
/// Where the antennae sit relative to the root, in centimetres: a little ahead
/// of it and to either side of the midline, on a body 0.25 cm long.
///
/// Derived from the root pose rather than from the antenna BODIES, which would
/// need mjData's `xpos`. At this baseline the two differ by far less than the
/// plume varies over one body length.
pub const HEAD_AHEAD_CM: f64 = 0.10;
pub const ANTENNA_HALF_BASE_CM: f64 = 0.012;

/// Odour concentration at each antenna, `(left, right)`.
///
/// `exp(-r / PLUME_DECAY_CM)` from the source, normalised to 1.0 at the food
/// itself: a diffusive plume with no wind, because the arena has none and a
/// plume model with a wind direction nothing else in the simulation knows
/// about would be inventing physics to sense.
///
/// The bilateral difference this produces is SMALL - the antennae are about a
/// tenth of a body length apart and the field is smooth - which is a fact
/// about fly chemotaxis rather than a shortcoming: a real fly turns on a
/// difference of a few percent and supplements it by casting, which is a
/// behaviour and not a sensor.
pub fn plume(food: [f64; 3], position: [f64; 3], yaw: f64) -> (f32, f32) {
    // Forward is +x at yaw 0 and left is +y, the same convention the bearing
    // to the food is computed in.
    let (c, s) = (yaw.cos(), yaw.sin());
    let at = |left: f64| {
        let (px, py) = (
            position[0] + c * HEAD_AHEAD_CM - s * left,
            position[1] + s * HEAD_AHEAD_CM + c * left,
        );
        let r = ((food[0] - px).powi(2) + (food[1] - py).powi(2) + (food[2] - position[2]).powi(2)).sqrt();
        (-r / PLUME_DECAY_CM).exp() as f32
    };
    (at(ANTENNA_HALF_BASE_CM), at(-ANTENNA_HALF_BASE_CM))
}

/// Yaw and the distance to a point, from a root pose in MuJoCo's `qpos` order.
pub fn yaw_and_range(qpos: &[f64], to: [f64; 3]) -> (f64, f64) {
    let (w, x, y, z) = (
        qpos.get(3).copied().unwrap_or(1.0),
        qpos.get(4).copied().unwrap_or(0.0),
        qpos.get(5).copied().unwrap_or(0.0),
        qpos.get(6).copied().unwrap_or(0.0),
    );
    let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
    let (px, py, pz) = (
        qpos.first().copied().unwrap_or(0.0),
        qpos.get(1).copied().unwrap_or(0.0),
        qpos.get(2).copied().unwrap_or(0.0),
    );
    let range = ((to[0] - px).powi(2) + (to[1] - py).powi(2) + (to[2] - pz).powi(2)).sqrt();
    (yaw, range)
}
