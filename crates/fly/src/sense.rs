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
