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
