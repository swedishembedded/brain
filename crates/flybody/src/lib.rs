// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a nervous system meets a body.
//!
//! A connectome names its motor neurons by the MUSCLE they innervate; a MuJoCo
//! model names its actuators by the JOINT they move. This crate is the map
//! between those two vocabularies, and it is built from the connectome's own
//! annotations rather than from a hand-written list of neuron ids - so a
//! dataset revision that adds motor neurons picks them up, and one that
//! renames a muscle fails loudly instead of silently dropping it.
//!
//! ## What is and is not established here
//!
//! Which DoF a muscle acts on is anatomy and is asserted with confidence: the
//! tibia flexor moves the tibia. Which SIGN that is, against flybody's own
//! joint axis conventions, is not established by anything in this crate - it
//! is a stated convention (flexors negative, extensors positive) that has to
//! be pinned by replaying recorded kinematics, which is a later milestone.
//! [`MuscleAction::polarity`] says so in its own documentation rather than
//! leaving a reader to assume it was verified.
//!
//! Swedish Embedded AB implements sensorimotor interfaces between neural
//! models and physical or simulated bodies. If your team needs a controller
//! bound to real actuators with the provenance kept intact, you can procure
//! our services by sending an email to info@swedishembedded.com.

use connectome::Connectome;

/// Which thoracic segment a leg belongs to. The connectome writes these as
/// `LegNpT1`/`T2`/`T3`; flybody writes them as `_T1_`/`_T2_`/`_T3_`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// Prothoracic: the front legs.
    T1,
    /// Mesothoracic: the middle legs.
    T2,
    /// Metathoracic: the hind legs.
    T3,
}

impl Segment {
    fn suffix(self) -> &'static str {
        match self {
            Segment::T1 => "T1",
            Segment::T2 => "T2",
            Segment::T3 => "T3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn parse(s: &str) -> Option<Side> {
        match s.trim().to_ascii_lowercase().as_str() {
            "left" | "lhs" => Some(Side::Left),
            "right" | "rhs" => Some(Side::Right),
            _ => None,
        }
    }
    fn suffix(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

/// A flybody leg degree of freedom. These are the eight actuated joints the
/// model exposes per leg, proximal to distal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegDof {
    CoxaAbduct,
    CoxaTwist,
    Coxa,
    FemurTwist,
    Femur,
    Tibia,
    Tarsus,
    Tarsus2,
}

impl LegDof {
    fn stem(self) -> &'static str {
        match self {
            LegDof::CoxaAbduct => "coxa_abduct",
            LegDof::CoxaTwist => "coxa_twist",
            LegDof::Coxa => "coxa",
            LegDof::FemurTwist => "femur_twist",
            LegDof::Femur => "femur",
            LegDof::Tibia => "tibia",
            LegDof::Tarsus => "tarsus",
            LegDof::Tarsus2 => "tarsus2",
        }
    }

    /// The flybody actuator name for this DoF on one leg, e.g.
    /// `coxa_abduct_T1_left`.
    pub fn actuator(self, seg: Segment, side: Side) -> String {
        format!("{}_{}_{}", self.stem(), seg.suffix(), side.suffix())
    }
}

/// How one named muscle acts on one degree of freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuscleAction {
    pub dof: LegDof,
    /// Agonist/antagonist polarity, `+1.0` or `-1.0`.
    ///
    /// Two things about this sign are settled and one is not, and the
    /// difference matters more than it looks.
    ///
    /// **Settled, by construction:** a flexor and its extensor land on the
    /// same DoF with opposite polarity, so an antagonist pair is always a pair.
    ///
    /// **Settled, by measurement** (`tests/actuator_convention.rs`): flybody
    /// mirrors its own joint axes, so positive control moves the left and
    /// right legs the same anatomical way, to better than 0.01% on 21 of 24
    /// DoF pairs. One polarity per muscle is therefore correct and NO per-side
    /// flip is needed. Had it been otherwise, a uniform convention would have
    /// driven the left legs forward and the right legs backward, and the fly
    /// would have turned in circles while every unit test passed.
    ///
    /// **Not settled:** the absolute sense - whether a flexor's contraction is
    /// a positive or negative joint displacement in flybody's frame. Getting
    /// that backwards flips every leg identically, which a learning system can
    /// absorb and a gait metric will detect. It is a benign ambiguity because
    /// the dangerous one, the per-side kind, is ruled out above.
    pub polarity: f32,
}

/// The muscle vocabulary, from the connectome's own `Sub Class` annotations.
///
/// Names arrive as they are published (`Acc._ti_flexor`, `Tergopleural/Pleural_promotor`),
/// matched exactly rather than by substring: `Ti_flexor` and `Acc._ti_flexor`
/// are different muscles acting on the same joint, and a substring match would
/// conflate them.
pub fn leg_muscle(name: &str) -> Option<MuscleAction> {
    use LegDof::*;
    let (dof, polarity) = match name {
        // Thoraco-coxal group: these move the coxa against the thorax.
        "Sternal_anterior_rotator" => (CoxaTwist, 1.0),
        "Sternal_posterior_rotator" => (CoxaTwist, -1.0),
        "Pleural_remotor/abductor" => (CoxaAbduct, 1.0),
        "Sternal_adductor" => (CoxaAbduct, -1.0),
        "Tergopleural/Pleural_promotor" => (Coxa, 1.0),
        "Tergotr." => (Coxa, -1.0),
        // Trochanter-femur joint. flybody folds the trochanter into `femur`.
        "Tr_extensor" => (Femur, 1.0),
        "Tr_flexor" => (Femur, -1.0),
        "Acc._tr_flexor" => (Femur, -1.0),
        "Sternotrochanter" => (Femur, 1.0),
        "Fe_reductor" => (FemurTwist, -1.0),
        // Femur-tibia joint.
        "Ti_extensor" => (Tibia, 1.0),
        "Ti_flexor" => (Tibia, -1.0),
        "Acc._ti_flexor" => (Tibia, -1.0),
        // Tarsus. The long tendon muscle acts distally through its tendon;
        // `ltm1`/`ltm2` are named for where they ORIGINATE (tibia, femur), not
        // for what they move, which is the easy misreading here.
        "Ta_levator" => (Tarsus, 1.0),
        "Ta_depressor" => (Tarsus, -1.0),
        "ltm" => (Tarsus2, -1.0),
        "ltm1-tibia" => (Tarsus2, -1.0),
        "ltm2-femur" => (Tarsus2, -1.0),
        _ => return None,
    };
    Some(MuscleAction { dof, polarity })
}

/// One motor neuron's connection to one actuator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drive {
    /// Index into the connectome's neuron list.
    pub neuron: u32,
    /// Index into the actuator-name list this map was built against.
    pub actuator: usize,
    pub polarity: f32,
}

/// A motor neuron that could not be attached, and why. Listed rather than
/// dropped: a body that silently ignores a fifth of its motor neurons looks
/// exactly like one that is wired correctly.
#[derive(Clone, Debug, PartialEq)]
pub struct Unmapped {
    pub neuron: u32,
    pub sub_class: String,
    pub reason: String,
}

/// The built map.
#[derive(Clone, Debug, Default)]
pub struct MotorMap {
    pub drives: Vec<Drive>,
    pub unmapped: Vec<Unmapped>,
}

impl MotorMap {
    /// Motor neurons successfully attached to an actuator.
    pub fn mapped(&self) -> usize {
        self.drives.len()
    }

    /// Distinct actuators driven by at least one motor neuron.
    pub fn actuators_driven(&self) -> usize {
        let mut v: Vec<usize> = self.drives.iter().map(|d| d.actuator).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Every neuron driving a given actuator, with its polarity.
    pub fn neurons_for(&self, actuator: usize) -> Vec<(u32, f32)> {
        self.drives.iter().filter(|d| d.actuator == actuator).map(|d| (d.neuron, d.polarity)).collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} motor neurons attached to {} actuators, {} unmapped",
            self.mapped(),
            self.actuators_driven(),
            self.unmapped.len()
        )
    }
}

/// Parse a connectome `Sub Class` of the form `MN-LegNpT2-Ti_flexor`.
fn parse_sub_class(sub: &str) -> Option<(Segment, String)> {
    let rest = sub.strip_prefix("MN-")?;
    let (np, muscle) = rest.split_once('-')?;
    let seg = match np {
        "LegNpT1" => Segment::T1,
        "LegNpT2" => Segment::T2,
        "LegNpT3" => Segment::T3,
        _ => return None,
    };
    Some((seg, muscle.to_string()))
}

/// Build the map for every leg motor neuron in `c`, against a model's
/// actuator names.
///
/// `actuators` is taken as a plain name list rather than a loaded model so the
/// map is testable with no MuJoCo present, and so the same function serves a
/// model loaded from any source.
pub fn build(c: &Connectome, actuators: &[String]) -> MotorMap {
    let mut map = MotorMap::default();
    for (i, n) in c.neurons.iter().enumerate() {
        if n.super_class != "motor" {
            continue;
        }
        let neuron = i as u32;
        // Leg classes only. Wing, haltere, neck, abdomen and proboscis motor
        // neurons are real and annotated, but flybody's wing DoFs are driven
        // by a wingbeat generator rather than per-muscle, and the rest have no
        // actuator in this model: they are named as such rather than silently
        // skipped.
        if !matches!(n.class.as_str(), "fl" | "ml" | "hl") {
            map.unmapped.push(Unmapped {
                neuron,
                sub_class: n.sub_class.clone(),
                reason: format!("class {:?} is not a leg motor neuron", n.class),
            });
            continue;
        }
        let sub = n.sub_class.clone();
        let Some((seg, muscle)) = parse_sub_class(&sub) else {
            map.unmapped.push(Unmapped { neuron, sub_class: sub, reason: "Sub Class is not MN-LegNpT<n>-<muscle>".into() });
            continue;
        };
        let Some(action) = leg_muscle(&muscle) else {
            map.unmapped.push(Unmapped {
                neuron,
                sub_class: sub,
                reason: format!("no muscle named {muscle:?} in the vocabulary"),
            });
            continue;
        };
        let Some(side) = Side::parse(&n.soma_side) else {
            map.unmapped.push(Unmapped { neuron, sub_class: sub, reason: format!("soma side {:?} is neither left nor right", n.soma_side) });
            continue;
        };
        let want = action.dof.actuator(seg, side);
        let Some(actuator) = actuators.iter().position(|a| *a == want) else {
            map.unmapped.push(Unmapped { neuron, sub_class: sub, reason: format!("the model has no actuator {want:?}") });
            continue;
        };
        map.drives.push(Drive { neuron, actuator, polarity: action.polarity });
    }
    map
}
