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

pub mod scene;

use connectome::Connectome;
pub use scene::{flight_model, flight_scene, world, world_extent, Arena, Flight, World};

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
    pub fn parse(s: &str) -> Option<Side> {
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

/// Which leg an actuator belongs to, from its flybody name.
///
/// The inverse of [`LegDof::actuator`], and it is built by matching against
/// that function's own output rather than by parsing the name - so a rename
/// cannot make the two disagree.
pub fn leg_of(actuator: &str) -> Option<(Segment, Side)> {
    use LegDof::*;
    for seg in [Segment::T1, Segment::T2, Segment::T3] {
        for side in [Side::Left, Side::Right] {
            for dof in [CoxaAbduct, CoxaTwist, Coxa, FemurTwist, Femur, Tibia, Tarsus, Tarsus2] {
                if dof.actuator(seg, side) == actuator {
                    return Some((seg, side));
                }
            }
        }
    }
    None
}

/// The six legs in a fixed order, with the alternating tripod each belongs to.
///
/// Tripod 0 is front-left, middle-right, hind-left; an insect's alternating
/// tripod gait swings one triangle while the other bears weight. This ordering
/// is the vocabulary every gait measurement in this workspace uses, so that
/// "leg 3" means the same leg everywhere.
pub const LEGS: [(Segment, Side, usize); 6] = [
    (Segment::T1, Side::Left, 0),
    (Segment::T2, Side::Right, 0),
    (Segment::T3, Side::Left, 0),
    (Segment::T1, Side::Right, 1),
    (Segment::T2, Side::Left, 1),
    (Segment::T3, Side::Right, 1),
];

/// A wing degree of freedom in flybody. Three per wing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WingDof {
    /// Stroke position: the fore-aft sweep that does the work.
    Yaw,
    /// Stroke-plane deviation.
    Roll,
    /// Feathering: the angle of attack, which flips at each stroke reversal.
    Pitch,
}

impl WingDof {
    fn stem(self) -> &'static str {
        match self {
            WingDof::Yaw => "wing_yaw",
            WingDof::Roll => "wing_roll",
            WingDof::Pitch => "wing_pitch",
        }
    }

    /// The flybody actuator name, e.g. `wing_yaw_left`.
    pub fn actuator(self, side: Side) -> String {
        format!("{}_{}", self.stem(), side.suffix())
    }
}

/// What a wing muscle does, which is NOT what a leg muscle does.
///
/// A leg muscle moves a joint and a motor neuron's firing maps onto that
/// joint's position. A wing muscle does not, and treating it as though it did
/// is the single easiest way to build a fly that cannot fly.
///
/// The power muscles - the dorsal longitudinal and dorsoventral groups - are
/// ASYNCHRONOUS: they are stretch-activated and contract many times per motor
/// spike, so their firing sets how much power goes into the thorax's resonant
/// oscillation and has no fixed phase relationship to the wingbeat at all.
/// Driving a wing joint from a DLM motor neuron's spike train would produce a
/// wingbeat at the motor neuron's rate, which is one or two orders of magnitude
/// too slow.
///
/// The steering muscles ARE phase-locked, one spike per beat or fewer, and
/// they act on the wing hinge's sclerites to bias the stroke rather than to
/// drive it. So they modulate amplitude and angle of attack, per wing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WingAction {
    /// Power to the thoracic oscillator: sets wingbeat amplitude, not phase.
    Power,
    /// Biases this wing's stroke amplitude, by the given polarity.
    Amplitude(f32),
    /// Biases this wing's angle of attack, by the given polarity.
    AngleOfAttack(f32),
}

/// The wing muscle vocabulary, from the connectome's own `Sub Class`.
///
/// Names arrive as published (`MN-WTct-DLM_c-f`, `MN-multi-i1`), so the muscle
/// is the part after the last `-`. Which muscle does what is anatomy and is
/// asserted with confidence; the SIGN of a steering muscle's effect is a
/// stated convention in the same sense as [`MuscleAction::polarity`], to be
/// pinned by measurement rather than assumed here. The one that is not a
/// convention is the basalare/axillary split: b1, b2 and b3 increase stroke
/// amplitude and the first and second axillary muscles reduce it, which is why
/// they carry opposite polarity and not the same one.
pub fn wing_muscle(name: &str) -> Option<WingAction> {
    // `MN-<neuropil>-<muscle>`, split at the FIRST hyphen after the prefix
    // rather than the last: the published power-muscle names carry a fibre
    // RANGE that contains a hyphen of its own (`DLM_c-f`, `DVM_1a-c`), so
    // taking the last field yields "f" and "c" and silently files the two
    // largest muscles in the thorax as unknown.
    let muscle = name.strip_prefix("MN-")?.split_once('-')?.1;
    // The power groups are published with the fibre range in the name, so they
    // are matched by prefix rather than exactly: `DLM_a,_b` and `DLM_c-f` are
    // two rows of the same muscle.
    if muscle.starts_with("DLM") || muscle.starts_with("DVM") {
        return Some(WingAction::Power);
    }
    Some(match muscle {
        // Basalares and the tergopleural group: stroke amplitude up.
        "b1" | "b2" | "b3" | "tp" | "tp1" | "tp2" | "ps1" | "ps2" => WingAction::Amplitude(1.0),
        // Third axillary and the anterior haltere group: also amplitude.
        "iii1" | "iii3" | "hg1" | "hg2" => WingAction::Amplitude(1.0),
        // First and second axillary: amplitude down.
        "i1" | "i2" => WingAction::Amplitude(-1.0),
        // Posterior haltere group: wing pitch, hence angle of attack.
        "hg3" | "hg4" => WingAction::AngleOfAttack(1.0),
        _ => return None,
    })
}

/// One wing motor neuron's role.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WingDrive {
    /// Index into the connectome's neuron list.
    pub neuron: u32,
    /// `None` for a power motor neuron, which acts on the whole thorax.
    pub side: Option<Side>,
    pub action: WingAction,
}

/// Every wing motor neuron in `c`, with what it does.
///
/// Power motor neurons are returned with `side: None` even though they have a
/// soma side: the two dorsal longitudinal groups drive one shared resonant
/// thorax, and attributing their output to one wing would invent a steering
/// signal that the animal does not have.
pub fn wing_map(c: &Connectome) -> Vec<WingDrive> {
    let mut out = Vec::new();
    for (i, n) in c.neurons.iter().enumerate() {
        if n.super_class != "motor" || n.class != "wm" {
            continue;
        }
        let Some(action) = wing_muscle(&n.sub_class) else { continue };
        let side = match action {
            WingAction::Power => None,
            _ => Side::parse(&n.soma_side),
        };
        if !matches!(action, WingAction::Power) && side.is_none() {
            continue;
        }
        out.push(WingDrive { neuron: i as u32, side, action });
    }
    out
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
    /// Claw adhesion, one entry per leg that has both an adhesion actuator
    /// and motor neurons to command it.
    ///
    /// Adhesion is a SEPARATE actuator in this body and nothing about driving
    /// the leg joints engages it. That is not a detail of the model: this
    /// animal's published walking depends on it, the adhesion actuators were
    /// added to the physics engine for exactly this reason, and a fly whose
    /// feet cannot grip has its legs slide out from under it however good the
    /// rhythm driving them is. Every walking result in this repository up to
    /// now was produced with all eight adhesion actuators at zero.
    pub adhesion: Vec<Adhesion>,
}

/// One leg's grip, and the motor neurons that decide it.
///
/// Driven by the tarsus muscles rather than by contact or by a phase
/// variable, because the connectome names them: `Ta_depressor` presses the
/// tarsus onto the substrate and `Ta_levator` lifts it off, ten and five
/// neurons per segment in MANC. Taking the grip from the animal's own
/// depressor population keeps the claw on the same footing as every other
/// muscle here - it is commanded by the cord, not by the simulator noticing
/// that a foot is down.
#[derive(Clone, Debug, PartialEq)]
pub struct Adhesion {
    pub segment: Segment,
    pub side: Side,
    /// Index into the actuator-name list.
    pub actuator: usize,
    /// Motor neurons that press the tarsus down.
    pub depressor: Vec<u32>,
    /// Motor neurons that lift it.
    pub levator: Vec<u32>,
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
/// `MN-LegNpT<n>-<muscle>` split into its segment and muscle name.
pub fn parse_sub_class(sub: &str) -> Option<(Segment, String)> {
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
/// The adhesion actuator for one leg, by flybody's own naming.
pub fn claw_actuator(seg: Segment, side: Side) -> String {
    format!("adhere_claw_{}_{}", seg.suffix(), side.suffix())
}

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

    // Claw adhesion, from the tarsus muscles of each leg.
    for (seg, side, _) in LEGS.iter() {
        let want = claw_actuator(*seg, *side);
        let Some(actuator) = actuators.iter().position(|a| *a == want) else {
            continue;
        };
        let mut adhesion = Adhesion { segment: *seg, side: *side, actuator, depressor: Vec::new(), levator: Vec::new() };
        for (i, n) in c.neurons.iter().enumerate() {
            if n.super_class != "motor" || !matches!(n.class.as_str(), "fl" | "ml" | "hl") {
                continue;
            }
            let Some((s, muscle)) = parse_sub_class(&n.sub_class) else { continue };
            if s != *seg || Side::parse(&n.soma_side) != Some(*side) {
                continue;
            }
            match muscle.as_str() {
                "Ta_depressor" => adhesion.depressor.push(i as u32),
                "Ta_levator" => adhesion.levator.push(i as u32),
                _ => {}
            }
        }
        if !adhesion.depressor.is_empty() {
            map.adhesion.push(adhesion);
        }
    }
    map
}
