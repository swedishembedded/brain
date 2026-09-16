// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Motor-map gates.
//!
//! The checkpoint-free tests pin the vocabulary and the antagonist structure.
//! The real-dataset test attaches MANC's own motor neurons to flybody's own
//! actuator names and reports what did not attach, which is the number that
//! matters: a body that silently ignores a fifth of its motor neurons looks
//! exactly like one that is wired correctly.

use flybody::{build, leg_muscle, LegDof, Segment, Side};

/// flybody's actuator names for all six legs, generated the same way the model
/// names them. Used where the real model is not present.
fn leg_actuators() -> Vec<String> {
    let dofs = [
        LegDof::CoxaAbduct,
        LegDof::CoxaTwist,
        LegDof::Coxa,
        LegDof::FemurTwist,
        LegDof::Femur,
        LegDof::Tibia,
        LegDof::Tarsus,
        LegDof::Tarsus2,
    ];
    let mut out = Vec::new();
    for seg in [Segment::T1, Segment::T2, Segment::T3] {
        for side in [Side::Left, Side::Right] {
            for d in dofs {
                out.push(d.actuator(seg, side));
            }
        }
    }
    out
}

#[test]
fn an_actuator_name_matches_what_the_model_calls_it() {
    assert_eq!(LegDof::Coxa.actuator(Segment::T1, Side::Left), "coxa_T1_left");
    assert_eq!(LegDof::CoxaAbduct.actuator(Segment::T3, Side::Right), "coxa_abduct_T3_right");
    assert_eq!(LegDof::Tarsus2.actuator(Segment::T2, Side::Left), "tarsus2_T2_left");
}

#[test]
fn antagonist_pairs_land_on_one_dof_with_opposite_polarity() {
    // This is the part of the polarity convention that IS guaranteed: the
    // absolute sign against flybody's joint axis is unverified, but a flexor
    // and its extensor must always oppose each other on the same joint.
    for (a, b) in [
        ("Ti_flexor", "Ti_extensor"),
        ("Tr_flexor", "Tr_extensor"),
        ("Ta_depressor", "Ta_levator"),
        ("Sternal_anterior_rotator", "Sternal_posterior_rotator"),
        ("Pleural_remotor/abductor", "Sternal_adductor"),
        ("Tergopleural/Pleural_promotor", "Tergotr."),
    ] {
        let (x, y) = (leg_muscle(a).unwrap(), leg_muscle(b).unwrap());
        assert_eq!(x.dof, y.dof, "{a} and {b} should act on one joint");
        assert_eq!(x.polarity, -y.polarity, "{a} and {b} should oppose each other");
    }
}

#[test]
fn an_accessory_flexor_agrees_with_its_principal() {
    // Accessory flexors are separate muscles on the same joint, pulling the
    // same way. Matching by substring would have merged them with the
    // principal; matching exactly keeps them distinct AND consistent.
    let ti = leg_muscle("Ti_flexor").unwrap();
    let acc = leg_muscle("Acc._ti_flexor").unwrap();
    assert_eq!((ti.dof, ti.polarity), (acc.dof, acc.polarity));
    assert!(leg_muscle("ti_flexor").is_none(), "matching must be exact, not case-folded or fuzzy");
    assert!(leg_muscle("Ti_flexor_typo").is_none());
}

#[test]
fn the_long_tendon_muscles_act_distally_not_where_they_originate() {
    // ltm1-tibia and ltm2-femur are named for their ORIGIN. They act through
    // the long tendon on the tarsus, and reading the name as the target is the
    // easy mistake this pins.
    for m in ["ltm", "ltm1-tibia", "ltm2-femur"] {
        assert_eq!(leg_muscle(m).unwrap().dof, LegDof::Tarsus2, "{m} acts on the tarsus");
    }
}

#[test]
fn a_neuron_whose_actuator_is_absent_is_reported_not_dropped() {
    let neurons = "\
Root ID,Flow,Super Class,Class,Sub Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence
1,efferent,motor,fl,MN-LegNpT1-Ti_flexor,ProLN_L,left,MN01,ACH,0.9
2,efferent,motor,fl,MN-LegNpT1-Ti_flexor,ProLN_R,right,MN02,ACH,0.9
3,efferent,motor,wm,MN-WTct-DLM_c-f,,left,MN03,ACH,0.9
4,efferent,motor,fl,MN-LegNpT1-front_leg,ProLN_L,left,MN04,ACH,0.9
";
    let edges = "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n1,2,X,3,\n";
    let c = connectome::load_readers("t", neurons.as_bytes(), edges.as_bytes()).unwrap();

    // Only the LEFT T1 actuators exist in this model.
    let actuators = vec!["tibia_T1_left".to_string()];
    let m = build(&c, &actuators);

    assert_eq!(m.mapped(), 1, "only the left tibia flexor can attach");
    assert_eq!(m.unmapped.len(), 3);
    let reasons: Vec<&str> = m.unmapped.iter().map(|u| u.reason.as_str()).collect();
    assert!(reasons.iter().any(|r| r.contains("tibia_T1_right")), "the missing actuator must be named: {reasons:?}");
    assert!(reasons.iter().any(|r| r.contains("not a leg motor neuron")), "{reasons:?}");
    assert!(reasons.iter().any(|r| r.contains("front_leg")), "a coarse label must say which one: {reasons:?}");
}

/// Attach MANC's real motor neurons to flybody's real actuator names.
#[test]
fn manc_motor_neurons_attach_to_the_fly() {
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_CONNECTOME_DIR unset");
        return;
    };
    let Ok((neurons, edges)) = connectome::find(std::path::PathBuf::from(root), "manc") else {
        brain_testutil::skip("no MANC export under BRAIN_CONNECTOME_DIR");
        return;
    };
    let c = connectome::load("manc", &neurons, &edges).expect("MANC loads");

    let actuators = leg_actuators();
    let m = build(&c, &actuators);
    eprintln!("{}", m.summary());

    let motor = c.population(|n| n.super_class == "motor").len();
    let leg = c.population(|n| n.super_class == "motor" && matches!(n.class.as_str(), "fl" | "ml" | "hl")).len();
    assert_eq!(motor, 737);
    assert_eq!(leg, 396);

    // Exact accounting: every leg motor neuron is attached, or carries a
    // coarse label that names no muscle, or has a Sub Class that is not a leg
    // neuropil at all. Nothing is dropped.
    let coarse = m.unmapped.iter().filter(|u| u.reason.contains("no muscle named")).count();
    let malformed = m.unmapped.iter().filter(|u| u.reason.contains("not MN-LegNpT")).count();
    assert_eq!(m.mapped() + coarse + malformed, leg, "leg motor neurons do not account for themselves");
    assert_eq!((m.mapped(), coarse, malformed), (330, 60, 6));

    // THE FINDING THIS TEST EXISTS TO PIN. MANC annotates the tarsus muscles
    // and the coxa promotor on the FRONT legs only, so four actuators have no
    // motor neuron at all and four more can only be pulled one way. That is a
    // property of the dataset's annotation, not of this map, and the walking
    // milestone has to handle it rather than discover it. Asserted by name so
    // that a dataset revision which fills the gap shows up here as a failing
    // expectation rather than passing unnoticed.
    let undriven: Vec<&str> = actuators
        .iter()
        .enumerate()
        .filter(|(i, _)| m.neurons_for(*i).is_empty())
        .map(|(_, n)| n.as_str())
        .collect();
    assert_eq!(
        undriven,
        ["tarsus_T2_left", "tarsus_T2_right", "tarsus_T3_left", "tarsus_T3_right"],
        "the set of actuators with no motor neuron changed"
    );

    let mut one_way: Vec<&str> = Vec::new();
    for (i, name) in actuators.iter().enumerate() {
        let pol: Vec<f32> = m.neurons_for(i).into_iter().map(|(_, p)| p).collect();
        if pol.is_empty() {
            continue;
        }
        if !(pol.iter().any(|&p| p > 0.0) && pol.iter().any(|&p| p < 0.0)) {
            one_way.push(name);
        }
    }
    one_way.sort_unstable();

    // Sixteen actuators can only be pulled one way, for THREE different
    // reasons, and conflating them would be the mistake:
    //
    //  * `tarsus2_*` (6) is CORRECT BIOLOGY. The long tendon muscle flexes the
    //    tarsus and there is no tarsal extensor in an insect leg - the tarsus
    //    returns elastically. A model that "fixed" this by inventing an
    //    antagonist would be wrong.
    //  * `femur_twist_*` (6) has only the femur reductor annotated. Whether
    //    that rotation has an antagonist at all, or is simply unannotated, is
    //    not something this map can settle.
    //  * `coxa_T2/T3_*` (4) is an ANNOTATION GAP: the promotor is annotated on
    //    the front legs only, so the middle and hind coxae have a retractor
    //    and nothing to oppose it.
    //
    // Pinned by name so a dataset revision that fills the gap shows up as a
    // failing expectation rather than passing unnoticed.
    let expected_one_way = [
        "coxa_T2_left", "coxa_T2_right", "coxa_T3_left", "coxa_T3_right",
        "femur_twist_T1_left", "femur_twist_T1_right",
        "femur_twist_T2_left", "femur_twist_T2_right",
        "femur_twist_T3_left", "femur_twist_T3_right",
        "tarsus2_T1_left", "tarsus2_T1_right",
        "tarsus2_T2_left", "tarsus2_T2_right",
        "tarsus2_T3_left", "tarsus2_T3_right",
    ];
    assert_eq!(one_way, expected_one_way, "the set of actuators that can only be pulled one way changed");

    eprintln!("  {coarse} coarse-labelled and {malformed} non-leg-neuropil MNs reported, not dropped");
    eprintln!("  {} of {} leg actuators driven; {} can only be pulled one way", m.actuators_driven(), actuators.len(), one_way.len());
}
