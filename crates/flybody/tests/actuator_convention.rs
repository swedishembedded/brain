// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the model does when you push an actuator, measured rather than assumed.
//!
//! The motor map assigns each muscle a polarity, and applies that polarity
//! identically on the left and the right leg. That is only correct if flybody's
//! joint axes are mirrored in the MODEL, so the same control sign produces the
//! same anatomical motion on both sides. If instead the axes were shared and
//! the mirroring left to the caller, a uniform convention would drive the left
//! legs forward and the right legs backward, and the fly would turn in circles
//! while every unit test passed.
//!
//! That is the dangerous ambiguity, and this measures it away.

use flybody::{LegDof, Segment, Side};
use mujoco::{Data, Model, MuJoCo, StateSpec};

const DOFS: [LegDof; 8] = [
    LegDof::CoxaAbduct,
    LegDof::CoxaTwist,
    LegDof::Coxa,
    LegDof::FemurTwist,
    LegDof::Femur,
    LegDof::Tibia,
    LegDof::Tarsus,
    LegDof::Tarsus2,
];

/// Drive one actuator alone and report which generalized coordinate moved
/// most, and by how much.
///
/// Finding the joint by argmax rather than by reading `actuator_trnid` keeps
/// the binding free of another mjModel field: for a single actuator driven in
/// isolation from rest, the coordinate that moves most IS its joint.
fn response(m: &Model, d: &mut Data, actuator: usize) -> (usize, f64) {
    d.reset(m);
    d.forward(m);
    let base = d.get(m, StateSpec::QPOS);
    let mut ctrl = vec![0.0f64; m.nu()];
    ctrl[actuator] = 1.0;
    d.set(m, StateSpec::CTRL, &ctrl).unwrap();
    for _ in 0..50 {
        d.step(m);
    }
    let now = d.get(m, StateSpec::QPOS);
    base.iter()
        .zip(&now)
        .enumerate()
        .map(|(k, (a, b))| (k, b - a))
        .max_by(|x, y| x.1.abs().partial_cmp(&y.1.abs()).unwrap())
        .unwrap()
}

#[test]
fn positive_control_moves_left_and_right_legs_the_same_way() {
    let Ok(mj) = MuJoCo::load() else {
        brain_testutil::skip_unavailable("MuJoCo not loadable");
        return;
    };
    let Some(xml) = std::env::var_os("BRAIN_FLYBODY_XML").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLYBODY_XML unset");
        return;
    };
    let m = Model::from_xml(&mj, std::path::PathBuf::from(xml)).expect("the fly loads");
    let names: Vec<String> = m.actuator_names().into_iter().map(|n| n.unwrap_or_default()).collect();
    let mut d = Data::new(&m).unwrap();

    let mut moved: Vec<usize> = Vec::new();
    for seg in [Segment::T1, Segment::T2, Segment::T3] {
        for dof in DOFS {
            let li = names.iter().position(|n| *n == dof.actuator(seg, Side::Left)).expect("left actuator");
            let ri = names.iter().position(|n| *n == dof.actuator(seg, Side::Right)).expect("right actuator");
            let (lq, ld) = response(&m, &mut d, li);
            let (rq, rd) = response(&m, &mut d, ri);
            moved.push(lq);
            moved.push(rq);

            // THE claim: same sign on both sides. The model mirrors its own
            // axes, so the motor map applies one polarity per muscle and never
            // a per-side flip.
            assert!(
                ld > 0.0 && rd > 0.0,
                "{:?} {:?}: positive control gave {ld:+.6} left and {rd:+.6} right; a sign that differs by side means the map needs a per-side flip",
                seg,
                dof
            );
            // And the same magnitude, because the two legs are the same leg
            // mirrored. A large asymmetry would mean the model is not the
            // mirror this assumes.
            // And near-identical magnitude, because the two legs are the same
            // leg mirrored. NOT exact: flybody is built from a real scan
            // rather than a mirrored idealisation, so a few joints carry a
            // genuine left/right asymmetry. Measured across all 24 DoF pairs,
            // 21 agree to better than 0.01% and the worst (T3 coxa abduction)
            // is 0.22%; 1% is therefore a generous bound on "this is a
            // mirrored model" while still catching a model that is not.
            let rel = (ld - rd).abs() / ld.abs().max(rd.abs());
            assert!(rel < 0.01, "{seg:?} {dof:?}: left {ld:+.6} vs right {rd:+.6} differ by {:.2}%", rel * 100.0);

            // Distinct joints: an actuator that moved its neighbour's
            // coordinate most would mean the argmax is not finding the joint.
            assert_ne!(lq, rq, "{seg:?} {dof:?}: left and right moved the same coordinate");
        }
    }

    moved.sort_unstable();
    let distinct = {
        let mut v = moved.clone();
        v.dedup();
        v.len()
    };
    assert_eq!(distinct, 48, "expected 48 leg coordinates, each moved by exactly one actuator");
}
