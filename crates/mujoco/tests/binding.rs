// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Binding gates.
//!
//! Every one of these skips cleanly when MuJoCo is not installed, which is the
//! property that lets this crate be an ordinary default member: absence is a
//! missing capability, not a broken build.

use mujoco::{Data, Model, MuJoCo, ObjType, StateSpec};

fn lib() -> Option<std::sync::Arc<MuJoCo>> {
    match MuJoCo::load() {
        Ok(m) => Some(m),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("MuJoCo not loadable: {e}"));
            None
        }
    }
}

/// The fly, if this machine has it. `$BRAIN_FLYBODY_XML` names the MJCF.
fn fly(mj: &std::sync::Arc<MuJoCo>) -> Option<Model> {
    let Some(path) = std::env::var_os("BRAIN_FLYBODY_XML").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLYBODY_XML unset (the flybody MJCF is not in this repo)");
        return None;
    };
    match Model::from_xml(mj, std::path::PathBuf::from(path)) {
        Ok(m) => Some(m),
        Err(e) => panic!("BRAIN_FLYBODY_XML is set but the model did not load: {e}"),
    }
}

#[test]
fn the_library_loads_and_reports_a_3x_version() {
    let Some(mj) = lib() else { return };
    let v = mj.version();
    assert!((3_000_000..4_000_000).contains(&v), "unexpected MuJoCo version {v}");
    eprintln!("MuJoCo {v}");
}

#[test]
fn a_trivial_model_round_trips_its_state() {
    let Some(mj) = lib() else { return };
    // A one-hinge pendulum, written inline: this test must not depend on any
    // file, so the binding itself is covered even where the fly is not.
    let xml = r#"<mujoco>
      <worldbody>
        <body name="arm" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom name="rod" type="capsule" size="0.02 0.2" pos="0 0 -0.2"/>
        </body>
      </worldbody>
      <actuator><motor name="drive" joint="hinge"/></actuator>
    </mujoco>"#;
    let dir = std::env::temp_dir().join(format!("brain-mujoco-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pendulum.xml");
    std::fs::write(&path, xml).unwrap();

    let m = Model::from_xml(&mj, &path).expect("pendulum compiles");
    assert_eq!((m.nq(), m.nv(), m.nu()), (1, 1, 1));
    assert_eq!(m.id_of(ObjType::Joint, "hinge"), Some(0));
    assert_eq!(m.id_of(ObjType::Actuator, "drive"), Some(0));
    assert_eq!(m.id_of(ObjType::Actuator, "nonesuch"), None);
    assert_eq!(m.actuator_names(), vec![Some("drive".to_string())]);

    let mut d = Data::new(&m).unwrap();
    d.set(&m, StateSpec::QPOS, &[0.3]).unwrap();
    d.forward(&m);
    assert!((d.get(&m, StateSpec::QPOS)[0] - 0.3).abs() < 1e-12, "qpos should round-trip exactly");

    // A short slice would be an out-of-bounds read inside MuJoCo, so it is
    // refused rather than padded.
    assert!(d.set(&m, StateSpec::QPOS, &[]).is_err());
    assert!(d.set(&m, StateSpec::QPOS, &[0.1, 0.2]).is_err());

    // Gravity does work on a horizontal pendulum: it must move, and time must
    // advance. A binding that stepped nothing would pass neither.
    let t0 = d.time(&m);
    for _ in 0..200 {
        d.step(&m);
    }
    assert!(d.time(&m) > t0, "time did not advance");
    assert!((d.get(&m, StateSpec::QPOS)[0] - 0.3).abs() > 1e-3, "the pendulum did not swing");
    assert!(d.get(&m, StateSpec::QPOS)[0].is_finite(), "the simulation diverged");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_fly_loads_and_its_actuators_are_the_ones_the_motor_map_needs() {
    let Some(mj) = lib() else { return };
    let Some(m) = fly(&mj) else { return };
    eprintln!("flybody: nq={} nv={} nu={}", m.nq(), m.nv(), m.nu());
    assert!(m.nu() > 50, "expected a fly-sized actuator count, got {}", m.nu());

    let names: Vec<String> = m.actuator_names().into_iter().flatten().collect();
    // The chain the motor map attaches MANC's named muscles to.
    for joint in ["coxa_T1_left", "femur_T1_left", "tibia_T1_left", "tarsus_T1_left"] {
        assert!(names.iter().any(|n| n == joint), "no actuator named {joint}; the motor map cannot attach");
    }
    let legs = names.iter().filter(|n| n.contains("_T1_") || n.contains("_T2_") || n.contains("_T3_")).count();
    assert!(legs >= 36, "expected at least 6 DoF on each of 6 legs, found {legs}");
    let wings = names.iter().filter(|n| n.starts_with("wing_")).count();
    assert!(wings >= 6, "expected 3 wing DoF per side, found {wings}");
}

#[test]
fn the_fly_integrates_stably_under_gravity() {
    let Some(mj) = lib() else { return };
    let Some(m) = fly(&mj) else { return };
    let mut d = Data::new(&m).unwrap();
    d.reset(&m);

    // No control input, whatever the model's own worldbody provides. The claim
    // is narrow and is the one the body milestone needs: the model integrates
    // stably. A model that explodes here cannot be driven by anything.
    //
    // 1000 steps is 0.1 s at flybody's own 1e-4 timestep, not a second. The
    // fly is in free fall for all of it, so `max |qvel|` in the tens is
    // expected and the assertion below is a divergence check, not a stillness
    // one.
    for _ in 0..1000 {
        d.step(&m);
    }
    let qpos = d.get(&m, StateSpec::QPOS);
    let qvel = d.get(&m, StateSpec::QVEL);
    assert!(qpos.iter().all(|v| v.is_finite()), "position diverged to a non-finite value");
    assert!(qvel.iter().all(|v| v.is_finite()), "velocity diverged to a non-finite value");
    let worst = qvel.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    eprintln!("after {:.3}s: max |qvel| = {worst:.3}", d.time(&m));
    assert!(worst < 1e4, "the fly is being flung: max |qvel| = {worst}");
}
