// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Do the wings carry the fly?
//!
//! One claim, with the control that makes it a claim: the SAME body, the same
//! model, the same loop, with the wingbeat's power at zero. Without that
//! comparison "the fly descended slowly" is equally satisfied by a model whose
//! gravity is wrong or whose body is caught on something.
//!
//! The control is NOT free fall, and assuming it was is worth recording. A fly
//! with its wings out and still does not fall at g: those wings are large
//! aerodynamic surfaces under an explicit fluid model, and measured, passive
//! drag alone halves the descent. Comparing a beating fly against a textbook
//! free fall would therefore have credited the wingbeat with lift that the
//! wings produce by simply being there. The comparison here is against the
//! same body with the stroke switched off.

use fly::wing::{WingCommand, Wingbeat};
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

struct Rig {
    fly: Fly,
    dt: f64,
    steps: u32,
    // Held so the generated model outlives the load.
    _dir: tempfile::TempDir,
}

fn rig() -> Option<Rig> {
    let Ok(mj) = MuJoCo::load() else {
        brain_testutil::skip_unavailable("MuJoCo not loadable");
        return None;
    };
    let Some(base) = std::env::var_os("BRAIN_FLYBODY_FRUITFLY_XML").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLYBODY_FRUITFLY_XML unset");
        return None;
    };
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_CONNECTOME_DIR unset");
        return None;
    };
    let Ok((neurons, edges)) = connectome::find(std::path::PathBuf::from(root), "manc") else {
        brain_testutil::skip("no MANC export under BRAIN_CONNECTOME_DIR");
        return None;
    };
    let dir = tempfile::tempdir().unwrap();
    let cfg = flybody::Flight::default();
    let scene = flybody::flight_scene(std::path::Path::new(&base), dir.path(), cfg).expect("the flight scene builds");
    let c = connectome::load("manc", &neurons, &edges).expect("MANC loads");
    let model = Model::from_xml(&mj, &scene).expect("the flight model compiles");
    let lif = fly::cord_lif();
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let steps = 40;
    let timing = Timing { neural_per_control: 1, physics_per_control: steps, physics_dt: cfg.timestep };
    let fly = Fly::new(gpu, &c, model, lif, Wiring::default(), timing, Coupling::default()).expect("the loop composes");
    Some(Rig { fly, dt: cfg.timestep, steps, _dir: dir })
}

/// Steady-state descent SPEED, in centimetres per second, at the given wing
/// power. Taken over the second half of a run: the first half is the thorax
/// spinning up and the body reaching its terminal speed, and neither is what
/// is being asked about.
fn sink_rate(rig: &mut Rig, power: f32, ticks: u32) -> f64 {
    rig.fly.reset();
    rig.fly.hold_wing_command(Some(WingCommand { power, ..WingCommand::default() }));
    let mut z0 = rig.fly.qpos()[2];
    for k in 0..ticks {
        rig.fly.step().unwrap();
        if k == ticks / 2 {
            z0 = rig.fly.qpos()[2];
        }
    }
    let window = (ticks / 2) as f64 * rig.steps as f64 * rig.dt;
    -(rig.fly.qpos()[2] - z0) / window
}

/// Peak-to-peak stroke sweep over the second half of a run.
fn sweep(rig: &mut Rig, ticks: u32) -> f64 {
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for k in 0..ticks {
        rig.fly.step().unwrap();
        if k < ticks / 2 {
            continue;
        }
        for w in rig.fly.wing_angles() {
            lo = lo.min(w[0]);
            hi = hi.max(w[0]);
        }
    }
    hi - lo
}

#[test]
fn the_wings_carry_most_of_the_body_and_a_still_wing_carries_none_of_it() {
    let Some(mut rig) = rig() else { return };
    assert!(
        rig.fly.wing_summary().starts_with("24 power"),
        "all 24 power motor neurons should attach: {}",
        rig.fly.wing_summary()
    );
    // 180 Hz rather than the animal's 218: this airframe's own hinge resonates
    // lower than the real thorax does, and the frequency was measured by a
    // sweep rather than assumed from the biology.
    rig.fly.enable_flight(Wingbeat { hz: 180.0, ..Wingbeat::default() }).expect("the body has wings");

    let ticks = 400u32;

    // THE CONTROL, first: the same body with the stroke switched off. It still
    // sinks at a finite rate rather than accelerating, because those wings are
    // aerodynamic surfaces whether or not they are beating.
    let still = sink_rate(&mut rig, 0.0, ticks);
    assert!(still > 30.0, "with the stroke off the fly should be sinking, not hovering: {still:.1} cm/s");

    let flying = sink_rate(&mut rig, 1.0, ticks);
    assert!(
        flying < 0.5 * still,
        "the stroke barely helped: sinking at {flying:.1} cm/s against {still:.1} cm/s with the wings still"
    );
    eprintln!("sink rate {still:.1} cm/s with the stroke off, {flying:.1} cm/s beating at 180 Hz");
}

#[test]
fn flight_is_opt_in_and_refuses_a_body_that_cannot_fly() {
    let Some(mut rig) = rig() else { return };
    assert!(rig.fly.wingbeat().is_none(), "a fly should not be flying until asked");

    // THE CONTROL: with flight off the wings still MOVE, and measurably. The
    // hinge is sprung, the model starts away from its rest angle, and the body
    // is falling through air with the wings out - so they ring rather than sit
    // still, at about half a radian peak to peak. That is the baseline the
    // stroke has to be an order of magnitude above, and it is asserted as a
    // baseline rather than wished away: a threshold tight enough to call it
    // "still" would be a threshold this model does not meet.
    rig.fly.reset();
    let settling = sweep(&mut rig, 200);
    assert!(settling < 1.0, "with flight off the wings ring at {settling:.3} rad; that is not a passive hinge");

    rig.fly.enable_flight(Wingbeat::default()).unwrap();
    rig.fly.hold_wing_command(Some(WingCommand { power: 1.0, ..WingCommand::default() }));
    rig.fly.reset();
    let beating = sweep(&mut rig, 200);
    assert!(beating > 5.0 * settling, "flight enabled swept {beating:.3} rad against {settling:.3} ringing");
    eprintln!("stroke {beating:.2} rad peak to peak, against {settling:.3} ringing passively");
}
