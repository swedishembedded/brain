// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for the closed loop.
//!
//! Two claims, and the second is the one that is easy to fake. A loop that
//! runs at rate but whose sensing changes nothing is an open loop wearing a
//! closed loop's clothes, and it would pass every other test in this repo.

use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

/// Descending command strength. Above threshold, see `build`.
const COMMAND: f32 = 2.0;

struct Rig {
    c: connectome::Connectome,
    mj: std::sync::Arc<MuJoCo>,
    xml: std::path::PathBuf,
}

/// Everything this test needs, or a clean skip naming what was missing.
fn rig() -> Option<Rig> {
    let Ok(mj) = MuJoCo::load() else {
        brain_testutil::skip_unavailable("MuJoCo not loadable");
        return None;
    };
    let Some(xml) = std::env::var_os("BRAIN_FLYBODY_XML").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLYBODY_XML unset");
        return None;
    };
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_CONNECTOME_DIR unset");
        return None;
    };
    let dir = std::path::PathBuf::from(root).join("manc-codex");
    if !dir.join("neurons.csv.gz").is_file() {
        brain_testutil::skip("manc-codex not present");
        return None;
    }
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz"))
        .expect("MANC loads");
    Some(Rig { c, mj, xml: std::path::PathBuf::from(xml) })
}

fn build(r: &Rig) -> Fly {
    let model = Model::from_xml(&r.mj, &r.xml).expect("the fly loads");
    // The weight scale is the one free number in this loop, and it was chosen
    // by sweeping it rather than by reasoning about it. Measured, driving the
    // descending neurons at 2.0:
    //
    //   1e-3 -> 1.1% of the cord active but ZERO motor spikes
    //   1e-2 -> 1.6% active, 15 motor spikes per tick
    //   3e-2 -> 2.5% active, 49 motor spikes per tick
    //   1e-1 -> 3.5% active, 68 motor spikes per tick
    //
    // 3e-2 sits in the sparse regime a nerve cord should occupy and actually
    // reaches the muscles. At 1e-3 the network is alive and the body is not
    // driven at all, which is the failure mode worth naming: "the cord is
    // active" is not the same claim as "the cord is driving something".
    //
    // Sign comes from the connectome (52.1% excitatory, 47.7% inhibitory) and
    // applying it is not optional - an all-excitatory network has no brake.
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    // Command must EXCEED threshold: with r = 1 and v_th = 1, a command of
    // exactly 1.0 makes v approach threshold asymptotically and never reach
    // it. That is how an earlier version of this test measured a silent cord.
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    Fly::new(gpu, &r.c, model, lif, 3e-2, None, Timing::default(), Coupling::default()).expect("the loop composes")
}

#[test]
fn the_loop_composes_and_every_channel_is_populated() {
    let Some(r) = rig() else { return };
    let f = build(&r);
    eprintln!(
        "descending {}, proprioceptors {}, {}",
        f.descending_count(),
        f.proprioceptor_count(),
        f.motor_map().summary()
    );
    assert_eq!(f.descending_count(), 1328, "the command channel");
    // 304 of MANC's 1,201 proprioceptor-class neurons resolve to a LEG. The
    // rest watch the wings, halteres, neck and abdomen, which this body has no
    // proprioceptive channel for; they are excluded rather than attached to a
    // leg they do not belong to.
    assert_eq!(f.proprioceptor_count(), 304, "the sensory channel");
    assert_eq!(f.motor_map().mapped(), 330, "the motor channel");
}

#[test]
fn closing_the_loop_costs_little_over_the_body_alone() {
    let Some(r) = rig() else { return };
    let mut f = build(&r);
    let cmd = vec![COMMAND; f.descending_count()];
    f.set_descending(&cmd).unwrap();

    // Warm up: the first ticks pay for pipeline creation and page faults.
    for _ in 0..20 {
        f.step().unwrap();
    }
    let ticks = 200;
    let t0 = std::time::Instant::now();
    let mut spikes = 0u64;
    for _ in 0..ticks {
        spikes += f.step().unwrap().total_spikes as u64;
    }
    let hz = ticks as f64 / t0.elapsed().as_secs_f64();

    // The same number of physics steps, with no nervous system at all.
    let model = Model::from_xml(&r.mj, &r.xml).unwrap();
    let mut d = mujoco::Data::new(&model).unwrap();
    for _ in 0..200 {
        d.step(&model);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..(ticks * 20) {
        d.step(&model);
    }
    let body_hz = ticks as f64 / t0.elapsed().as_secs_f64();

    let closed_ms = 1000.0 / hz;
    let body_ms = 1000.0 / body_hz;
    let overhead_ms = closed_ms - body_ms;
    eprintln!(
        "closed loop {hz:.0} Hz ({closed_ms:.2} ms/tick), body alone {body_hz:.0} Hz ({body_ms:.2} ms/tick), \
         neural overhead {overhead_ms:.2} ms, {spikes} spikes"
    );
    assert!(spikes > 0, "the cord was silent, so this measures an idle loop");

    // The claim is about the ABSOLUTE cost of the nervous system per control
    // tick, not about a ratio. A ratio moves with the body: on the floorless
    // model the body alone ran at 189 Hz and the same neural work was 2% of
    // the tick, while with contact physics the body drops to ~131 Hz and the
    // identical work is a larger share of a larger number. Neither tells you
    // anything about brain. The overhead does, and it is one GPU round trip
    // per neural tick (profiled at 0.81 ms) plus the drive write.
    //
    // 3 ms is a generous ceiling on that, chosen well above the profile rather
    // than fitted to a run: it catches a regression that adds a second sync
    // without failing on ordinary contention.
    assert!(
        overhead_ms < 3.0,
        "closing the loop added {overhead_ms:.2} ms per control tick; the profiled cost is about 1 ms"
    );

    // flybody runs slower than real time on this hardware regardless, so the
    // 500 Hz rate its own walking tasks use is a property of the machine
    // rather than of this loop. Reported, never asserted.
    eprintln!("  (500 Hz would need {:.2} ms/tick; the body alone needs {body_ms:.2} ms)", 2.0);
}

#[test]
fn lesioning_proprioception_changes_the_trajectory() {
    let Some(r) = rig() else { return };

    // Same command, same initial state, same everything except the sensory
    // channel. Two fresh flies rather than one reset, so no state can leak.
    let run = |on: bool| {
        let mut f = build(&r);
        f.set_proprioception(on);
        let cmd = vec![COMMAND; f.descending_count()];
        f.set_descending(&cmd).unwrap();
        let mut spikes = 0u64;
        for _ in 0..150 {
            f.step().unwrap();
            spikes += f.proprioceptor_spikes() as u64;
        }
        (f.qpos(), spikes)
    };

    let (intact, proprio_spikes) = run(true);
    let (lesioned, _) = run(false);

    // The assertion that would have caught this loop's first real defect.
    // "Connected" is not "transmitting": with the original sensory gain the
    // proprioceptors sat below threshold, injected current every tick, never
    // fired, and changed nothing - while the cord spiked, the body moved and
    // every other check passed. Assert the channel actually CARRIES something
    // before asserting that removing it matters.
    assert!(proprio_spikes > 0, "no proprioceptor ever fired, so the sensory channel is connected but silent");
    assert_eq!(intact.len(), lesioned.len());

    let worst = intact.iter().zip(&lesioned).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let moved = intact.iter().zip(&lesioned).filter(|(a, b)| (*a - *b).abs() > 1e-9).count();
    eprintln!("lesion changed {moved} of {} coordinates, worst {worst:.6} ({proprio_spikes} proprioceptor spikes)", intact.len());

    // THE claim: the feedback is load-bearing. If sensing changed nothing, the
    // two trajectories would be bit-identical and this loop would be open.
    assert!(worst > 1e-6, "disabling proprioception changed nothing; the loop is not closed");

    // And the intact run must be sane, not merely different: a diverged
    // simulation also differs from everything.
    assert!(intact.iter().all(|v| v.is_finite()), "the intact run diverged");
    assert!(lesioned.iter().all(|v| v.is_finite()), "the lesioned run diverged");
}
