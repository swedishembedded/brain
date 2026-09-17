// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for the closed loop.
//!
//! Two claims, and the second is the one that is easy to fake. A loop that
//! runs at rate but whose sensing changes nothing is an open loop wearing a
//! closed loop's clothes, and it would pass every other test in this repo.

use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

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
    let Ok((neurons, edges)) = connectome::find(std::path::PathBuf::from(root), "manc") else {
        brain_testutil::skip("no MANC export under BRAIN_CONNECTOME_DIR");
        return None;
    };
    let c = connectome::load("manc", &neurons, &edges).expect("MANC loads");
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
    let lif = fly::cord_lif();
    // Command must EXCEED threshold: with r = 1 and v_th = 1, a command of
    // exactly 1.0 makes v approach threshold asymptotically and never reach
    // it. That is how an earlier version of this test measured a silent cord.
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    Fly::new(gpu, &r.c, model, lif, Wiring::default(), Timing::default(), Coupling::default()).expect("the loop composes")
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

/// Time one repeat of `f`, taking the FASTEST of several.
///
/// The minimum rather than the mean, and that is not cherry-picking: every
/// source of error in a wall-clock measurement on a shared machine is
/// one-sided. Contention, migration between a performance core and an
/// efficiency one, and a scheduler slice lost to another process can only
/// make a run slower, never faster. The fastest repeat is therefore the
/// closest estimate of what the work actually costs, and it is the only
/// summary that does not drift with whatever else the box is doing - which
/// matters here because this test is habitually run while a parameter search
/// is using every core.
fn fastest(repeats: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let t0 = std::time::Instant::now();
        f();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
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
    let mut spikes = 0u64;
    let closed = fastest(3, || {
        spikes = 0;
        for _ in 0..ticks {
            spikes += f.step().unwrap().total_spikes as u64;
        }
    });

    // The same number of physics steps, with no nervous system at all.
    let model = Model::from_xml(&r.mj, &r.xml).unwrap();
    let mut d = mujoco::Data::new(&model).unwrap();
    for _ in 0..200 {
        d.step(&model);
    }
    let body = fastest(3, || {
        for _ in 0..(ticks * 20) {
            d.step(&model);
        }
    });

    // What the CORD costs on its own, in this process, on this device: the
    // same network, the same number of ticks, with no body attached.
    //
    // Measured rather than assumed, because it is a property of the machine
    // and spans an order of magnitude across the ones this runs on - 0.6 ms
    // per tick on 22 CPU threads against 6 ms on an integrated GPU, whose
    // memory bandwidth is a seventh of the CPU's for a kernel that is pure
    // streaming. A fixed millisecond ceiling here is therefore a statement
    // about the hardware, and this test had one: three milliseconds, chosen
    // against a profile of 0.81 ms that was itself measuring a submission
    // backlog rather than a tick.
    let w = Wiring::default();
    let mut bare = neuro::SpikingNet::new(
        gpu_core::testgpu::dev(&neuro::KERNELS),
        &r.c.network(w.weight_scale, w.size_limit, w.min_synapses),
        fly::cord_lif(),
    )
    .expect("the cord builds");
    let mut spike = vec![0.0f32; r.c.neurons.len()];
    use neuro::DynamicalSystem;
    for _ in 0..20 {
        bare.step();
        bare.read(neuro::Port::Spike, &mut spike).unwrap();
    }
    let cord = fastest(3, || {
        for _ in 0..ticks {
            bare.step();
            bare.read(neuro::Port::Spike, &mut spike).unwrap();
        }
    });

    let per = |seconds: f64| 1000.0 * seconds / ticks as f64;
    let (closed_ms, body_ms, cord_ms) = (per(closed), per(body), per(cord));
    let overhead_ms = closed_ms - body_ms;
    eprintln!(
        "closed loop {closed_ms:.2} ms/tick, body alone {body_ms:.2} ms, cord alone {cord_ms:.2} ms, \
         neural overhead {overhead_ms:.2} ms, {spikes} spikes"
    );
    assert!(spikes > 0, "the cord was silent, so this measures an idle loop");

    // THE CLAIM, and it is device-independent because both sides of it are
    // measured on the device in front of it: closing the loop costs the body
    // plus the cord and NOTHING ELSE. What that rules out is what a
    // regression here would actually be - a second synchronisation per tick,
    // an allocation in the step, a readback nobody needed - each of which
    // shows up as overhead the cord alone does not account for.
    //
    // It caught one: the proprioceptors were rebuilding an actuator name with
    // `format!` and linear-searching for it on every sensor on every tick,
    // 150,000 string allocations a second, and the loop cost five
    // milliseconds a tick more than the body and the cord together.
    //
    // The margin is a millisecond plus half the cord's own cost, because the
    // two measurements are taken under different cache pressure and the
    // closed loop interleaves the body between the submit and the readback.
    let budget = cord_ms * 1.5 + 1.0;
    assert!(
        overhead_ms < budget,
        "closing the loop added {overhead_ms:.2} ms per control tick against a cord that costs \
         {cord_ms:.2} ms on its own; the loop is paying for something besides the nervous system"
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

#[test]
fn severing_a_muscle_changes_what_the_body_does_and_an_intact_one_does_not() {
    let Some(r) = rig() else { return };
    let mut f = build(&r);
    let cmd = vec![COMMAND; f.descending_count()];

    let trajectory = |f: &mut Fly| -> Vec<f64> {
        f.reset();
        f.set_descending(&cmd).unwrap();
        for _ in 0..60 {
            f.step().unwrap();
        }
        f.qpos()
    };

    let intact = trajectory(&mut f);

    // THE CONTROL, and it comes first. Setting every muscle to the strength it
    // already has must change nothing at all - otherwise "the lesion changed
    // the trajectory" is equally satisfied by a loop that is simply not
    // reproducible, and the real assertion below would prove nothing.
    let n = f.lesion_matching("_T1_left", 1.0).expect("the pattern matches");
    assert!(n > 0, "the pattern matched no actuator");
    let unchanged = trajectory(&mut f);
    assert_eq!(intact, unchanged, "a no-op lesion changed the trajectory; the loop is not reproducible");

    // Now sever them.
    f.lesion_matching("_T1_left", 0.0).expect("the pattern matches");
    let severed = trajectory(&mut f);
    assert_ne!(intact, severed, "severing {n} muscles changed nothing; the lesion is not reaching the body");

    // And it must be restorable, or a perturbation experiment could never
    // measure the same animal twice.
    f.lesion_matching("_T1_left", 1.0).unwrap();
    assert_eq!(intact, trajectory(&mut f), "restoring the muscles did not restore the behaviour");
}

#[test]
fn a_lesion_that_matches_nothing_is_an_error_rather_than_a_silent_no_op() {
    let Some(r) = rig() else { return };
    let mut f = build(&r);
    let err = f.lesion_matching("no_such_actuator", 0.0).unwrap_err();
    assert!(err.contains("no_such_actuator"), "the error must name the pattern: {err}");
    assert!(f.set_muscle_strength(usize::MAX, 0.0).is_err(), "an out-of-range actuator must be refused");
}
