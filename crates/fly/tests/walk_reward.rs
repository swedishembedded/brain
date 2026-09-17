// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reward has to be earnable ONLY by walking.
//!
//! This crate has already shipped a reward that a corpse could collect. Under
//! DeepMimic imitation, measured: a fly with every muscle severed scored 98.7%
//! of the best driven score, and driving the animal harder made the score
//! monotonically WORSE - correct behaviour for an imitation reward applied to
//! a body that cannot track the reference, and useless as an objective. Every
//! search run under it was optimising immobility.
//!
//! So the paralysed control is now a GATE rather than an experiment somebody
//! remembered to run, and it comes with the two other ways a locomotion reward
//! is usually cheated: scoring a body that is dragged without stepping, and
//! scoring legs that oscillate beautifully while the animal goes nowhere.

use fly::gait::{scripted_tripod, Gait, Trace};
use fly::learn::{episode, Condition, Lcg, Objective, RewardConfig};
use fly::{Coupling, Fly, Timing, Wiring};
use mujoco::{Model, MuJoCo};

fn rig() -> Option<Fly> {
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
    let model = Model::from_xml(&mj, std::path::PathBuf::from(xml)).expect("the fly loads");
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    Some(
        Fly::new(gpu, &c, model, fly::cord_lif(), Wiring::default(), Timing::default(), Coupling::default())
            .expect("the loop composes"),
    )
}

fn run(fly: &mut Fly, command: f32) -> fly::learn::Episode {
    let cfg = RewardConfig { objective: Objective::walk(), ticks: 1000, command, ..RewardConfig::default() };
    episode(fly, cfg, Condition::Frozen, &mut Lcg::new(7)).expect("an episode runs")
}

/// The gate the imitation reward failed.
#[test]
fn a_paralysed_fly_scores_nothing() {
    let Some(mut fly) = rig() else { return };
    let driven = run(&mut fly, 1.0);

    for a in 0..fly.actuator_count() {
        fly.set_muscle_strength(a, 0.0).expect("an actuator exists");
    }
    let corpse = run(&mut fly, 1.0);

    // Not "less than the driven one" - a corpse has to score essentially
    // NOTHING, because the only thing that moves it is gravity settling.
    assert!(
        corpse.score().abs() < 0.01,
        "a severed body scored {:.4} (driven {:.4}); the reward is not about the animal",
        corpse.score(),
        driven.score()
    );
    assert!(corpse.spikes > 0, "the cord should still be firing; this is a PERIPHERAL lesion");
}

/// Travel without a rhythm is a lunge, and a lunge is not a gait.
///
/// Checked on the score FUNCTION with a synthesised trace rather than on the
/// animal, because a fly that lunges on demand is exactly the thing nothing in
/// this repo can produce yet - and the property being asserted is the reward's,
/// not the fly's.
#[test]
fn travel_without_a_rhythm_scores_nothing_and_so_does_rhythm_without_travel() {
    let dt = 0.002;
    let walking = fly::analyse_gait(&scripted_tripod(12.0, dt, 1000)).expect("a tripod is analysable");
    assert!(walking.score() > 0.5, "a perfect tripod should score well, got {:.3}", walking.score());

    // Legs that do not move: the body is being dragged.
    let mut still = Trace::new(dt);
    for _ in 0..1000 {
        still.push([0.0; 6]);
    }
    let dragged = fly::analyse_gait(&still).map(|g| g.score()).unwrap_or(0.0);
    assert!(dragged < 0.01, "a motionless leg trace scored {dragged:.4}");

    // And the product: a perfect gait that travels nowhere is worth nothing,
    // however good the rhythm.
    let ep = |net: f64, gait: Option<Gait>| fly::learn::Episode {
        net,
        gait,
        objective: Some(Objective::walk()),
        ..Default::default()
    };
    assert_eq!(ep(0.0, Some(walking)).score(), 0.0, "running on the spot is not walking");
    assert!(ep(1.0, None).score().abs() < 1e-12, "travel with no analysable gait is not walking");
    assert!(ep(1.0, Some(walking)).score() > 0.5, "a tripod that travels should score");
}

/// Dropping below the threshold ends the episode rather than diluting it.
///
/// Tested by moving the THRESHOLD rather than the fly, because an episode
/// begins with `Fly::reset` and anything a caller does to the pose beforehand
/// is discarded - so a fly cannot be dropped from a height into an episode.
/// The pair is the point: a threshold under the settling depth terminates, one
/// above it does not, and without the second row "it terminated" would also be
/// satisfied by an episode that always terminates.
#[test]
fn dropping_below_the_fall_threshold_ends_the_episode() {
    let Some(mut fly) = rig() else { return };
    let run = |fly: &mut Fly, terminal_fall| {
        let cfg = RewardConfig {
            objective: Objective::Walk { terminal_fall, terminal_tip: 1.0 },
            ticks: 1000,
            command: 1.0,
            ..RewardConfig::default()
        };
        episode(fly, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs")
    };
    // A fly settles onto its legs by a fraction of a body length; a threshold
    // below that is tripped, one at half a body length is not.
    let tight = run(&mut fly, 0.0005);
    let loose = run(&mut fly, 0.125);
    assert!(tight.terminated && tight.ticks < 1000, "a threshold under the settling depth did not terminate");
    assert!(!loose.terminated, "a normal stance tripped a half-body-length fall threshold");
}

/// An episode too short to hold a rhythm is refused, not scored zero.
#[test]
fn a_walking_episode_too_short_to_score_is_an_error() {
    let Some(mut fly) = rig() else { return };
    let cfg = RewardConfig { objective: Objective::walk(), ticks: 100, ..RewardConfig::default() };
    let e = episode(&mut fly, cfg, Condition::Frozen, &mut Lcg::new(1)).expect_err("too short to score");
    assert!(e.contains("would score zero"), "{e}");
}

/// Two identical episodes have to produce identical numbers.
///
/// The gate for a bug this crate had for real: `Fly::reset` restored the
/// network, the body and the muscle activation and left the wingbeat's phase,
/// the cord's wing command and the per-joint opposition counts running.
/// Measured before the fix, the same 400-tick flight episode four times over:
/// 53, 30, 36 and 30 ticks airborne. Every episodic experiment here - the
/// control matrix, the ceiling instrument, the evolution strategy - compares
/// one episode against another, so a reset that leaks is not noise, it is the
/// comparison being made against leftovers.
///
/// The CONTROL is the other direction, and it matters just as much: a reset
/// that also undid what a caller configured would be the bug this crate had in
/// its other form, when `reset` restored the connectome's original weights and
/// every episode of a learning run unlearned.
#[test]
fn two_identical_episodes_are_identical_and_a_reset_keeps_what_was_configured() {
    let Some(mut fly) = rig() else { return };
    let first = run(&mut fly, 1.0);
    let second = run(&mut fly, 1.0);
    assert_eq!(
        (first.net, first.ticks, first.spikes),
        (second.net, second.ticks, second.spikes),
        "two identical episodes differed; reset is leaking state"
    );

    // Configuration survives: a lesioned body is still lesioned. Every
    // actuator rather than one, because the cord drives some of them barely at
    // all and a single lesion that happens to land on one of those is a
    // control that passes without testing anything.
    for a in 0..fly.actuator_count() {
        fly.set_muscle_strength(a, 0.0).expect("an actuator exists");
    }
    let lesioned = run(&mut fly, 1.0);
    assert_ne!(lesioned.net, first.net, "a lesion had no effect, so reset undid it");
    let again = run(&mut fly, 1.0);
    assert_eq!(lesioned.net, again.net, "the lesion did not persist across a second reset");
}

/// A tuning has to reach the animal, and a tuning that names nothing has to be
/// visible as such.
///
/// The silent failure this guards: a tuning whose gains name the bare cord's
/// cell classes, applied to a creature built on the joined brain-and-cord
/// network, matches nothing, changes almost nothing, and leaves the caller
/// believing the animal is tuned. `Applied` reports the count so a caller can
/// refuse it - the SDK does.
#[test]
fn a_tuning_reaches_the_animal_and_reports_what_it_did_not_recognise() {
    let Some(mut fly) = rig() else { return };
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else { return };
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(root), "manc").expect("MANC");
    let c = connectome::load("manc", &neurons, &edges).expect("MANC loads");

    let before = run(&mut fly, 1.0);

    let mut t = fly::Tuning::default();
    // Silence the descending population outright: whatever else changes, an
    // animal whose command cannot reach its cord is a different animal.
    t.set("gain:descending", 0.0);
    t.set("activation_gain", 0.2);
    t.set("not_a_parameter", 1.0);
    let report = t.apply(&mut fly, &c, Wiring::default()).expect("a tuning applies");
    assert_eq!(report.gains, 1, "the one gain it names should have matched");
    assert_eq!(report.unknown, vec!["not_a_parameter".to_string()]);

    let after = run(&mut fly, 1.0);
    assert_ne!(before.net, after.net, "the tuning did not reach the animal");
    assert_eq!(fly.coupling().activation_gain, 0.2, "a non-gain parameter did not apply");

    // A tuning naming nothing this connectome has is not an error here - it is
    // a report of zero, which is what lets the SDK refuse it with a message
    // naming the file.
    let mut foreign = fly::Tuning::default();
    foreign.set("gain:optic_lobe_intrinsic", 2.0);
    let report = foreign.apply(&mut fly, &c, Wiring::default()).expect("still applies");
    assert_eq!(report.gains, 0, "a cord has no optic lobe");
    assert_eq!(report.unknown, vec!["gain:optic_lobe_intrinsic".to_string()]);
}

/// Falling over has to end a walk, and a height threshold does not catch it.
///
/// The measurement this exists for: under height-only termination a search
/// found a tuning scoring 0.089 - twenty times the imported connectome - that
/// finished 2.39 radians from upright. A fly on its back keeps its root
/// height, keeps its legs oscillating and covers ground, so every term in the
/// score was satisfied by a body that was not walking at all. The shuffled
/// control scored 0.044 and tipped 2.43, which is two animals falling over at
/// similar rates rather than a structural result.
#[test]
fn a_walk_that_ends_on_its_back_scores_nothing() {
    // The reward's own arithmetic, with no fly needed: an episode that ended
    // because the animal tipped over has too short a trace to hold a rhythm,
    // and a gait it cannot score is worth zero however far the body travelled.
    let ep = |net: f64, gait: Option<Gait>| fly::learn::Episode {
        net,
        gait,
        objective: Some(Objective::walk()),
        ..Default::default()
    };
    assert_eq!(ep(5.0, None).score(), 0.0, "a long slide with no scorable gait is not a walk");

    // And the threshold is in the objective rather than only in the report.
    match Objective::walk() {
        Objective::Walk { terminal_tip, .. } => {
            assert!(terminal_tip > 0.0 && terminal_tip < std::f64::consts::FRAC_PI_2);
        }
        other => panic!("walk() should be a Walk objective, got {other:?}"),
    }

    let Some(mut fly) = rig() else { return };
    // A tip threshold a walking fly is already past ends the episode at once;
    // the ordinary one does not. The pair is what makes it a test of the
    // THRESHOLD rather than of an episode that always terminates.
    let run = |fly: &mut Fly, terminal_tip| {
        let cfg = RewardConfig {
            objective: Objective::Walk { terminal_fall: 0.125, terminal_tip },
            ticks: 1000,
            command: 1.0,
            ..RewardConfig::default()
        };
        episode(fly, cfg, Condition::Frozen, &mut Lcg::new(1)).expect("an episode runs")
    };
    let strict = run(&mut fly, 0.001);
    let normal = run(&mut fly, 1.0);
    assert!(strict.terminated && strict.ticks < 1000, "a threshold under the standing attitude did not terminate");
    assert_eq!(strict.score(), 0.0, "an episode that ended immediately still scored {}", strict.score());
    assert!(!normal.terminated, "a standing fly tripped a one-radian attitude threshold");
}

/// An episode has to be reproducible after OTHER episodes, not just after
/// itself.
///
/// The existing gate runs the same episode twice in a row and compares. That
/// passes on a creature which drifts, because two consecutive episodes drift
/// together. What a search actually does is evaluate hundreds of different
/// parameter sets against one creature and compare their scores, and this is
/// the property that makes those comparisons mean anything: the same
/// parameters, before and after other work, have to give the same answer.
///
/// It was not true. Measured: a tuning scored 0.0954 when the search
/// re-evaluated it after 600-odd episodes, and 0.0 in a fresh process - the
/// same parameters, the same seed, a deterministic episode.
#[test]
fn the_same_parameters_score_the_same_before_and_after_other_episodes() {
    let Some(mut fly) = rig() else { return };
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else { return };
    let (neurons, edges) = connectome::find(std::path::PathBuf::from(root), "manc").expect("MANC");
    let c = connectome::load("manc", &neurons, &edges).expect("MANC loads");
    let gains = fly::learn::GainSearch::new(&c, Wiring::default());
    let base = fly.lif();

    // Everything a search sets, set the same way every time - WEIGHTS
    // included, because a search changes those on every evaluation and a
    // creature that remembered the last ones would make every comparison it
    // draws a comparison with history in it.
    let measure = |fly: &mut Fly| {
        fly.set_weights(&gains.weights(&gains.unit_gains())).expect("unit gains");
        fly.set_lif(base).expect("the published dynamics");
        fly.set_coupling(Coupling::default());
        run(fly, 1.0)
    };

    let first = measure(&mut fly);

    // Other work, of the kind a search does: different weights, different
    // dynamics, different coupling, episode after episode.
    for k in 0..12 {
        let f = 0.2 + 0.05 * k as f32;
        let mut g = gains.unit_gains();
        for (i, v) in g.iter_mut().enumerate() {
            *v = 0.3 + 0.2 * ((i + k) % 7) as f32;
        }
        fly.set_weights(&gains.weights(&g)).expect("a legal weight vector");
        fly.set_lif(neuro::LifParams { dt_over_tau_syn: f, dt_over_tau_inh: f * 0.5, ..base })
            .expect("a legal parameter set");
        fly.set_coupling(Coupling { activation_gain: 0.02 + 0.02 * k as f32, ..Coupling::default() });
        run(&mut fly, 1.0 + 0.1 * k as f32);
    }

    let again = measure(&mut fly);
    assert_eq!(
        (first.net, first.ticks, first.spikes),
        (again.net, again.ticks, again.spikes),
        "the same parameters gave a different episode after other episodes had run; \
         every comparison a search makes is between numbers measured this way"
    );
}
