// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two halves of DeepMimic, as properties of an episode.
//!
//! Early termination and reference-state initialisation are what make a
//! product reward learnable, and both are easy to implement in a way that
//! looks right and does nothing. These assert that each is load-bearing, with
//! the control that would fail if it were not.

use fly::learn::{episode_with, Condition, Lcg, Objective, RewardConfig, Start};
use fly::{Coupling, Fly, Reference, Timing};
use mujoco::{Model, MuJoCo};
use neuro::LifParams;

struct Rig {
    fly: Fly,
    reference: Reference,
}

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
    let Some(refpath) = std::env::var_os("BRAIN_FLY_REFERENCE").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLY_REFERENCE unset");
        return None;
    };
    let dir = std::path::PathBuf::from(root).join("manc-codex");
    if !dir.join("neurons.csv.gz").is_file() {
        brain_testutil::skip("manc-codex not present");
        return None;
    }
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz"))
        .expect("MANC loads");
    let model = Model::from_xml(&mj, std::path::PathBuf::from(xml)).expect("the fly loads");
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let fly = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default())
        .expect("the loop composes");
    let reference = Reference::load(std::path::PathBuf::from(refpath)).expect("the reference loads");
    Some(Rig { fly, reference })
}

#[test]
fn a_random_start_puts_the_body_somewhere_new_and_a_fixed_one_does_not() {
    let Some(mut rig) = rig() else { return };
    let cfg = |start| RewardConfig { objective: Objective::imitate(start), ticks: 4, ..RewardConfig::default() };

    // THE CONTROL, and it comes first: Start::Fixed must put the body in the
    // SAME place every time. Without it, "random starts differ" is also
    // satisfied by an episode whose initial pose is simply not set at all and
    // drifts from wherever the last one ended.
    let mut rng = Lcg::new(1);
    let mut fixed = Vec::new();
    for _ in 0..4 {
        episode_with(&mut rig.fly, cfg(Start::Fixed(0)), Condition::Frozen, Some(&rig.reference), &mut rng)
            .expect("a fixed-start episode runs");
        fixed.push(rig.fly.qpos());
    }
    for (i, q) in fixed.iter().enumerate().skip(1) {
        assert_eq!(q, &fixed[0], "fixed-start episode {i} did not begin where episode 0 did");
    }

    // Now the claim. Random starts must land on genuinely different frames of
    // the recording, which shows up as different body positions after the same
    // number of ticks.
    let mut starts = Vec::new();
    for _ in 0..8 {
        episode_with(
            &mut rig.fly,
            cfg(Start::Random { min_frames: 50 }),
            Condition::Frozen,
            Some(&rig.reference),
            &mut rng,
        )
        .expect("a random-start episode runs");
        starts.push(rig.fly.qpos());
    }
    let distinct = starts.iter().filter(|q| **q != starts[0]).count();
    assert!(distinct >= 6, "only {distinct} of 7 random starts differed from the first; the start is not random");
}

#[test]
fn an_episode_ends_when_the_body_loses_the_reference_and_runs_on_when_it_cannot() {
    let Some(mut rig) = rig() else { return };
    let mut rng = Lcg::new(7);
    // Bounded by the snippet, so "ran its whole budget" is a claim about the
    // termination radius and not about the recording running out.
    let budget = (rig.reference.len(0) as u32).min(300);
    assert!(budget > 100, "snippet 0 is only {budget} frames, too short to tell termination from exhaustion");

    // The creature cannot walk, so the recorded fly leaves it behind and the
    // episode must end well short of its budget.
    let tight = RewardConfig {
        objective: Objective::imitate(Start::Fixed(0)),
        ticks: budget,
        ..RewardConfig::default()
    };
    let ended = episode_with(&mut rig.fly, tight, Condition::Frozen, Some(&rig.reference), &mut rng)
        .expect("an episode runs");
    assert!(ended.terminated, "the episode did not terminate on losing the reference");
    assert!(ended.ticks < budget, "it terminated but still ran its whole budget of {budget} ticks");

    // THE CONTROL. With the radius opened up beyond anything the body can
    // reach, the SAME episode must run to its budget - which is what proves
    // the early ending above was the termination criterion firing rather than
    // some other limit, and that `ticks` is not simply always short.
    let loose = RewardConfig {
        objective: Objective::Imitate {
            start: Start::Fixed(0),
            reward: fly::ImitationReward::default(),
            terminal_com_dist: 1e9,
        },
        ticks: budget,
        ..RewardConfig::default()
    };
    let mut rng = Lcg::new(7);
    let full = episode_with(&mut rig.fly, loose, Condition::Frozen, Some(&rig.reference), &mut rng)
        .expect("an episode runs");
    assert!(!full.terminated, "with an unreachable radius the episode still terminated");
    assert_eq!(full.ticks, budget, "with an unreachable radius the episode should run its whole budget");
    assert!(
        full.reward > ended.reward,
        "a longer episode over the same trajectory should earn at least as much: {} vs {}",
        full.reward,
        ended.reward
    );
}
