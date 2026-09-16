// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does a trained creature keep what it has?
//!
//! "Weights frozen after training retains, does not adapt" is an entry in the
//! definition-of-done control matrix, and it is testable independently of
//! whether the thing learned is any good: retention is a property of the
//! MECHANISM. If freezing after training does not hold the weights and the
//! behaviour still, then nothing the creature learns can be claimed to persist,
//! whatever it learned.

use fly::learn::{episode, Condition, Lcg, RewardConfig};
use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo};
use neuro::{LifParams, PlasticityParams};

fn rig() -> Option<(connectome::Connectome, std::sync::Arc<MuJoCo>, std::path::PathBuf)> {
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
    Some((c, mj, std::path::PathBuf::from(xml)))
}

#[test]
fn freezing_after_training_holds_both_the_weights_and_the_behaviour() {
    let Some((c, mj, xml)) = rig() else { return };
    let model = Model::from_xml(&mj, &xml).expect("the fly loads");
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut f = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default())
        .expect("the loop composes");
    let bound = 1.5 * f.initial_weight_scale();
    f.enable_plasticity(PlasticityParams { eta: 0.02, w_min: -bound, w_max: bound, ..Default::default() })
        .unwrap();

    // Short episodes: this test is about the mechanism, not about how much is
    // learned, so it buys nothing by training for longer.
    let cfg = RewardConfig { ticks: 60, ..RewardConfig::default() };
    let mut rng = Lcg::new(0x5A1);
    let before = f.weights();
    for _ in 0..4 {
        episode(&mut f, cfg, Condition::Learning, &mut rng).unwrap();
    }
    let trained = f.weights();

    // Training must actually have moved something, or "retention" is retention
    // of nothing and this test is vacuous.
    let moved = before.iter().zip(&trained).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(
        moved > before.len() / 100,
        "only {moved} of {} weights moved during training; there is nothing to retain",
        before.len()
    );

    // Now freeze and run more episodes.
    let a = episode(&mut f, cfg, Condition::Frozen, &mut rng).unwrap();
    let after_first = f.weights();
    let b = episode(&mut f, cfg, Condition::Frozen, &mut rng).unwrap();
    let after_second = f.weights();

    // Retention of the WEIGHTS: bit-identical, not merely close. The learning
    // rate multiplies the modulator, and `Frozen` skips the update kernel
    // entirely, so anything other than bit-identical means a weight moved
    // through some path this test does not know about.
    let drifted = trained.iter().zip(&after_second).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(drifted, 0, "{drifted} weights moved after plasticity was disabled");
    assert_eq!(after_first, after_second);

    // Retention of the BEHAVIOUR: a frozen creature replaying the same episode
    // from the same reset is deterministic, so two frozen episodes must agree
    // exactly. A difference here would mean state is leaking between episodes.
    assert_eq!(a.distance.to_bits(), b.distance.to_bits(), "two frozen episodes diverged");
    assert_eq!(a.spikes, b.spikes, "frozen episodes differ in spike count");

    // And the frozen creature must still be doing something, or "it retained
    // its behaviour" is a claim about a corpse.
    assert!(a.spikes > 0, "the frozen creature is silent");
    eprintln!(
        "trained {moved} weights, then 2 frozen episodes: {} spikes each, distance {:+.6} cm both",
        a.spikes, a.distance
    );
}
