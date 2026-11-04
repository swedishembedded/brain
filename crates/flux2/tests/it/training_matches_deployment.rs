// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ways a FLUX.2 fine-tune used to optimise something other than what
//! `brain flux2 generate` actually runs, each stated as a gate:
//!
//! 1. the noise level σ a training step draws must be one the deployed sampler
//!    visits;
//! 2. the text encoder a caption is embedded through must be the tier
//!    `generate` would build for the same DiT;
//! 3. sample order must not be a fixed cycle, which confounds sample identity
//!    with epoch for the whole run;
//! 4. a folder's reference-image pairing must be all or nothing, because a run
//!    trains ONE joint sequence layout.
//!
//! Swedish Embedded AB implements deployment-faithful training pipelines for
//! its clients. If your team needs expertise in closing train/serve skew, you
//! can procure our services by sending an email to info@swedishembedded.com.

use flux2::{finetune, Flux2Config, Precision};

// ---- 1. sigma sampling ----------------------------------------------------

/// **A training step must land on a σ the sampler actually visits.**
///
/// klein is a distilled FIXED 4-step sampler: at 512 px it evaluates the DiT
/// at four σ values and nowhere else. Drawing σ ~ U(0,1) spent roughly two
/// thirds of every run optimising a regime generation never enters. The
/// schedule this reads is `diffusion::scheduler::klein_sigmas` - the ONE
/// schedule `pipeline`'s Euler loop integrates - so the two cannot drift.
#[test]
fn training_sigmas_are_the_schedule_the_sampler_integrates() {
    let cfg = Flux2Config::klein_4b();
    let size = 512u32;
    let n_gen = ((size / 16) * (size / 16)) as usize;
    let got = finetune::training_sigmas(&cfg, size);

    // The sampler's own schedule, minus its terminal 0 - the model is never
    // evaluated at σ = 0, that entry only closes the last Euler interval.
    let mut want = diffusion::scheduler::klein_sigmas(4, n_gen);
    want.pop();
    assert_eq!(got, want, "training must draw from the deployed schedule");

    // The measured klein-4-step band at 512 px, so a change to the scheduler
    // that silently moves it shows up here as a number, not as a diff.
    let round = |v: f32| (v * 1e4).round() / 1e4;
    assert_eq!(got.iter().copied().map(round).collect::<Vec<_>>(), vec![1.0, 0.9581, 0.8840, 0.7175]);

    // Resolution-dependent, because `empirical_mu` is: a run configured at a
    // different size trains on that size's band, not on 512's.
    let other = finetune::training_sigmas(&cfg, 768);
    assert_ne!(other, got, "the band shifts with resolution");

    // An undistilled base variant integrates its own, much longer schedule.
    assert_eq!(finetune::training_sigmas(&Flux2Config::klein_base_4b(), size).len(), 50);
}

/// **Every draw is on the schedule, and the whole schedule is drawn.** A
/// sampler that only ever returned σ = 1 would satisfy "on the schedule" and
/// train nothing useful, so coverage is asserted too.
#[test]
fn every_drawn_sigma_is_on_the_schedule_and_all_of_it_is_reached() {
    let cfg = Flux2Config::klein_4b();
    let sched = finetune::training_sigmas(&cfg, 512);
    let mut seen = vec![0usize; sched.len()];
    for step in 0..2000u64 {
        let s = finetune::step_sigma(&sched, step, 0x0051_67a1);
        let at = sched.iter().position(|&v| v as f64 == s).expect("a drawn sigma must be ON the schedule");
        seen[at] += 1;
    }
    assert!(seen.iter().all(|&c| c > 0), "every scheduled sigma must be reachable: {seen:?}");
    // Uniform over the schedule: no entry may collapse or dominate.
    for c in &seen {
        assert!(*c > 2000 / (2 * sched.len()), "sigma coverage is lopsided: {seen:?}");
    }
}

// ---- 2. text-encoder precision --------------------------------------------

/// **Training must embed captions through the encoder `generate` would
/// build.** The tier follows the DiT's: `plan_parts` asks for int8 beside an
/// int8 DiT and f32 beside an f32 one, and training now resolves the DiT's
/// effective precision the same way `generate` does - so a `.gguf` DiT, which
/// `generate` always runs int8, gets the int8 encoder here too, without the
/// caller having to know that.
///
/// Fitting an adapter against f32 conditioning vectors and then deploying it
/// against int8 ones is a silent mismatch in the one input the adapter is
/// supposed to be keyed on.
#[test]
fn training_picks_the_text_encoder_tier_generate_would_pick() {
    for (dit, requested, want_int8) in [
        ("transformer/model.safetensors", Precision::F32, false),
        ("transformer/model.safetensors", Precision::Int8, true),
        // A Q8_0 GGUF has no fp32 tier at all: `generate` forces int8, so
        // training must too even though nothing was asked for.
        ("klein-9b-Q8_0.gguf", Precision::F32, true),
        ("klein-9b-Q8_0.gguf", Precision::Int8, true),
    ] {
        let te = finetune::text_encoder_placement(dit, requested).expect("resolvable");
        assert_eq!(te.int8, want_int8, "{dit} at {requested:?}");
        // ... and it is the same tier the placement planner tries FIRST for
        // that DiT, which is what `generate` ends up with unless the card
        // cannot hold it.
        let effective = flux2::pipeline::effective_dit_precision(dit, requested, false).unwrap();
        assert_eq!(te.int8, flux2::pipeline::te_tier_int8(effective), "{dit} at {requested:?}");
    }
}

// ---- 3. sample order ------------------------------------------------------

/// **Sample order must not be a fixed cycle.** `encoded[step % n]` makes
/// sample identity and epoch the same variable for the whole run: sample `i`
/// is only ever seen at steps `i`, `i+n`, `i+2n`, always in the same order,
/// always paired with the same position in the learning-rate and σ streams.
///
/// What replaces it is random-without-replacement per pass, so each epoch is a
/// permutation: every sample is still seen exactly once per `n` steps (no
/// sample starves and none is over-weighted), but which one, and in what
/// order, changes from epoch to epoch.
#[test]
fn sample_order_is_a_fresh_permutation_each_epoch() {
    let (n, seed) = (12usize, 0xa11ce);
    let order = |epoch: u64| -> Vec<usize> { (0..n).map(|k| finetune::sample_index(n, epoch * n as u64 + k as u64, seed)).collect() };

    let e0 = order(0);
    let e1 = order(1);
    for (e, o) in [(0, &e0), (1, &e1)] {
        let mut sorted = (*o).clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "epoch {e} must be a permutation: {o:?}");
    }
    assert_ne!(e0, e1, "two epochs must not walk the same order");
    assert_ne!(e0, (0..n).collect::<Vec<_>>(), "the order must not be the cyclic one it replaced");

    // Deterministic: the same run re-run - or resumed - replays its schedule.
    assert_eq!(order(0), e0);
    assert_eq!(finetune::sample_index(n, 5, seed), e0[5]);
    // ... and it is the SEED's schedule, not a global one.
    let other: Vec<usize> = (0..n).map(|k| finetune::sample_index(n, k as u64, seed ^ 1)).collect();
    assert_ne!(other, e0, "a different seed must draw a different order");

    // A single-sample dataset is not a special case.
    assert_eq!(finetune::sample_index(1, 7, seed), 0);
}

// ---- 4. paired datasets ---------------------------------------------------

/// **A run trains one joint sequence layout, so a folder pairs all of its
/// targets or none.** Half-pairing is a dataset mistake with no safe reading:
/// a blank reference teaches "no photograph here", and quietly dropping the
/// unpaired samples trains on a subset the operator did not choose.
#[test]
fn a_half_paired_dataset_is_refused_by_name() {
    let sample = |name: &str, reference: Option<Vec<f32>>| data::imageset::Sample {
        path: std::path::PathBuf::from(name),
        prompt: "a room".into(),
        hwc: vec![0.0; 3],
        size: 32,
        reference,
        ref_path: None,
    };
    let all_paired = vec![sample("a.jpg", Some(vec![0.0; 3])), sample("b.jpg", Some(vec![0.0; 3]))];
    assert_eq!(finetune::dataset_refs(&all_paired, 2, 2).unwrap(), vec![(2, 2)]);

    let none_paired = vec![sample("a.jpg", None), sample("b.jpg", None)];
    assert!(finetune::dataset_refs(&none_paired, 2, 2).unwrap().is_empty(), "caption-only is still caption-only");

    let mixed = vec![sample("a.jpg", Some(vec![0.0; 3])), sample("lonely.jpg", None)];
    let err = finetune::dataset_refs(&mixed, 2, 2).unwrap_err();
    assert!(err.contains("lonely.jpg"), "the error must name the offending sample: {err}");
    assert!(err.contains("pairs.yaml"), "... and the manifest to fix: {err}");
}
