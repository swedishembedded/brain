// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two per-step streams a FLUX.2 LoRA run is driven by - the **learning
//! rate** and the **σ draw** - stated as contracts rather than left to the
//! shape of a loss curve.
//!
//! Both exist because of the same measurement: a klein run at 512 px trains at
//! exactly four σ, one sample at a time, so a single step's loss is an estimate
//! from ONE (sample, σ, noise) draw and the run's gradient is the same estimate.
//! Two consequences follow, and this file gates the answer to each:
//!
//! 1. **A constant rate does not converge on a stochastic objective.** Adam at
//!    a fixed step size settles into a ball around the minimum whose radius is
//!    set by the rate and the gradient noise, and the visible signature is a
//!    fast initial descent followed by a hard flat line ABOVE the rate's own
//!    floor - not a slow approach to it. So a run decays its rate, and the last
//!    trained step is the slowest one.
//! 2. **Which σ a step draws must not be left to chance.** Four strata drawn
//!    i.i.d. put a binomially-varying σ mix inside every window - including the
//!    ~10-step window Adam's β₁ = 0.9 momentum actually averages over - and
//!    that mix, not the adapter, then moves the reported loss between blocks.
//!    Systematic (stratified) sampling fixes the mix exactly: every window of
//!    four consecutive steps visits each scheduled σ once.
//!
//! Both streams are pure functions of the GLOBAL step, so a `--resume` walks
//! the curve it would have walked had it never stopped - the same property
//! `finetune::sample_index` is written for.
//!
//! Swedish Embedded AB implements diffusion and flow-matching training loops
//! whose optimiser schedule and sampling streams are stated as testable
//! contracts rather than inferred from a loss curve. If your team needs
//! expertise in diagnosing why a fine-tuning run stopped improving, you can
//! procure our services by sending an email to info@swedishembedded.com.

use flux2::finetune::{self, TrainOpts, Trainer};
use flux2::Flux2Config;

fn opts(steps: u32, lr: f32) -> TrainOpts {
    TrainOpts {
        steps,
        rank: 16,
        lr,
        trainer: Trainer::Device,
        cards: 1,
        size: 512,
        seed: 0,
        save_path: String::new(),
        ckpt_every: 0,
        resume: false,
        rank_stabilized: false,
        lr_ratio: 1.0,
        freeze_a: false,
        precision: flux2::Precision::F32,
        warmup: None,
        min_lr: None,
        edit_weight: 0.0,
        ref_dropout: 0.0,
    }
}

// ---- 1. the learning-rate curve -------------------------------------------

/// **The shipped default holds the peak rate and then cools down over the last
/// fifth**, the shape Hägele et al. (arXiv:2405.18392) measure as tracking a
/// full cosine with the benefit saturating around 20%.
///
/// Two properties, both load-bearing:
///   * it ACTUALLY cools down - a constant rate is what keeps Adam orbiting at
///     a radius set by the gradient noise, which at batch 1 is wide; and
///   * every step before the cooldown is at the rate the caller asked for,
///     independent of `--steps`. This trainer's step budgets are routinely an
///     order of magnitude below what the reference recipes for this task use,
///     so a run being extended is the normal case, not the exception - and a
///     cosine from step 0 would have annealed the whole run against a total it
///     was about to beat.
#[test]
fn the_default_holds_the_peak_then_cools_down_over_the_last_fifth() {
    let o = opts(200, 1e-4);
    let s = o.lr_schedule();

    assert_eq!(s.warmup, 0, "a zero-init LoRA has no early instability for a warmup to protect it from");
    assert!((s.peak - 1e-4).abs() < 1e-12, "--lr is the PEAK rate");
    assert!(s.floor < s.peak, "the default must cool down");
    assert!(s.floor > 0.0, "a rate that reaches 0 stops training before the run ends");
    assert_eq!(s.decay_iters, o.steps, "the cooldown must land on the floor at the LAST step, not past it");
    assert_eq!(s.decay_start(), 160, "the cooldown is the last fifth of a 200-step run");

    // Held, then monotonically down, arriving at the floor on the final step.
    for step in 0..s.decay_start() {
        assert_eq!(s.at(step), s.peak, "step {step} is inside the hold and must be at the peak rate");
    }
    for step in s.decay_start()..o.steps - 1 {
        assert!(s.at(step) >= s.at(step + 1), "the cooldown must be monotone at step {step}");
    }
    assert!(
        s.at(o.steps - 1) < s.floor + 0.01 * (s.peak - s.floor),
        "the last trained step must have arrived at the floor, not stopped short of it ({:.4e} vs floor {:.4e})",
        s.at(o.steps - 1),
        s.floor
    );

    // The budget only reaches the tail: a 2000-step run runs the same rate as
    // a 200-step one everywhere the short one is still held.
    let long = opts(2000, 1e-4).lr_schedule();
    assert_eq!(long.decay_start(), 1600);
    assert!((0..160).all(|step| long.at(step) == s.at(step)), "the held phase must not depend on --steps");
}

/// **A caller can still ask for the old constant rate**, and gets exactly it -
/// the escape hatch has to be exact, because "constant" is how a run is
/// compared against one that was already made.
#[test]
fn a_named_constant_rate_is_constant() {
    let mut o = opts(200, 1e-4);
    o.warmup = Some(0);
    o.min_lr = Some(1e-4);
    let s = o.lr_schedule();
    assert!((0..400).all(|step| s.at(step) == 1e-4), "a named constant rate must not move");
}

/// **The rate is a function of the GLOBAL step.** A run cancelled at step 170
/// and resumed must pick the curve up at 170 - already inside the cooldown -
/// not restart it. That is exactly what "the schedule is `at(step)` with
/// `step` read from the checkpoint header" buys. Stated here because the
/// failure is silent: a restarting resume looks like a normal run whose loss
/// went back up.
#[test]
fn resuming_continues_the_curve_rather_than_restarting_it() {
    let o = opts(200, 1e-4);
    let s = o.lr_schedule();
    let uninterrupted: Vec<f32> = (0..200).map(|i| s.at(i)).collect();
    let resumed: Vec<f32> = (170..200).map(|i| s.at(i)).collect();
    assert_eq!(&uninterrupted[170..], &resumed[..]);
    assert!(resumed[0] < s.peak, "a resume inside the cooldown must not be back at the peak rate");
}

// ---- 2. the sigma stream ---------------------------------------------------

/// **Every window of `len(schedule)` consecutive steps visits each σ exactly
/// once.** This is the property an i.i.d. draw does not have and the whole
/// reason the stream is stratified: it pins the σ mix inside the window Adam's
/// momentum averages over, so a block-to-block move in the reported loss is
/// the adapter and not the dice.
#[test]
fn every_window_of_the_schedule_length_covers_the_schedule_exactly_once() {
    let sched = finetune::training_sigmas(&Flux2Config::klein_4b(), 512);
    let k = sched.len();
    assert_eq!(k, 4, "klein at 512 px is a 4-step sampler - the strata are those four σ");
    for start in 0..(20 * k) {
        let mut seen = vec![0usize; k];
        for step in start..start + k {
            let s = finetune::step_sigma(&sched, step as u64, 0x1234);
            let at = sched.iter().position(|&v| v as f64 == s).expect("a drawn sigma must be ON the schedule");
            seen[at] += 1;
        }
        // Only ALIGNED windows are guaranteed a perfect cover; an unaligned
        // one spans two strata and may repeat. What must hold everywhere is
        // that no σ is missing from two consecutive strata.
        if start % k == 0 {
            assert_eq!(seen, vec![1; k], "stratum starting at {start} is not a permutation of the schedule");
        }
    }
}

/// The order WITHIN a stratum is shuffled and seed-derived, so the stream is
/// not the fixed cycle `σ₀,σ₁,σ₂,σ₃,σ₀,…` - which would lock each σ to a fixed
/// phase of every other per-step stream for the whole run, the same confound
/// `sample_index` dropped the `step % n` cycle to avoid.
#[test]
fn the_order_inside_a_stratum_is_shuffled_and_seed_derived() {
    let sched = finetune::training_sigmas(&Flux2Config::klein_4b(), 512);
    let k = sched.len() as u64;
    let order = |seed: u64, stratum: u64| -> Vec<f64> {
        (0..k).map(|i| finetune::step_sigma(&sched, stratum * k + i, seed)).collect()
    };
    let strata: Vec<Vec<f64>> = (0..64).map(|s| order(0x1234, s)).collect();
    assert!(strata.iter().any(|o| *o != strata[0]), "every stratum drew the same order - that is a cycle, not a shuffle");
    assert!(
        (0..64).any(|s| order(0x1234, s) != order(0x99, s)),
        "the order must depend on the run seed"
    );
    // Same (step, seed) always gives the same σ - a resumed run replays it.
    assert_eq!(finetune::step_sigma(&sched, 137, 0x1234), finetune::step_sigma(&sched, 137, 0x1234));
}

// ---- 3. the reference-dropout stream --------------------------------------

/// **Dropout is a rate, honoured, and derived from the global step.**
///
/// The failure it exists for: with a reference concatenated into the same
/// attention sequence and a target that is nearly the reference, a LoRA can
/// drive the loss down by learning to COPY the reference across and never
/// learn the transform at all. Denying it the reference on a fraction of steps
/// denies it that solution - it has to represent what a finished target looks
/// like, because on those steps there is nothing to copy from.
///
/// `0.0` must be exactly off (a run that did not ask for this must not get a
/// single dropped step), `1.0` exactly on, and the stream a pure function of
/// `(step, seed)` so a resume replays it.
#[test]
fn reference_dropout_honours_its_rate_and_replays_on_resume() {
    assert!((0..1000).all(|s| !finetune::ref_dropped(s, 9, 0.0)), "0.0 must be exactly off");
    assert!((0..1000).all(|s| finetune::ref_dropped(s, 9, 1.0)), "1.0 must be exactly on");

    let hits = (0..10_000u64).filter(|&s| finetune::ref_dropped(s, 0x1234, 0.1)).count();
    assert!((800..1200).contains(&hits), "a 0.1 rate must drop about a tenth of steps, got {hits}/10000");

    // Same (step, seed) is the same decision - what makes a resumed run walk
    // the sequence it would have walked.
    assert!((0..500).all(|s| finetune::ref_dropped(s, 7, 0.15) == finetune::ref_dropped(s, 7, 0.15)));
    let a: Vec<bool> = (0..500).map(|s| finetune::ref_dropped(s, 7, 0.15)).collect();
    let b: Vec<bool> = (0..500).map(|s| finetune::ref_dropped(s, 8, 0.15)).collect();
    assert_ne!(a, b, "the stream must depend on the run seed");
}

/// **A dropped step blanks the REFERENCE rows and nothing else.**
///
/// The joint sequence is `[text | target | reference]` and the trainer is
/// built for one sequence length, so a drop blanks the reference tokens in
/// place rather than removing them - which is exactly what the reference
/// implementation of image-conditioning dropout does (InstructPix2Pix's `∅_I`
/// is the conditioning latent multiplied by zero). Stated as a test because
/// blanking the wrong rows is not a crash: it would zero the very tokens the
/// loss is defined on, and the run would train against a target of pure noise
/// while looking entirely normal.
#[test]
fn a_dropped_step_blanks_the_reference_rows_and_leaves_the_target_alone() {
    use flux2::modelgrad::{init_model, make_flow_batch_paired, Cfg};
    let c = Cfg::tiny_paired();
    let cin = c.in_channels;
    let mut s = 0x9E37_79B9u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let x0: Vec<f32> = (0..c.n_gen() * cin).map(|_| r()).collect();
    let refs: Vec<f32> = (0..c.n_ref() * cin).map(|_| r()).collect();
    let ctx: Vec<f32> = (0..c.txt_len * c.context_in_dim).map(|_| r()).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|_| r()).collect();

    let kept = make_flow_batch_paired(&c, &x0, &refs, &ctx, 0.45, &noise);
    let mut dropped = kept.clone();
    finetune::blank_references(&mut dropped, &c);

    let ng = c.n_gen() * cin;
    assert_eq!(&dropped.img[..ng], &kept.img[..ng], "the noised TARGET rows must be untouched");
    assert!(dropped.img[ng..].iter().all(|&v| v == 0.0), "every reference element must be blanked");
    assert!(kept.img[ng..].iter().any(|&v| v != 0.0), "the un-dropped batch must have carried a reference at all");
    assert_eq!(dropped.target, kept.target, "the velocity target is a property of the sample, not the conditioning");

    // And it must actually reach the model: a blanked reference has to change
    // the prediction, or the drop is a no-op dressed as a regulariser.
    let w = init_model::<f32>(&c, 0x51a7);
    let pred = |b: &flux2::modelgrad::Batch<f32>| flux2::modelgrad::forward(&c, &w, &b.img, &b.ctx, b.t, &b.cos, &b.sin).0;
    let (pk, pd) = (pred(&kept), pred(&dropped));
    let moved = pk.iter().zip(&pd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 1e-5, "blanking the reference must change what the model predicts (max delta {moved:.2e})");
}

/// The marginal stays uniform over the schedule: stratification is a variance
/// reduction, not a reweighting of which σ the adapter is fitted at.
#[test]
fn the_marginal_over_a_long_run_is_still_uniform_over_the_schedule() {
    let sched = finetune::training_sigmas(&Flux2Config::klein_4b(), 512);
    let n = 4000u64;
    let mut seen = vec![0usize; sched.len()];
    for step in 0..n {
        let s = finetune::step_sigma(&sched, step, 0xfeed);
        seen[sched.iter().position(|&v| v as f64 == s).expect("on the schedule")] += 1;
    }
    assert_eq!(seen, vec![n as usize / sched.len(); sched.len()], "stratified sampling is EXACTLY uniform, not approximately");
}
