// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting trained models into
// the memory a deployment has, without guessing which layers can take it, for
// its clients. If your team needs expertise in mixed-precision model
// compression then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The weight precision search, end to end on a trained model.
//!
//! A tiny qwen3 learns a task that needs two tokens of context
//! (`x[t+1] = x[t] + x[t-1] mod vocab`), so its linears carry real, unevenly
//! important structure. The search ranks every site and tier on calibration
//! batches, solves for a byte budget halfway between uniform int8 and uniform
//! q4, and the result is then measured on held-out batches it never saw:
//!
//! * it meets its budget,
//! * its held-out loss is lower than every uniform tier that fits the same
//!   budget (the comparison a mixed policy has to win to be worth having),
//! * it beats the median of plans with the same tiers on the same bytes but
//!   dealt to sites of the same shape at random, so the win comes from the
//!   sensitivity ranking and not from the bytes alone,
//! * measuring a policy leaves the model's weights exactly as they were.

use gpu_core::select::Dtype;
use model::{Batch, Model, IGNORE};
use modelopt::{held_out_loss, measure_policy, search_weights, sites, Assignment, Budget, Progress, SearchRequest};

const ROWS: u32 = 8;

/// `ROWS` sequences of the recurrence from random starting pairs; the first
/// target of each row is unpredictable and masked.
fn recurrence(rng: &mut data::rng::Rng, vocab: u32, seq: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut tokens, mut targets) = (Vec::new(), Vec::new());
    for _ in 0..ROWS {
        let mut x = vec![rng.gen_range_inclusive(0, vocab as i64 - 1) as u32, rng.gen_range_inclusive(0, vocab as i64 - 1) as u32];
        while x.len() < seq as usize + 1 {
            let next = (x[x.len() - 1] + x[x.len() - 2]) % vocab;
            x.push(next);
        }
        tokens.extend_from_slice(&x[..seq as usize]);
        targets.push(IGNORE);
        targets.extend_from_slice(&x[2..]);
    }
    (tokens, targets)
}

fn batches(rng: &mut data::rng::Rng, count: usize, vocab: u32, seq: u32) -> Vec<(Vec<u32>, Vec<u32>)> {
    (0..count).map(|_| recurrence(rng, vocab, seq)).collect()
}

fn as_batches(owned: &[(Vec<u32>, Vec<u32>)]) -> Vec<Batch<'_>> {
    owned.iter().map(|(tokens, targets)| Batch::Lm { tokens, targets }).collect()
}

/// A tiny qwen3 trained on the recurrence until it has learned it.
fn trained(rng: &mut data::rng::Rng) -> qwen3::Qwen {
    let cfg = qwen3::QwenConfig::tiny_i8();
    let (vocab, seq) = (cfg.vocab, cfg.block_size);
    let m = qwen3::Qwen::new(cfg, ROWS, seq, &qwen3::init_weights(&qwen3::QwenConfig::tiny_i8(), rng.next_u64()));
    let train = batches(rng, 64, vocab, seq);
    let first = held_out_loss(&m, &as_batches(&train[..4]));
    for step in 0..600u32 {
        let (tokens, targets) = &train[step as usize % train.len()];
        Model::zero_grads(&m);
        Model::set_batch(&m, Batch::Lm { tokens, targets });
        Model::forward(&m);
        Model::backward(&m);
        Model::adamw_step(&m, step + 1, 1e-2, 0.0, Default::default(), Some(1.0), 1.0);
    }
    let learned = held_out_loss(&m, &as_batches(&train[..4]));
    assert!(learned < 0.5 * first, "the fixture must learn the task before it is compressed: loss {first} -> {learned}");
    m
}

/// `plan`'s tiers dealt again among sites of the same tensor shapes: the same
/// bytes and the same tier counts, with the choice of which site gets which
/// left to chance.
fn shuffle_within_shapes(sites: &[modelopt::Site], plan: &[Assignment], rng: &mut data::rng::Rng) -> Vec<Assignment> {
    let shape = |s: &modelopt::Site| s.tensors.iter().map(|t| (t.n, t.k)).collect::<Vec<_>>();
    let mut out = plan.to_vec();
    let mut shapes: Vec<Vec<(usize, usize)>> = sites.iter().map(shape).collect();
    shapes.sort();
    shapes.dedup();
    for wanted in shapes {
        let idx: Vec<usize> = (0..sites.len()).filter(|&i| shape(&sites[i]) == wanted).collect();
        let mut tiers: Vec<Dtype> = idx.iter().map(|&i| plan[i].tier).collect();
        for i in (1..tiers.len()).rev() {
            tiers.swap(i, rng.gen_range_inclusive(0, i as i64) as usize);
        }
        for (&i, tier) in idx.iter().zip(tiers) {
            out[i].tier = tier;
        }
    }
    out
}

const TIERS: [Dtype; 5] = [Dtype::F32, Dtype::BF16, Dtype::I8, Dtype::NF4, Dtype::Q4];

#[test]
fn searched_policy_beats_every_uniform_tier_within_its_budget() {
    let mut rng = data::rng::Rng::random();
    let m = trained(&mut rng);
    let (vocab, seq) = (m.config().vocab, m.config().block_size);

    let sites = sites(&m).expect("qwen3 declares its linear sites");
    assert_eq!(sites.len(), 8, "per layer: fused Q/K/V, wo, fused gate/up, down");
    let calibration = batches(&mut rng, 4, vocab, seq);
    let held_out = batches(&mut rng, 8, vocab, seq);
    let (calibration, held_out) = (as_batches(&calibration), as_batches(&held_out));

    let bytes_at = |tier: Dtype| sites.iter().flat_map(|s| &s.tensors).map(|t| model::ops::Weight::bytes_for(tier, t.n, t.k)).sum::<u64>();
    // `Budget::Midway`: halfway between the smallest uniform size (q4 and nf4 tie) and the next (int8).
    let budget = bytes_at(Dtype::Q4) + (bytes_at(Dtype::I8) - bytes_at(Dtype::Q4)) / 2;
    let original = held_out_loss(&m, &held_out);
    let mut progress = Vec::new();
    let report = search_weights(&m, &SearchRequest { tiers: &TIERS, budget: Budget::Midway, calibration: &calibration, held_out: &held_out, timing: None }, &mut |p| progress.push(p)).expect("search");
    let count = |f: fn(&Progress) -> bool| progress.iter().filter(|p| f(p)).count();
    assert_eq!(count(|p| matches!(p, Progress::Calibrated { .. })), calibration.len(), "one event per calibration batch");
    assert_eq!(count(|p| matches!(p, Progress::Estimated { .. })), sites.len(), "one event per site");
    let measured: Vec<&str> = progress.iter().filter_map(|p| if let Progress::Measuring { label } = p { Some(label.as_str()) } else { None }).collect();
    assert_eq!(measured.len(), 2 + report.baselines.len(), "the original, the plan and every baseline: {measured:?}");
    assert_eq!(held_out_loss(&m, &held_out), original, "a search must leave every weight as it was");
    assert_eq!(report.budget_bytes, budget, "a midway budget");
    assert!(report.plan_bytes <= budget, "plan {} bytes over its budget {budget}", report.plan_bytes);
    assert_eq!(report.original_loss, original);
    let fitting: Vec<&str> = TIERS.iter().filter(|&&t| bytes_at(t) <= budget).map(|t| t.name()).collect();
    assert_eq!(report.baselines.iter().map(|b| b.tier).collect::<Vec<_>>(), fitting, "every uniform tier that fits is a baseline");
    assert!(report.beats_baselines(), "searched loss {} must beat every uniform tier that fits: {:?}", report.plan_loss, report.baselines);

    let policy = model::ops::TierPolicy::parse(&report.policy).expect("the report's policy parses as --weight-format");
    for group in m.coupled_tensors() {
        let tiers: Vec<Dtype> = group.iter().map(|t| policy.want(t)).collect();
        assert!(tiers.windows(2).all(|w| w[0] == w[1]), "a fused group takes one tier: {group:?} -> {tiers:?}");
    }

    let plan: Vec<Assignment> = report.sites.iter().map(|s| Assignment { site: s.site.clone(), tier: Dtype::from_name(s.tier).expect("a tier name") }).collect();
    let mut shuffled: Vec<f64> = (0..9).map(|_| measure_policy(&m, &sites, &shuffle_within_shapes(&sites, &plan, &mut rng), &held_out)).collect();
    shuffled.sort_by(f64::total_cmp);
    assert!(report.plan_loss < shuffled[4], "searched loss {} must beat the median {} of the same tiers dealt at random ({shuffled:?})", report.plan_loss, shuffled[4]);
    eprintln!("original {original}, searched {} at {:.2} bits/weight ({}), shuffled median {}, baselines {:?}", report.plan_loss, report.bits_per_weight, policy.summary(), shuffled[4], report.baselines);
}

#[test]
fn a_search_refuses_to_accept_on_the_data_it_ranked_on() {
    let mut rng = data::rng::Rng::random();
    let cfg = qwen3::QwenConfig::tiny_i8();
    let m = qwen3::Qwen::new(cfg.clone(), ROWS, cfg.block_size, &qwen3::init_weights(&cfg, rng.next_u64()));
    let owned = batches(&mut rng, 2, cfg.vocab, cfg.block_size);
    let both = as_batches(&owned);
    let err = search_weights(&m, &SearchRequest { tiers: &TIERS, budget: Budget::BitsPerWeight(8.5), calibration: &both, held_out: &both[1..], timing: None }, &mut |_| {}).expect_err("overlapping sets are refused");
    assert!(err.contains("held-out"), "{err}");
}

/// With the sites timed (here by a synthetic device on which nf4 is small but
/// slow), a latency limit keeps the plan off the slow tier even where its
/// bytes would fit, every distinct product shape is timed once per tier, and
/// the report carries the seconds of the plan and of each baseline.
#[test]
fn a_latency_limit_keeps_a_slow_tier_out_of_the_plan() {
    let mut rng = data::rng::Rng::random();
    let m = trained(&mut rng);
    let (vocab, seq) = (m.config().vocab, m.config().block_size);
    let calibration = batches(&mut rng, 2, vocab, seq);
    let held_out = batches(&mut rng, 4, vocab, seq);
    let (calibration, held_out) = (as_batches(&calibration), as_batches(&held_out));
    let calls = std::cell::Cell::new(0usize);
    let per_element = |tier: Dtype| match tier {
        Dtype::F32 => 4.0,
        Dtype::BF16 => 2.0,
        Dtype::I8 => 1.0,
        Dtype::Q4 => 0.6,
        _ => 8.0,
    };
    let timer = |rows: u32, n: usize, k: usize, tier: Dtype| -> Result<f64, String> {
        calls.set(calls.get() + 1);
        Ok(rows as f64 * n as f64 * k as f64 * per_element(tier) * 1e-9)
    };
    let timing = modelopt::Timing { rows: 1, limit: Some(modelopt::LatencyLimit::Uniform(Dtype::I8)), time: &timer };
    let report = search_weights(&m, &SearchRequest { tiers: &TIERS, budget: Budget::Midway, calibration: &calibration, held_out: &held_out, timing: Some(timing) }, &mut |_| {}).expect("search");
    let limit = report.latency_limit_seconds.expect("a latency limit");
    let plan_seconds = report.plan_seconds.expect("the plan is timed");
    assert!(plan_seconds <= limit * (1.0 + 1e-9), "plan {plan_seconds} s over its limit {limit} s");
    assert!(report.sites.iter().all(|s| s.tier != "nf4"), "nf4 is the slow tier: {}", report.policy);
    assert!(report.baselines.iter().all(|b| b.seconds.is_some_and(|s| s <= limit * (1.0 + 1e-9))), "a baseline fits both limits: {:?}", report.baselines);
    // tiny_i8 products: fused Q/K/V, wo, fused gate/up, down - four shapes, each at every tier it packs.
    assert_eq!(calls.get(), 4 * TIERS.len(), "every distinct shape is timed once per tier");
}
