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
use modelopt::{assign, held_out_loss, measure_policy, sensitivity, sites, Assignment};

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

fn uniform(sites: &[modelopt::Site], tier: Dtype) -> Vec<Assignment> {
    sites.iter().map(|s| Assignment { site: s.name.clone(), tier }).collect()
}

/// `plan`'s tiers dealt again among sites of the same shape: the same bytes
/// and the same tier counts, with the choice of which site gets which left to
/// chance.
fn shuffle_within_shapes(sites: &[modelopt::Site], plan: &[Assignment], rng: &mut data::rng::Rng) -> Vec<Assignment> {
    let mut out = plan.to_vec();
    let mut shapes: Vec<(usize, usize)> = sites.iter().map(|s| (s.n, s.k)).collect();
    shapes.sort();
    shapes.dedup();
    for shape in shapes {
        let idx: Vec<usize> = (0..sites.len()).filter(|&i| (sites[i].n, sites[i].k) == shape).collect();
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

#[test]
fn searched_policy_beats_every_uniform_tier_within_its_budget() {
    let cfg = qwen3::QwenConfig::tiny_i8();
    let (vocab, seq) = (cfg.vocab, cfg.block_size);
    let mut rng = data::rng::Rng::random();
    let init = qwen3::init_weights(&cfg, rng.next_u64());
    let m = qwen3::Qwen::new(cfg, ROWS, seq, &init);

    let train = batches(&mut rng, 64, vocab, seq);
    let first = held_out_loss(&m, &as_batches(&train[..4]));
    for step in 0..600u32 {
        let (tokens, targets) = &train[step as usize % train.len()];
        Model::zero_grads(&m);
        Model::set_batch(&m, Batch::Lm { tokens, targets });
        Model::forward(&m);
        Model::backward(&m);
        Model::adamw_step(&m, step + 1, 1e-2, 0.0, Default::default(), Some(1.0), 1.0);
    }
    let trained = held_out_loss(&m, &as_batches(&train[..4]));
    assert!(trained < 0.5 * first, "the fixture must learn the task before it is compressed: loss {first} -> {trained}");

    let sites = sites(&m).expect("qwen3 declares its linear sites");
    assert_eq!(sites.len(), 14, "two layers of seven linears");
    let calibration = batches(&mut rng, 4, vocab, seq);
    let held_out = batches(&mut rng, 8, vocab, seq);
    let (calibration, held_out) = (as_batches(&calibration), as_batches(&held_out));

    let tiers = [Dtype::F32, Dtype::BF16, Dtype::I8, Dtype::NF4, Dtype::Q4];
    let table = sensitivity(&m, &sites, &tiers, &calibration).expect("sensitivity");
    let bytes_at = |tier: Dtype| sites.iter().map(|s| model::ops::Weight::bytes_for(tier, s.n, s.k)).sum::<u64>();
    let budget = (bytes_at(Dtype::I8) + bytes_at(Dtype::Q4)) / 2;
    let plan = assign(&table, budget).expect("a budget above uniform q4 is feasible");
    assert!(plan.bytes <= budget, "plan {} bytes over its budget {budget}", plan.bytes);

    let original = held_out_loss(&m, &held_out);
    let searched = measure_policy(&m, &sites, &plan.assignments, &held_out);
    assert_eq!(held_out_loss(&m, &held_out), original, "measuring a policy must restore every weight");
    for tier in tiers.into_iter().filter(|&t| bytes_at(t) <= budget) {
        let baseline = measure_policy(&m, &sites, &uniform(&sites, tier), &held_out);
        assert!(searched < baseline, "searched policy ({} bytes) loss {searched} must beat uniform {tier:?} ({} bytes) loss {baseline}", plan.bytes, bytes_at(tier));
    }
    let mut shuffled: Vec<f64> = (0..9).map(|_| measure_policy(&m, &sites, &shuffle_within_shapes(&sites, &plan.assignments, &mut rng), &held_out)).collect();
    shuffled.sort_by(f64::total_cmp);
    assert!(searched < shuffled[4], "searched loss {searched} must beat the median {} of the same tiers dealt at random ({shuffled:?})", shuffled[4]);
    eprintln!("shuffled median {}", shuffled[4]);
    eprintln!("original {original}, searched {searched} at {} of {} bytes ({})", plan.bytes, bytes_at(Dtype::F32), plan.policy(&sites).summary());
}
