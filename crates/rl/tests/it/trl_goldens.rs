// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that a training engine computes what a reference trainer
// computes, number for number, for its clients. If your team needs expertise in porting reinforcement-learning
// objectives between frameworks without drift then you can procure our services by sending an email to
// info@swedishembedded.com.

//! brain against what TRL's `AsyncGRPOTrainer` computes, read from TRL's own code (`tests/fixtures/
//! trl_score_centring.json`, written by `tools/goldens/trl_score_centring_reference.py` from a pinned TRL source tree;
//! this test never runs Python):
//!
//! - one token's score-centred term ([`centred_term`]) against `compute_loss`'s locals: the mask, the weight at the
//!   sampled token, the extra-target coefficients (`-A residual`), the tail and the per-token loss;
//! - group advantages against `compute_advantages`. A rollout TRL scores NaN is a rollout that reached no outcome,
//!   which the record contract masks (`truncated`): it takes no part in its group and trains nothing (TRL: advantage
//!   0, also nothing). The one deliberate difference is the z-score's epsilon: brain divides by `std + 1e-4` (GRPO as
//!   published), TRL by `std + 1e-8`; it is asserted as exactly that and nothing else. A group with one scored rollout
//!   is a 0 advantage in TRL and refused by the record contract here (`group_of_one`).

use data::chat_rollout::{group_members, GroupEntry, GroupFault, TruncatedRollouts};
use rl::objective::advantage::{group_advantages_with, AdvantageNorm};
use rl::objective::score_centring::{centred_term, CentringBand};
use serde_json::Value;

fn fixture() -> Value {
    let path = brain_testutil::crate_path("tests/fixtures/trl_score_centring.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).expect("the fixture is JSON")
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn close(name: &str, field: &str, got: f64, want: f64) {
    assert!((got - want).abs() <= 2e-6 * want.abs().max(1.0), "{name}: {field}: brain {got}, TRL {want}");
}

#[test]
fn the_score_centred_term_is_trls() {
    let fixture = fixture();
    let cases = fixture["score_centring"].as_array().expect("cases");
    assert!(cases.len() >= 5, "the fixture holds the cases the tool writes");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let band = CentringBand { low: f(&case["low"]) as f32, high: f(&case["high"]) as f32 };
        let sampler: Vec<f32> = case["sampler_top_lp"].as_array().unwrap().iter().map(|v| v.as_f64().map_or(f32::NAN, |x| x as f32)).collect();
        let trainer: Vec<f32> = case["trainer_top_lp"].as_array().unwrap().iter().map(|v| f(v) as f32).collect();
        let advantage = f(&case["advantage"]);
        let mut coefs = vec![0f32; sampler.len()];
        let term = centred_term(f(&case["old_lp"]) as f32, f(&case["new_lp"]) as f32, advantage as f32, band, &sampler, &trainer, &mut coefs);
        assert_eq!(term.masked, case["is_masked"].as_bool().unwrap(), "{name}: is_masked");
        assert_eq!(term.tail, case["tail"].as_bool().unwrap(), "{name}: tail");
        // TRL's loss is -A (weight log pi(a) - correction) with weight = f(r_a): brain installs A f(r_a) at the token.
        close(name, "weight", f64::from(term.weight), advantage * f(&case["weight"]));
        for (j, (c, r)) in coefs.iter().zip(case["residual"].as_array().unwrap()).enumerate() {
            close(name, &format!("coefficient {j} (-A residual)"), f64::from(*c), -advantage * f(r));
        }
        close(name, "per-token loss", f64::from(term.loss), f(&case["per_token_loss"]));
        let mass: f64 = case["q"].as_array().unwrap().iter().map(f).sum();
        close(name, "sampler mass", f64::from(term.sampler_mass), mass);
    }
}

#[test]
fn group_advantages_are_trls_up_to_the_named_epsilon() {
    let fixture = fixture();
    for group in fixture["groups"].as_array().expect("groups") {
        let rewards: Vec<Option<f64>> = group["rewards"].as_array().unwrap().iter().map(Value::as_f64).collect();
        let trl: Vec<f64> = group["advantages"].as_array().unwrap().iter().map(f).collect();
        // A NaN reward is a rollout without an outcome: masked out of its group by the contract's one group reader.
        let entries: Vec<GroupEntry<u32, usize>> = rewards.iter().enumerate().map(|(i, r)| GroupEntry { group: 0, member: i, reward: r.unwrap_or(0.0), truncated: r.is_none() }).collect();
        let (groups, faults) = group_members(&entries, TruncatedRollouts::Mask);
        let scored: Vec<usize> = (0..rewards.len()).filter(|&i| rewards[i].is_some()).collect();
        if scored.len() == 1 {
            assert_eq!(faults, vec![GroupFault::OfOne { group: 0 }], "{group}: one scored rollout has nothing to compare with");
            assert!(trl.iter().all(|&a| a == 0.0), "{group}: TRL trains it with advantage 0");
            continue;
        }
        assert!(faults.is_empty(), "{group}: {faults:?}");
        let members = &groups[&0];
        let brain = group_advantages_with(AdvantageNorm::ZScore, &members.iter().map(|&(_, r)| r).collect::<Vec<_>>());
        let mean = members.iter().map(|&(_, r)| r).sum::<f64>() / members.len() as f64;
        let std = (members.iter().map(|&(_, r)| (r - mean) * (r - mean)).sum::<f64>() / members.len() as f64).sqrt();
        for (&(i, _), a) in members.iter().zip(&brain) {
            // Only the epsilon differs: brain's value rescaled to TRL's denominator is TRL's.
            let rescaled = a * (std + 1e-4) / (std + 1e-8);
            assert!((rescaled - trl[i]).abs() <= 1e-12 * trl[i].abs().max(1.0), "{group}: rollout {i}: brain {a} (rescaled {rescaled}), TRL {}", trl[i]);
        }
        for i in (0..rewards.len()).filter(|i| !scored.contains(i)) {
            assert!(members.iter().all(|&(m, _)| m != i) && trl[i] == 0.0, "{group}: rollout {i} without a score takes no part");
        }
    }
}
