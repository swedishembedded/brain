// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning every weight of a
// model over several cards from one served request for its clients. If your
// team needs expertise in sharded full-parameter training as a service then
// you can procure our services by sending an email to info@swedishembedded.com.

//! A served full fine-tune (`full_train`, objective `full_sft`) on several cards of its server: the `cards` and `job`
//! of a `train` request, one data-parallel rank per card, sharded as the request asks (`sharding`) or as far as the
//! cards need, every claimed card charged one rank's sharded footprint, and the report saying how the run was spread.
//!
//! The server computes on the CPU backend (the suite's `BRAIN_DEVICE=cpu`) while the residency accounts for the
//! cards it placed the run on.

use std::collections::BTreeSet;

use residency::Device;
use serde_json::{json, Value};
use trainjob::consolidate::{full_spec, FULL_ACTION};

use crate::cards::{cards_charged, run_action_observed};
use crate::support::{assert_reservation_returned, data_digest, plain, rig_qwen3_cards, rig_qwen3_on, run_action, work_is_clean, Frames, Rig, BASE};

fn params(rig: &Rig, cards: u32, sharding: Option<&str>) -> Value {
    let train = data_digest(rig, &plain());
    let mut params = json!({"base": BASE, "steps": 2, "lr": 1e-3, "seed": 7, "max_block": 128, "cards": cards, "data": json!({"train": {"digest": train.hex()}}).to_string()});
    if let Some(sharding) = sharding {
        params["sharding"] = json!(sharding);
    }
    params
}

fn invocation(params: Value) -> capability::Invocation {
    let mut inv = capability::Invocation::new();
    inv.params = params;
    full_spec().validate_served(inv).expect("a valid full_train request")
}

/// What one rank of the toy base asks of a card, sharded `sharding` over `world` ranks.
fn rank_bytes(rig: &Rig, sharding: brain::Sharding, world: usize) -> u64 {
    let base = rig.dir.join("models").join("local").join("toy");
    brain::ChatFullFineTune::from_pretrained(base.to_str().unwrap()).max_block(128).sharded_footprint(128, sharding, world).expect("the footprint").device_bytes
}

#[test]
fn a_full_fine_tune_on_two_cards_shards_as_asked_charges_both_cards_and_reports_its_group() {
    let rig = rig_qwen3_on("full-cards-two", 16 << 30, &[0, 1], false);
    let observed = run_action_observed(&rig, FULL_ACTION, invocation(params(&rig, 2, Some("gradients"))), false);
    let held = observed.held.expect("a step was reported");
    let report = observed.result.expect("the two-card run").outputs;
    assert_eq!(cards_charged(&held), BTreeSet::from([0, 1]), "the run holds both cards: {held:?}");
    let charged: Vec<u64> = held.budgets.iter().filter(|b| matches!(b.device, Device::Gpu(_))).map(|b| b.used).collect();
    assert_eq!(charged, vec![rank_bytes(&rig, brain::Sharding::Gradients, 2); 2], "each card is charged one gradient-sharded rank");
    assert_eq!(observed.steps, 2, "rank 0 streams each step once");
    assert_eq!((report["status"].as_str(), report["objective"].as_str(), report["steps_completed"].as_u64()), (Some("completed"), Some("full_sft"), Some(2)), "{report}");
    assert_eq!((&report["parallel"]["world_size"], &report["parallel"]["ranks"], &report["parallel"]["sharding"]), (&json!(2), &json!([0, 2]), &json!("gradients")), "{report}");
    assert!(report["checkpoint_digest"].as_str().is_some() && report["weights_sha256"].as_str().is_some_and(|d| d.starts_with("sha256:")), "{report}");
    assert!(work_is_clean(&rig.work_root), "every rank's files are removed with the job");
    assert_reservation_returned(&rig);

    let again = run_action(&rig, FULL_ACTION, &full_spec(), params(&rig, 2, Some("gradients")), &[], &Frames::default()).expect("the repeat").outputs;
    assert_eq!(again["weights_sha256"], report["weights_sha256"], "the same request on two cards trains the same weights");

    // Unasked, a run whose replica fits its cards replicates.
    let replicated = run_action(&rig, FULL_ACTION, &full_spec(), params(&rig, 2, None), &[], &Frames::default()).expect("the replicated run").outputs;
    assert_eq!(replicated["parallel"]["sharding"], "replicated", "{replicated}");
    assert_reservation_returned(&rig);
}

#[test]
fn full_fine_tune_cards_and_sharding_are_refused_at_the_door_with_the_reason() {
    let rig = rig_qwen3_on("full-cards-door", 16 << 30, &[0, 1], false);
    let refused = |p: Value| run_action(&rig, FULL_ACTION, &full_spec(), p, &[], &Frames::default()).unwrap_err();
    let past = refused(params(&rig, 3, None));
    assert!(past.contains("cards 3") && past.contains("at most 2"), "{past}");
    let alone = refused(params(&rig, 1, Some("gradients")));
    assert!(alone.contains("two or more"), "a sharded stage needs a group: {alone}");
    assert!(refused(params(&rig, 2, Some("halves"))).contains("sharding"), "an unknown sharding is refused by name");
    let mut int8 = params(&rig, 2, Some("gradients"));
    int8["train_precision"] = json!("state=int8");
    let int8 = refused(int8);
    assert!(int8.contains("f32 AdamW"), "a precision the sharded optimizer does not hold is refused at the door: {int8}");
    let mut resumable = params(&rig, 2, None);
    resumable["resume_key"] = json!("two");
    assert!(refused(resumable).contains("resume_key"), "a full fine-tune on several cards does not resume");

    // What a check answers names the sharding and what each card is charged.
    let mut check = params(&rig, 2, Some("gradients"));
    check["validate_only"] = json!(true);
    let valid = run_action(&rig, FULL_ACTION, &full_spec(), check, &[], &Frames::default()).expect("the check").outputs;
    assert_eq!((&valid["cards"], &valid["sharding"]), (&json!(2), &json!("gradients")), "{valid}");
    assert_eq!(valid["device_bytes"], rank_bytes(&rig, brain::Sharding::Gradients, 2));
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// Cards too small for any sharding of the run refuse it at the door, with what each sharding needs and what the
/// card holds.
#[test]
fn a_run_no_sharding_fits_is_refused_with_the_numbers() {
    let rig = rig_qwen3_cards("full-cards-small", &[0, 1], 64 << 20);
    let refused = run_action(&rig, FULL_ACTION, &full_spec(), params(&rig, 2, None), &[], &Frames::default()).unwrap_err();
    assert!(refused.contains("no sharding over 2 rank(s) fits a card of 0.1 GiB") && refused.contains("replicated") && refused.contains("gradients") && refused.contains("parameters"), "{refused}");
    let asked = run_action(&rig, FULL_ACTION, &full_spec(), params(&rig, 2, Some("gradients")), &[], &Frames::default()).unwrap_err();
    assert!(asked.contains("sharding gradients over 2 rank(s) needs about"), "{asked}");
    assert!(rig.exec.residency().placements.is_empty(), "nothing refused at the door was placed");
}
