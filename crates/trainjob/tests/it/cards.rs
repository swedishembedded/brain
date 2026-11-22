// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training on several of a
// server's cards from one served request for its clients. If your team needs
// expertise in data-parallel fine-tuning as a service then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A served `train` job on several cards of its server (`cards`): one data-parallel rank per card, the cards held
//! together while it runs and given back however it ends, the same report as a run on one card, and the same adapter
//! from the same request.
//!
//! The server here is given two cards and computes on the CPU backend (the test suite's `BRAIN_DEVICE=cpu`), so every
//! rank trains on the host while the residency accounts for the cards it was placed on.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use capability::CancelToken;
use residency::{Device, Job, ResidencyReport};
use serde_json::{json, Value};

use crate::support::{assert_reservation_returned, data_digest, invocation, pairs, plain, rig_qwen3_on, Rig, BASE};

fn two_cards(tag: &str) -> Rig {
    rig_qwen3_on(tag, 16 << 30, &[0, 1], false)
}

fn params(rig: &Rig, cards: u32, steps: u32) -> Value {
    let train = data_digest(rig, &plain());
    json!({"base": BASE, "arch": "qwen3", "steps": steps, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "cards": cards, "return_adapter": true, "data": json!({"train": {"digest": train.hex()}}).to_string()})
}

/// What a run held while its first step was reported, and how it answered.
pub struct Observed {
    pub result: Result<capability::Outcome, String>,
    pub steps: usize,
    pub held: Option<ResidencyReport>,
}

fn run_observed(rig: &Rig, params: Value, cancel_at_first_step: bool) -> Observed {
    run_action_observed(rig, trainjob::ACTION, invocation(params, &[]).unwrap(), cancel_at_first_step)
}

/// Runs the validated invocation `inv` of `action`, noting what was held at its first step and cancelling there when
/// asked.
pub fn run_action_observed(rig: &Rig, action: &'static str, mut inv: capability::Invocation, cancel_at_first_step: bool) -> Observed {
    inv.cancel = CancelToken::armed();
    let token = inv.cancel.clone();
    let held: Arc<Mutex<Option<ResidencyReport>>> = Arc::default();
    let steps = Arc::new(Mutex::new(0usize));
    let (exec, seen, counted) = (rig.exec.clone(), held.clone(), steps.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    rig.exec.submit(
        Job::new(trainjob::MODEL, action, inv)
            .on_progress(move |p| {
                if p.event.as_ref().is_some_and(|e| e["type"] == "step") {
                    *counted.lock().unwrap() += 1;
                    let mut seen = seen.lock().unwrap();
                    if seen.is_none() {
                        *seen = Some(exec.residency());
                    }
                    if cancel_at_first_step {
                        token.cancel();
                    }
                }
            })
            .reply(move |r| {
                let _ = tx.send(r);
            }),
    );
    let result = rx.recv_timeout(std::time::Duration::from_secs(600)).expect("the run ended");
    let steps = *steps.lock().unwrap();
    let held = held.lock().unwrap().take();
    Observed { result, steps, held }
}

/// The cards a running job was charged on, from a residency report.
pub fn cards_charged(report: &ResidencyReport) -> BTreeSet<u32> {
    report.budgets.iter().filter(|b| b.used > 0).filter_map(|b| if let Device::Gpu(i) = b.device { Some(i) } else { None }).collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().expect("an object").keys().cloned().collect()
}

#[test]
fn a_run_on_two_cards_holds_both_and_answers_with_the_one_card_report_and_a_repeatable_adapter() {
    let rig = two_cards("cards-two");
    let one = run_observed(&rig, params(&rig, 1, 2), false);
    let one_outcome = one.result.expect("the one-card run");
    assert_eq!(cards_charged(&one.held.expect("a step was reported")).len(), 1, "a one-card run holds one card");
    assert_reservation_returned(&rig);

    let two = run_observed(&rig, params(&rig, 2, 2), false);
    let held = two.held.expect("a step was reported");
    let outcome = two.result.expect("the two-card run");
    assert_eq!(cards_charged(&held), BTreeSet::from([0, 1]), "the run holds both cards: {held:?}");
    assert_eq!(held.multi_placements.len(), 1);
    assert_eq!(two.steps, 2, "rank 0 streams each step once");
    assert_eq!(keys(&outcome.outputs), keys(&one_outcome.outputs), "the same report as a run on one card");
    assert_eq!((&outcome.outputs["status"], &outcome.outputs["steps_completed"], &outcome.outputs["seed"]), (&json!("completed"), &json!(2), &json!(7)));
    assert!(outcome.outputs["adapter_digest"].as_str().is_some_and(|d| trainjob::Digest::parse(d).is_ok()));
    assert_reservation_returned(&rig);
    assert!(crate::support::work_is_clean(&rig.work_root), "every rank's files are removed with the job");

    let again = run_observed(&rig, params(&rig, 2, 2), false).result.expect("the repeat");
    let adapter = |o: &capability::Outcome| o.blobs.get("adapter").expect("the adapter").bytes.clone();
    assert_eq!(adapter(&again), adapter(&outcome), "the same request on two cards trains the same adapter, byte for byte");
    assert_eq!(again.outputs["adapter_digest"], outcome.outputs["adapter_digest"]);
}

#[test]
fn cancelling_a_run_on_two_cards_stops_every_rank_and_gives_both_cards_back() {
    let rig = two_cards("cards-cancel");
    let observed = run_observed(&rig, params(&rig, 2, 1000), true);
    assert_eq!(observed.result.unwrap_err(), "cancelled");
    assert!(observed.steps <= 3, "the ranks stopped within a step of the cancellation, not at step 1000: {}", observed.steps);
    assert!(crate::support::work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

#[test]
fn cards_are_refused_at_the_door_past_the_servers_cards_and_for_objectives_that_train_on_one() {
    let rig = two_cards("cards-door");
    let refused = crate::support::run(&rig, params(&rig, 3, 1), &[], &crate::support::Frames::default()).unwrap_err();
    assert!(refused.contains("cards 3") && refused.contains("at most 2"), "{refused}");

    let mut zero = params(&rig, 1, 1);
    zero["cards"] = json!(0);
    assert!(invocation(zero, &[]).is_err(), "the spec bounds cards below at one");

    let pairs = data_digest(&rig, &pairs());
    let preference = json!({"base": BASE, "arch": "qwen3", "objective": "preference", "steps": 1, "rank": 2, "seed": 7, "max_block": 128, "cards": 2, "data": json!({"train": {"digest": pairs.hex()}}).to_string()});
    let refused = crate::support::run(&rig, preference, &[], &crate::support::Frames::default()).unwrap_err();
    assert!(refused.contains("cards 2") && refused.contains("preference objective trains on one card"), "{refused}");
    assert_reservation_returned(&rig);
}

/// A resumable run on two cards stopped mid-way continues on two cards to the adapter an uninterrupted two-card run
/// trains, and once complete keeps no training state of any rank.
#[test]
fn a_stopped_run_on_two_cards_resumes_to_the_uninterrupted_adapter_and_leaves_no_state() {
    let rig = two_cards("cards-resume");
    let mut straight = params(&rig, 2, 5);
    straight["return_adapter"] = json!(false);
    let straight = crate::support::run(&rig, straight.clone(), &[], &crate::support::Frames::default()).expect("the straight run").outputs;

    let mut keyed = params(&rig, 2, 5);
    keyed["return_adapter"] = json!(false);
    keyed["resume_key"] = json!("two-cards");
    keyed["checkpoint_every"] = json!(1);
    let stopped = run_observed(&rig, keyed.clone(), true);
    assert_eq!(stopped.result.unwrap_err(), "cancelled");
    let resumed = crate::support::run(&rig, keyed, &[], &crate::support::Frames::default()).expect("the resumed run").outputs;
    assert_eq!(resumed["resume"]["state"], "continued", "{resumed}");
    assert_eq!(resumed["adapter_digest"], straight["adapter_digest"], "the resumed run trains the adapter the straight run trained");

    let runs = rig.store.root().join(trainjob::runs::RUNS_DIR);
    let left: Vec<String> = std::fs::read_dir(&runs).unwrap().flatten().flat_map(|run| std::fs::read_dir(run.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned())).filter(|name| name != "run.json" && name != "lock").collect();
    assert!(left.is_empty(), "a completed run keeps no training state: {left:?}");
    assert!(crate::support::work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// One rank of a run on several cards fails alone, mid-run, while its peer waits for it in the step's reduction (on
/// the cards over NCCL when the server has CUDA cards, through host memory on the CPU suite): the peer fails at once
/// with the failed rank's reason instead of at the group's op timeout (600 s), and the run fails with the reason of
/// the rank that failed first, not the "aborted" its failure caused in the other.
#[test]
fn a_rank_that_fails_alone_mid_run_fails_the_run_at_once_with_its_reason() {
    let model = crate::fixture::qwen3_model_dir_of("cards-rank-fails", qwen3::QwenConfig::tiny());
    let train = crate::fixture::write_chat_dataset(&model, &[("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab"), ("dab", "bead"), ("cafe", "face")]);
    let ranks = brain::DataParallel::in_process(2).expect("a two-rank group");
    let breakers: Vec<brain::GroupBreaker> = ranks.iter().map(brain::DataParallel::breaker).collect();
    let abort = |i: usize, why: &str| breakers[i].abort(why);
    let slots: Vec<Mutex<Option<brain::DataParallel>>> = ranks.into_iter().map(|rank| Mutex::new(Some(rank))).collect();
    let peer_error: Mutex<Option<String>> = Mutex::new(None);
    let failed_at: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let body = |r: usize, _frames: Option<std::sync::mpsc::Sender<capability::Progress>>| -> Result<(), String> {
        let rank = slots[r].lock().unwrap().take().expect("each rank runs once");
        // As the served job's ranks do: the group is broken before a failed rank lets its communicator go.
        let guard = rank.guard();
        let tune = brain::ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).placed(true).dataset(&train).out_dir(model.join(format!("rank{r}"))).rank(2).steps(1000).lr(1e-2).seed(11).data_parallel(rank);
        let outcome = tune.run_with(&CancelToken::armed(), |step| {
            // Rank 1 fails after step 1, once the group (the device group, on cards) is formed and rank 0 is
            // on its way into step 2's reduction.
            if r == 1 && step.step == 1 {
                *failed_at.lock().unwrap() = Some(std::time::Instant::now());
                panic!("rank 1 lost its card");
            }
        });
        let error = outcome.map(drop).map_err(|e| e.to_string());
        guard.finish(&error);
        if r == 0 {
            *peer_error.lock().unwrap() = error.clone().err();
        }
        error
    };
    let devices = [Device::Gpu(0), Device::Gpu(1)];
    let why = trainjob::parallel::run_ranks(&devices, &abort, &|| breakers[0].first_failed(), body, &mut |_| {}).expect_err("the run fails with its rank");
    let waited = failed_at.lock().unwrap().expect("rank 1 failed after step 1").elapsed();
    assert_eq!(why, "train: the device failed during the run (see the server log)", "the run names the first rank's failure");
    let peer = peer_error.lock().unwrap().clone().expect("rank 0 failed too");
    assert!(peer.contains("rank 1 failed"), "rank 0 fails with rank 1's reason, not a timeout: {peer}");
    assert!(waited < std::time::Duration::from_secs(30), "rank 0 failed {waited:?} after rank 1, not at once");
}
