// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `resume_key` contract through the executor: a stopped run continues from its exact state to the weights a
//! straight run trains, a completed one answers with its recorded result and trains nothing, one key is one run, a
//! damaged state is refused rather than restarted, and resumption needs an artifact store.

use std::sync::{Arc, Mutex};

use capability::{Blob, CancelToken, Invocation, Media, Progress};
use serde_json::{json, Value};
use trainjob::pretrain::{spec, ACTION};

use crate::pretrain::{created, pretrain_params, tool_records};
use crate::support::{assert_reservation_returned, data_digest, rig_qwen3, work_is_clean, Frames, Rig};

/// Runs `action` with `params`, cancelling it once a step frame reaches `cancel_at`; the step frames seen are returned.
fn run_until(rig: &Rig, action: &str, action_spec: &capability::ActionSpec, params: Value, cancel_at: Option<u64>) -> (Result<capability::Outcome, String>, Vec<u64>) {
    let mut inv = Invocation::new();
    inv.params = params;
    let mut inv = action_spec.validate_served(inv).expect("valid");
    let token = CancelToken::armed();
    inv.cancel = token.clone();
    let steps = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&steps);
    let result = rig.exec.run_blocking(trainjob::MODEL, action, inv, move |p: Progress| {
        if let Some(step) = p.event.as_ref().filter(|e| e["type"] == "step").and_then(|e| e["step"].as_u64()) {
            seen.lock().unwrap().push(step);
            if cancel_at == Some(step) {
                token.cancel();
            }
        }
    });
    let steps = steps.lock().unwrap().clone();
    (result, steps)
}

fn with(mut params: Value, extra: Value) -> Value {
    params.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    params
}

fn state_files(rig: &Rig) -> usize {
    std::fs::read_dir(rig.store.root().join(trainjob::runs::RUNS_DIR)).map_or(0, |d| d.filter_map(Result::ok).filter(|e| e.path().join("state.safetensors").exists()).count())
}

#[test]
fn a_stopped_pretraining_continues_exactly_and_a_completed_one_answers_without_training() {
    let rig = rig_qwen3("resume", 16 << 30);
    created(&rig, "scratch");
    let base = pretrain_params(&rig, "local/scratch", "pre-straight");
    let mut straight_params = base.clone();
    straight_params.as_object_mut().unwrap().remove("install_as");
    let (straight, _) = run_until(&rig, ACTION, &spec(), straight_params.clone(), None);
    let straight = straight.expect("a straight run").outputs;

    let keyed = with(straight_params.clone(), json!({"resume_key": "pre-k1", "checkpoint_every": 2, "install_as": "pre-1"}));
    let (stopped, steps) = run_until(&rig, ACTION, &spec(), keyed.clone(), Some(3));
    assert!(stopped.unwrap_err().contains("cancel"), "a cancelled job is reported cancelled");
    assert_eq!(steps, [1, 2, 3]);
    assert_eq!(state_files(&rig), 1, "the stopped run left its state in the store's run directory");

    let (resumed, steps) = run_until(&rig, ACTION, &spec(), keyed.clone(), None);
    let resumed = resumed.expect("the resumed run").outputs;
    assert_eq!(steps, [4, 5, 6], "it continues after the last saved step");
    assert_eq!(resumed["resume"], json!({"key": "pre-k1", "state": "continued", "resumed_at": 3}));
    assert_eq!(resumed["weights_sha256"], straight["weights_sha256"], "the resumed run trains the weights the straight run trained");
    assert_eq!(resumed["evaluations"], straight["evaluations"]);
    assert_eq!(resumed["installed_as"], "local/pre-1");
    assert_eq!(state_files(&rig), 0, "a completed run keeps no training state");

    // Completed: the recorded result, no training, no new artifact.
    let (again, steps) = run_until(&rig, ACTION, &spec(), keyed.clone(), None);
    let again = again.expect("the recorded result").outputs;
    assert!(steps.is_empty(), "nothing is trained again");
    assert_eq!(again["resume"]["state"], "completed");
    assert_eq!((&again["checkpoint_digest"], &again["weights_sha256"]), (&resumed["checkpoint_digest"], &resumed["weights_sha256"]));

    // One key is one run.
    let (other, _) = run_until(&rig, ACTION, &spec(), with(keyed.clone(), json!({"steps": 7})), None);
    assert!(other.unwrap_err().contains("names another run"));

    // A damaged state is refused, never restarted from scratch.
    let damaged = with(straight_params.clone(), json!({"resume_key": "pre-k2", "checkpoint_every": 1}));
    let (stopped, _) = run_until(&rig, ACTION, &spec(), damaged.clone(), Some(2));
    assert!(stopped.is_err());
    let dir = rig.store.root().join(trainjob::runs::RUNS_DIR).join(trainjob::store::digest_of(b"pre-k2").hex());
    std::fs::write(dir.join("state.safetensors"), b"not a training state").unwrap();
    let (refused, steps) = run_until(&rig, ACTION, &spec(), damaged, None);
    let refused = refused.unwrap_err();
    assert!(refused.contains("cannot be continued") || refused.contains("state"), "{refused}");
    assert!(steps.is_empty(), "nothing trained over a damaged state: {steps:?}");
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

#[test]
fn a_stopped_chat_full_fine_tune_and_a_repeated_init_resume_under_their_keys() {
    use trainjob::consolidate::{full_spec, FULL_ACTION};
    let rig = rig_qwen3("resume-full", 16 << 30);
    created(&rig, "scratch");
    let chat = data_digest(&rig, &tool_records()).hex();
    let params = json!({"base": "local/scratch", "steps": 4, "lr": 5e-3, "seed": 3, "max_block": 512, "thinking": false, "resume_key": "chat-k", "checkpoint_every": 1, "install_as": "chat-r", "data": json!({"train": {"digest": chat}}).to_string()});
    let (stopped, _) = run_until(&rig, FULL_ACTION, &full_spec(), params.clone(), Some(2));
    assert!(stopped.is_err());
    let (done, steps) = run_until(&rig, FULL_ACTION, &full_spec(), params.clone(), None);
    let done = done.expect("the resumed fine-tune").outputs;
    assert_eq!((steps, done["resume"]["state"].as_str(), done["installed_as"].as_str()), (vec![3, 4], Some("continued"), Some("local/chat-r")));

    // An init is idempotent under its key: the second submission is the first one's result.
    let digest = data_digest(&rig, &crate::create::corpus(400)).hex();
    let init = with(crate::create::init_params(&digest, "init-r"), json!({"resume_key": "init-k"}));
    let first = crate::support::run_action(&rig, trainjob::create::ACTION, &trainjob::create::spec(), init.clone(), &[], &Frames::default()).expect("created").outputs;
    let second = crate::support::run_action(&rig, trainjob::create::ACTION, &trainjob::create::spec(), init, &[], &Frames::default()).expect("answered").outputs;
    assert_eq!((first["resume"]["state"].as_str(), second["resume"]["state"].as_str()), (Some("fresh"), Some("completed")));
    assert_eq!(first["checkpoint_digest"], second["checkpoint_digest"]);
}

#[test]
fn resumption_settings_are_refused_where_they_cannot_hold() {
    let rig = rig_qwen3("resume-door", 16 << 30);
    let train = data_digest(&rig, &crate::create::corpus(10)).hex();
    let ask = |extra: Value| {
        let params = with(json!({"base": crate::support::BASE, "data": json!({"train": {"digest": train}}).to_string()}), extra);
        crate::support::run_action(&rig, ACTION, &spec(), params, &[], &Frames::default()).unwrap_err()
    };
    assert!(ask(json!({"checkpoint_every": 3})).contains("name a resume_key"));
    for bad in ["", "a/b", ".hidden", "x y", &"k".repeat(200)] {
        assert!(ask(json!({"resume_key": bad})).contains("resume_key"), "{bad:?}");
    }
    let bare = crate::support::rig_with_store_flag("resume-no-store", false);
    let mut inv = Invocation::new();
    inv.params = json!({"base": crate::support::BASE, "data": json!({"train": {"blob": "dataset"}}).to_string(), "resume_key": "k", "install_as": "x"});
    inv.blobs.insert("dataset".into(), Blob::new(Media::Bytes, crate::create::corpus(10)));
    let inv = spec().validate_served(inv).unwrap();
    let e = bare.exec.run_blocking(trainjob::MODEL, ACTION, inv, |_| {}).unwrap_err();
    assert!(e.contains("artifact store"), "{e}");
}

/// A LoRA `train` job of every objective resumes under its key: stopped mid-run, submitted again, it continues after
/// the last saved step and publishes the adapter, byte for byte, that an uninterrupted run publishes; submitted once
/// more, it answers with that adapter without training.
#[test]
fn a_stopped_lora_job_of_every_objective_continues_to_the_uninterrupted_adapter() {
    use crate::support::{pairs, rollouts, BASE};
    use trainjob::request::action_spec;
    let rig = rig_qwen3("resume-lora", 16 << 30);
    let cases = [
        ("sft", crate::support::plain(), json!({})),
        ("policy_gradient", rollouts(), json!({"behaviour": "start_of_run_proxy", "kl_beta": 0.05, "reference": "start"})),
        ("dpo", pairs(), json!({"beta": 0.1})),
    ];
    for (objective, records, extra) in cases {
        let train = data_digest(&rig, &records).hex();
        let base = with(json!({"base": BASE, "arch": "qwen3", "objective": objective, "steps": 5, "rank": 2, "lr": 5e-2, "seed": 11, "max_block": 128, "data": json!({"train": {"digest": train}}).to_string()}), extra);
        let (straight, steps) = run_until(&rig, trainjob::ACTION, &action_spec(), base.clone(), None);
        let straight = straight.unwrap_or_else(|e| panic!("{objective}: the straight run: {e}")).outputs;
        assert_eq!(steps, [1, 2, 3, 4, 5], "{objective}");

        let key = format!("lora-{objective}");
        let keyed = with(base.clone(), json!({"resume_key": key, "checkpoint_every": 1, "return_adapter": true}));
        let (stopped, steps) = run_until(&rig, trainjob::ACTION, &action_spec(), keyed.clone(), Some(2));
        assert!(stopped.unwrap_err().contains("cancel"), "{objective}: a cancelled job is reported cancelled");
        assert_eq!(steps, [1, 2], "{objective}");
        assert_eq!(state_files(&rig), 1, "{objective}: the stopped run left its state in the store's run directory");

        let (resumed, steps) = run_until(&rig, trainjob::ACTION, &action_spec(), keyed.clone(), None);
        let resumed = resumed.unwrap_or_else(|e| panic!("{objective}: the resumed run: {e}"));
        assert_eq!(steps, [3, 4, 5], "{objective}: it continues after the last saved step");
        let report = &resumed.outputs;
        assert_eq!(report["resume"], json!({"key": key, "state": "continued", "resumed_at": 2}), "{objective}");
        assert_eq!(report["adapter_sha256"], straight["adapter_sha256"], "{objective}: the resumed run trains the adapter the straight run trained");
        assert_eq!(report["adapter_digest"], straight["adapter_digest"], "{objective}");
        assert_eq!(report["final_loss"], straight["final_loss"], "{objective}");
        assert_eq!(state_files(&rig), 0, "{objective}: a completed run keeps no training state");
        assert!(work_is_clean(&rig.work_root));

        let (again, steps) = run_until(&rig, trainjob::ACTION, &action_spec(), keyed, None);
        let again = again.unwrap_or_else(|e| panic!("{objective}: the recorded result: {e}"));
        assert!(steps.is_empty(), "{objective}: nothing is trained again");
        assert_eq!(again.outputs["resume"]["state"], "completed", "{objective}");
        assert_eq!(again.outputs["adapter_digest"], report["adapter_digest"], "{objective}");
        assert_eq!(again.blobs.get("adapter").map(|b| &b.bytes), resumed.blobs.get("adapter").map(|b| &b.bytes), "{objective}: the adapter asked back is the one trained");
    }
    assert_reservation_returned(&rig);
}
