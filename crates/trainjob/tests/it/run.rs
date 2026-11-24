// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real runs through the executor: the datasets arrive by digest and by blob, the steps stream while the run
//! goes, the adapter leaves by digest and by blob and is the one the trainer wrote.

use capability::CancelToken;
use residency::Job;
use serde_json::{json, Value};

use crate::support::{assert_reservation_returned, data_digest, invocation, pair, pairs, plain, rig, rig_qwen3, rollouts, run, work_is_clean, Frames, BASE};

fn hex64(v: &Value) -> bool {
    v.as_str().is_some_and(|s| trainjob::Digest::parse(s).is_ok())
}

/// The caller chooses the behaviour policy: rollouts that recorded none are not silently trained against the start of
/// the run.
#[test]
fn a_policy_gradient_job_over_unrecorded_rollouts_must_ask_for_the_proxy() {
    let rig = rig_qwen3("run-pg-behaviour", 16 << 30);
    let train = data_digest(&rig, &rollouts());
    let params = json!({
        "base": BASE, "arch": "qwen3", "objective": "policy_gradient", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128,
        "data": json!({"train": {"digest": train.hex()}}).to_string(),
    });
    let outcome = run(&rig, params, &[], &Frames::default());
    let why = format!("{:?}", outcome.expect_err("no recorded behaviour and no proxy asked for"));
    assert!(why.contains("start_of_run_proxy"), "{why}");
}

/// A plain `qwen3` base is served the same job: the objective belongs to the shared fine-tune path, not to one family.
#[test]
fn a_policy_gradient_job_trains_a_qwen3_base() {
    let rig = rig_qwen3("run-pg-qwen3", 16 << 30);
    let train = data_digest(&rig, &rollouts());
    let params = json!({
        "base": BASE, "arch": "qwen3", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128,
        "data": json!({"train": {"digest": train.hex()}}).to_string(),
    });
    let outcome = run(&rig, params, &[], &Frames::default()).expect("the run");
    assert_eq!(outcome.outputs["status"], "completed");
    assert_eq!(outcome.outputs["policy_gradient"]["steps_taken"], 2);
    assert_eq!(outcome.outputs["policy_gradient"]["sequences_with_advantage"], 4);
    assert_eq!(outcome.outputs["policy_gradient"]["behaviour"], "start_of_run_proxy", "the proxy was asked for and the result says so");
}

/// A named algorithm trains, and the result records it and every setting the run trained with: the algorithm's
/// expansion (Dr. GRPO's constant is the request's `max_block`) with a setting the request names over it.
#[test]
fn every_named_algorithm_trains_and_the_result_records_the_settings_it_expanded_to() {
    let rig = rig_qwen3("run-pg-algorithms", 16 << 30);
    let train = data_digest(&rig, &rollouts());
    for algorithm in brain::PolicyGradientAlgorithm::ALL {
        let params = json!({
            "base": BASE, "arch": "qwen3", "objective": "policy_gradient", "algorithm": algorithm.name(), "behaviour": "start_of_run_proxy", "temperature": 0.9,
            "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string(),
        });
        let outcome = run(&rig, params, &[], &Frames::default()).unwrap_or_else(|e| panic!("{}: {e}", algorithm.name()));
        assert_eq!((&outcome.outputs["status"], &outcome.outputs["algorithm"]), (&json!("completed"), &json!(algorithm.name())));
        let settings = &outcome.outputs["policy_gradient"]["settings"];
        let want = brain::PolicyGradientAlgorithm::config(algorithm, 128);
        assert_eq!(settings["loss_agg"], json!(want.loss.loss_agg.to_string()), "{}: {settings}", algorithm.name());
        assert_eq!((settings["advantage_norm"].as_str(), settings["truncated"].as_str()), (Some(want.loss.advantage_norm.name()), Some(want.truncated.name())), "{}", algorithm.name());
        assert_eq!((settings["kl_beta"].as_f64(), settings["clip_eps_high"].as_f64()), (Some(f64::from(want.loss.kl_beta)), want.loss.clip_eps_high.map(f64::from)), "{}", algorithm.name());
        assert_eq!((settings["behaviour"].as_str(), settings["temperature"].as_f64().map(|t| (t * 1e6).round() / 1e6)), (Some("start_of_run_proxy"), Some(0.9)), "the request's own settings are laid over the algorithm's");
        assert_eq!(outcome.outputs["policy_gradient"]["steps_taken"], 2);
    }
    let named = json!({
        "base": BASE, "arch": "qwen3", "objective": "policy_gradient", "algorithm": "dapo", "clip_eps_high": 0.4, "behaviour": "start_of_run_proxy",
        "steps": 1, "rank": 2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string(),
    });
    let outcome = run(&rig, named, &[], &Frames::default()).expect("the run");
    assert_eq!(outcome.outputs["policy_gradient"]["settings"]["clip_eps_high"].as_f64().map(|v| (v * 1e6).round() / 1e6), Some(0.4), "a named setting overrides the algorithm's");
}

/// A baseline-only record of a group is read with the rollouts: it takes part in its group and trains nothing, so the
/// job trains the same four rollouts and the dry run counts it as a valid record of no example.
#[test]
fn a_baseline_only_record_sets_its_groups_baseline_and_trains_nothing() {
    let rig = rig_qwen3("run-pg-baseline", 16 << 30);
    let baseline = json!({"reward": 0.0, "group": "toy", "rollout": "reference", "baseline_only": true, "messages": [{"role": "user", "content": "cab", "train": false}, {"role": "assistant", "content": "wig", "train": false}]}).to_string();
    let dataset = [rollouts(), b"\n".to_vec(), baseline.into_bytes()].concat();
    let train = data_digest(&rig, &dataset);
    let params = json!({
        "base": BASE, "arch": "qwen3", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128,
        "data": json!({"train": {"digest": train.hex()}}).to_string(),
    });
    let outcome = run(&rig, params, &[], &Frames::default()).expect("the run");
    assert_eq!((outcome.outputs["status"].as_str(), outcome.outputs["policy_gradient"]["sequences_with_advantage"].as_u64()), (Some("completed"), Some(4)), "{}", outcome.outputs);
    let report = crate::support::run_action(&rig, trainjob::dryrun::ACTION, &trainjob::dryrun::spec(), json!({"base": BASE, "dataset_digest": train.hex()}), &[], &Frames::default()).expect("the dry run").outputs;
    assert_eq!((report["records"].as_u64(), report["valid_records"].as_u64(), report["examples"].as_u64()), (Some(5), Some(5), Some(4)), "{report}");
}

#[test]
fn a_policy_gradient_job_streams_each_step_and_hands_back_the_adapter_by_digest_and_by_blob_then_continues_it() {
    let rig = rig("run-pg", 16 << 30, true);
    let train = data_digest(&rig, &rollouts());
    let params = json!({
        "base": BASE, "arch": "qwen35", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128,
        "data": json!({"train": {"digest": train.hex()}}).to_string(), "return_adapter": true,
    });
    let frames = Frames::default();
    let outcome = run(&rig, params.clone(), &[], &frames).expect("the run");

    // The steps arrived as they were taken, each with what the objective measured.
    let steps = frames.steps();
    assert_eq!(steps.len(), 2, "{:?}", frames.events());
    for (i, frame) in steps.iter().enumerate() {
        assert_eq!((frame["step"].as_u64(), frame["steps"].as_u64()), (Some(i as u64 + 1), Some(2)));
        assert!(frame["loss"].is_number() && frame["lr"].is_number() && frame["seconds"].is_number(), "{frame}");
        assert!(frame["pg"].get("token_advantage_mean").is_none(), "the teacher-advantage statistics exist only for records with a teacher context");
        for stat in ["loss", "clip_fraction", "kl", "ratio_mean", "ratio_min", "ratio_max", "mean_advantage", "scoring_seconds"] {
            assert!(frame["pg"][stat].is_number(), "step {} carries no pg.{stat}: {frame}", i + 1);
        }
    }
    assert_eq!(frames.events().iter().filter(|e| e["type"] == "stage").count(), 2, "staging, then loading and training");

    // The report: where the adapter is, and the whole series of the objective's statistics.
    let report = &outcome.outputs;
    assert_eq!(report["status"], "completed");
    assert!(hex64(&report["adapter_digest"]), "{report}");
    assert_eq!((report["steps_completed"].as_u64(), report["records"].as_u64(), report["block"].as_u64().is_some()), (Some(2), Some(4), true));
    assert_eq!(report["policy_gradient"]["steps_taken"], 2);
    assert!(report["policy_gradient"].get("steps").is_none(), "the series is not in the report: it rides one frame");
    assert_eq!(report["policy_gradient"]["sequences_with_advantage"], 4);
    assert!(report["adapter_sha256"].as_str().is_some_and(|s| s.starts_with("sha256:")), "the digest brain serves the adapter under");

    // The adapter in the store is the adapter returned as a blob, and it hashes to its name.
    let digest = trainjob::Digest::parse(report["adapter_digest"].as_str().unwrap()).unwrap();
    let blob = outcome.blobs.get("adapter").expect("the adapter blob was asked for");
    let stored = std::fs::read(rig.store.root().join(digest.hex())).expect("published to the store");
    assert_eq!(stored, blob.bytes);
    assert_eq!(trainjob::store::digest_of(&stored), digest);
    let record: Value = serde_json::from_slice(&outcome.blobs.get("record").expect("the training record, returned as a blob because the adapter was").bytes).unwrap();
    assert_eq!(record["policy_gradient"]["steps"].as_array().map(Vec::len), Some(2));
    let record_digest = trainjob::Digest::parse(report["record_digest"].as_str().expect("and published to the store")).unwrap();
    assert_eq!(std::fs::read(rig.store.root().join(record_digest.hex())).unwrap(), outcome.blobs["record"].bytes);

    // The run is over: its files are gone and its reservation is back.
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);

    // The adapter is the starting point of the next run, named by its digest.
    let mut next = params;
    next["steps"] = json!(1);
    next["data"] = json!(json!({"train": {"digest": train.hex()}, "start_adapter": {"digest": digest.hex()}}).to_string());
    next["return_adapter"] = json!(false);
    let continued = run(&rig, next, &[], &Frames::default()).expect("the continued run");
    assert_eq!(continued.outputs["trained_from"], report["adapter_sha256"], "the new adapter records its parent");
    assert!(!continued.blobs.contains_key("adapter"), "an adapter nobody asked back is only in the store");
    assert!(hex64(&continued.outputs["adapter_digest"]));
}

/// A preference job reads pairs, trains against a frozen reference, and reports the preference margin it moved: the
/// architecture is the base's own (none was named), and the adapter leaves like any other job's.
#[test]
fn a_dpo_job_trains_a_qwen3_base_on_pairs_and_reports_how_much_more_it_prefers_the_chosen_answers() {
    let rig = rig_qwen3("run-dpo", 16 << 30);
    let train = data_digest(&rig, &pairs());
    let data = json!({"train": {"digest": train.hex()}, "held_out": {"blob": "held_out"}}).to_string();
    let params = json!({"base": BASE, "objective": "preference", "beta": 0.1, "steps": 6, "rank": 2, "lr": 1e-1, "seed": 7, "max_block": 128, "data": data, "return_adapter": true});
    let frames = Frames::default();
    let held_out = pair("cad", "bed", "kiwi").into_bytes();
    let outcome = run(&rig, params, &[("held_out", held_out)], &frames).expect("the run");

    assert_eq!(frames.steps().len(), 6, "{:?}", frames.events());
    assert!(frames.steps().iter().all(|f| f["loss"].is_number() && f.get("pg").is_none()));
    let report = &outcome.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str()), (Some("completed"), Some("preference")));
    assert_eq!((report["records"].as_u64(), report["steps_completed"].as_u64(), report["beta"].as_f64().map(|b| (b * 10.0).round())), (Some(4), Some(6), Some(1.0)));
    assert!(hex64(&report["adapter_digest"]) && hex64(&report["record_digest"]), "{report}");
    assert!(report["adapter_sha256"].as_str().is_some_and(|s| s.starts_with("sha256:")));
    let initial = report["initial_loss"].as_f64().unwrap();
    assert!((initial - std::f64::consts::LN_2).abs() < 1e-3, "the policy starts at its reference, so the loss starts at ln 2: {initial}");
    assert!(report["final_loss"].as_f64().unwrap() < initial, "the loss fell: {report}");
    assert_eq!(report["train_score"]["pairs"], 4);
    assert!(report["train_score"]["mean_margin"].as_f64().unwrap() > 0.0, "the tuned policy prefers the chosen answers more than its reference does: {report}");
    assert_eq!(report["held_out_score"]["pairs"], 1);
    assert!(outcome.blobs.contains_key("adapter") && outcome.blobs.contains_key("record"));
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// A feedback (KTO) job reads judged completions, trains against the frozen start, and reports the examples by
/// judgement and where each kind ended against the start: the desirable completions above the undesirable ones.
#[test]
fn a_feedback_job_trains_a_qwen3_base_on_judged_completions() {
    let rig = rig_qwen3("run-feedback", 16 << 30);
    let train = data_digest(&rig, &crate::support::feedback());
    let params = json!({"base": BASE, "objective": "feedback", "beta": 0.1, "steps": 12, "grad_accum": 2, "rank": 2, "lr": 5e-2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});
    let frames = Frames::default();
    let outcome = run(&rig, params, &[], &frames).expect("the run");
    assert_eq!(frames.steps().len(), 12);
    let report = &outcome.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str()), (Some("completed"), Some("feedback")));
    assert_eq!((report["desirable_examples"].as_u64(), report["undesirable_examples"].as_u64()), (Some(2), Some(2)));
    let initial = report["initial_loss"].as_f64().unwrap();
    assert!((initial - 0.5).abs() < 1e-3, "at the start every term is 1 - sigmoid(0): {initial}");
    assert!(report["final_loss"].as_f64().unwrap() < initial, "{report}");
    assert!(report["desirable_logratio"].as_f64().unwrap() > report["undesirable_logratio"].as_f64().unwrap(), "{report}");
    assert!(hex64(&report["adapter_digest"]), "{report}");
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// A distillation job takes its teacher from the artifact store by digest: an adapter an sft job published teaches a
/// fresh adapter of the same base, whose cross-entropy to it falls; the result names the teacher by its digest.
#[test]
fn a_distill_job_copies_an_adapter_of_the_store_into_a_fresh_one() {
    let rig = rig_qwen3("run-distill", 16 << 30);
    let train = data_digest(&rig, &crate::support::plain());
    let data = json!({"train": {"digest": train.hex()}}).to_string();
    let sft = run(&rig, json!({"base": BASE, "steps": 20, "rank": 4, "lr": 2e-2, "seed": 3, "max_block": 128, "data": data}), &[], &Frames::default()).expect("the teacher's run");
    let teacher = sft.outputs["adapter_digest"].as_str().expect("published").to_string();
    let params = json!({"base": BASE, "objective": "distill", "teacher": format!("adapter:{teacher}"), "top_k": 4, "steps": 20, "rank": 4, "lr": 2e-2, "seed": 7, "max_block": 128, "data": data});
    let outcome = run(&rig, params, &[], &Frames::default()).expect("the distillation");
    let report = &outcome.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str(), report["teacher"].as_str(), report["top_k"].as_u64()), (Some("completed"), Some("distill"), Some("adapter"), Some(4)), "{report}");
    assert_eq!(report["teacher_adapter_sha256"], sft.outputs["adapter_sha256"], "the teacher is the published adapter");
    assert!(report["final_loss"].as_f64().unwrap() < report["initial_loss"].as_f64().unwrap(), "{report}");
    assert!(hex64(&report["adapter_digest"]), "{report}");
    let missing = json!({"base": BASE, "objective": "distill", "teacher": format!("adapter:{}", "ab".repeat(32)), "steps": 1, "max_block": 128, "data": data});
    assert!(run(&rig, missing, &[], &Frames::default()).is_err(), "a teacher the store does not hold is refused");
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

/// Every pair loss beside DPO trains through the same job: the margin over the pairs moves toward the chosen answers,
/// and the result names the loss.
#[test]
fn a_preference_job_trains_with_every_pair_loss() {
    let rig = rig_qwen3("run-pref-losses", 16 << 30);
    let train = data_digest(&rig, &pairs());
    for (loss, extra) in [("simpo", json!({"beta": 2.0, "gamma": 0.5})), ("orpo", json!({"beta": 1.0})), ("ipo", json!({"beta": 0.5}))] {
        let mut params = json!({"base": BASE, "objective": "preference", "loss": loss, "steps": 20, "rank": 2, "lr": 1e-1, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});
        params.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let report = run(&rig, params, &[], &Frames::default()).unwrap_or_else(|e| panic!("{loss}: {e}")).outputs;
        assert_eq!((report["objective"].as_str(), report["loss"].as_str()), (Some("preference"), Some(loss)), "{report}");
        // A step's loss is one pair's, so the margin over every pair is the measure.
        // ORPO's supervised term raises what the two answers share (the end of the turn) as well, so its margin to the
        // start is not its measure here; that it learns its own objective is held in the SDK's and rl's tests.
        if loss != "orpo" {
            assert!(report["train_score"]["mean_margin"].as_f64().unwrap() > 0.0, "{loss}: {report}");
        }
        assert!(report["final_loss"].as_f64().is_some_and(f64::is_finite), "{loss}: {report}");
    }
    assert_reservation_returned(&rig);
}

/// A preference dataset is checked against the base's own tokenizer and template before a card is claimed, and a
/// record that does not parse is refused by line.
#[test]
fn a_dpo_job_refuses_a_bad_pair_by_record_and_a_record_longer_than_max_block() {
    let rig = rig_qwen3("run-dpo-refuse", 16 << 30);
    let broken = [pair("cab", "bad", "wig"), r#"{"prompt": [], "chosen": {"role": "assistant", "content": "x"}}"#.to_string()].join("\n").into_bytes();
    let params = |block: u32| json!({"base": BASE, "objective": "preference", "steps": 1, "rank": 2, "max_block": block, "data": json!({"train": {"blob": "dataset"}}).to_string(), "return_adapter": true});
    let why = run(&rig, params(128), &[("dataset", broken)], &Frames::default()).expect_err("the second record is malformed");
    assert!(why.contains("record 2") || why.contains(":2"), "{why}");
    let long = pair("cab", &"b".repeat(200), "wig").into_bytes();
    let why = run(&rig, params(64), &[("dataset", long)], &Frames::default()).expect_err("the record is past max_block");
    assert!(why.contains("max_block 64"), "{why}");
    assert!(work_is_clean(&rig.work_root));
}

#[test]
fn a_supervised_job_without_a_store_takes_its_datasets_as_blobs_scores_base_and_tuned_and_returns_the_adapter_as_a_blob() {
    let rig = rig("run-sft", 16 << 30, false);
    let data = json!({"train": {"blob": "dataset"}, "held_out": {"blob": "held_out"}}).to_string();
    let params = json!({"base": BASE, "arch": "qwen35", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "eval_every": 1, "data": data, "return_adapter": true, "adapter_id": "local/toy:local:chat:cand-1"});
    let frames = Frames::default();
    let outcome = run(&rig, params, &[("dataset", plain()), ("held_out", plain())], &frames).expect("the run");

    let report = &outcome.outputs;
    assert_eq!(frames.steps().len(), 2);
    assert!(frames.steps().iter().all(|f| f.get("pg").is_none()), "a supervised step carries no policy-gradient statistics");
    assert!(report["adapter_digest"].is_null(), "no store, so no digest to name it by");
    for score in ["base_score", "tuned_score"] {
        assert_eq!(report[score]["records"], 4, "{score}: {report}");
        assert!(report[score]["loss"].is_number());
    }
    assert!(!report["curve"].as_array().unwrap().is_empty(), "eval_every 1 monitors as it trains");
    assert_eq!(report["selection"], "best_monitor_loss");
    assert!(report["policy_gradient"].is_null());
    let adapter = &outcome.blobs.get("adapter").expect("the adapter").bytes;
    assert!(!adapter.is_empty());
    let file = rig.dir.join("returned.safetensors");
    std::fs::write(&file, adapter).unwrap();
    let card = checkpoint::st::read_card(file.to_str().unwrap()).unwrap();
    assert!(card.is_some_and(|c| c.id == "local/toy:local:chat:cand-1"), "the adapter carries the id it was asked to");
    assert!(work_is_clean(&rig.work_root));
}

#[test]
fn cancelling_a_run_stops_it_at_a_step_boundary_and_frees_everything() {
    let rig = rig("run-cancel", 16 << 30, true);
    let train = data_digest(&rig, &rollouts());
    let params = json!({"base": BASE, "arch": "qwen35", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 1000, "grad_accum": 1, "rank": 2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});
    let mut inv = invocation(params, &[]).unwrap();
    inv.cancel = CancelToken::armed();
    let token = inv.cancel.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let steps_seen = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = steps_seen.clone();
    rig.exec.submit(
        Job::new(trainjob::MODEL, trainjob::ACTION, inv)
            .on_progress(move |p| {
                if p.event.as_ref().is_some_and(|e| e["type"] == "step") {
                    seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    token.cancel();
                }
            })
            .reply(move |r| {
                let _ = tx.send(r);
            }),
    );
    let reply = rx.recv_timeout(std::time::Duration::from_secs(600)).expect("the run ended");
    assert_eq!(reply.unwrap_err(), "cancelled");
    assert!(steps_seen.load(std::sync::atomic::Ordering::SeqCst) <= 2, "it stopped at the next step boundary, not at step 1000");
    assert!(work_is_clean(&rig.work_root), "a cancelled run leaves no state behind");
    assert_reservation_returned(&rig);
}
