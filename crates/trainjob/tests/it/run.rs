// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real runs through the executor: the datasets arrive by digest and by blob, the steps stream while the run
//! goes, the adapter leaves by digest and by blob and is the one the trainer wrote.

use capability::CancelToken;
use residency::Job;
use serde_json::{json, Value};

use crate::support::{data_digest, invocation, plain, rig, rollouts, run, work_is_clean, Frames, BASE};

fn hex64(v: &Value) -> bool {
    v.as_str().is_some_and(|s| trainjob::Digest::parse(s).is_ok())
}

#[test]
fn a_policy_gradient_job_streams_each_step_and_hands_back_the_adapter_by_digest_and_by_blob_then_continues_it() {
    let rig = rig("run-pg", 16 << 30, true);
    let train = data_digest(&rig, &rollouts());
    let params = json!({
        "base": BASE, "arch": "qwen35", "objective": "policy_gradient", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128,
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
    let placed = rig.exec.residency();
    assert!(placed.placements.is_empty() && placed.budgets.iter().all(|b| b.used == 0), "{placed:?}");

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
    let params = json!({"base": BASE, "arch": "qwen35", "objective": "policy_gradient", "steps": 1000, "grad_accum": 1, "rank": 2, "seed": 7, "max_block": 128, "data": json!({"train": {"digest": train.hex()}}).to_string()});
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
    let placed = rig.exec.residency();
    assert!(placed.placements.is_empty() && placed.budgets.iter().all(|b| b.used == 0));
}
