// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training one model across the
// servers of a fleet for its clients. If your team needs expertise in
// distributed fine-tuning as a service then you can procure our services by
// sending an email to info@swedishembedded.com.

//! One training job spread over several servers (a `train` request's `job`): each server runs its range of the
//! plan's ranks on its own cards, joined over the job's sealed transport, and the server of rank 0 publishes the run.
//! Two servers in this process stand for two machines; they meet over loopback. The suite computes on the CPU
//! backend, so the ranks reduce through the transport; on CUDA cards they form one NCCL group.

use std::net::TcpListener;

use serde_json::{json, Value};
use trainjob::job::JobAccess;

use crate::support::{assert_reservation_returned, data_digest, plain, rig_qwen3_on, rig_qwen3_serving_jobs, run, Frames, Rig, BASE};

/// A loopback address no one listens on now.
fn free_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

fn server(tag: &str, secret: &brain::JobSecret, coordinator: Option<&str>) -> Rig {
    rig_qwen3_serving_jobs(tag, 16 << 30, &[0, 1], false, Some(JobAccess { secret: secret.clone(), coordinator: coordinator.map(str::to_string) }))
}

fn params(rig: &Rig, cards: u32) -> Value {
    let train = data_digest(rig, &plain());
    json!({"base": BASE, "arch": "qwen3", "steps": 3, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "cards": cards, "data": json!({"train": {"digest": train.hex()}}).to_string()})
}

fn plan(world: usize, coordinator: &str, handshake_ms: u64) -> Value {
    json!({"job": "fleet-test", "world_size": world, "tp": 1, "pp": 1, "dp": world, "coordinator": coordinator, "handshake_timeout_ms": handshake_ms, "op_timeout_ms": 60000})
}

fn with_job(mut params: Value, plan: &Value, first_rank: usize, digest: &str) -> Value {
    params["job"] = json!(json!({"plan": plan, "first_rank": first_rank, "secret_sha256": digest}).to_string());
    params
}

#[test]
fn a_job_over_two_servers_trains_the_adapter_three_cards_of_one_server_train_and_rank_zeros_server_publishes_it() {
    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let first = server("fleet-a", &secret, Some(&coordinator));
    let second = server("fleet-b", &secret, None);
    // Uneven parts: two cards on one server, one on the other.
    let plan = plan(3, &coordinator, 60_000);
    let digest = secret.digest_hex();
    let (a, b) = (with_job(params(&first, 2), &plan, 0, &digest), with_job(params(&second, 1), &plan, 2, &digest));
    let (published, worker) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| run(&second, b, &[], &Frames::default()));
        let published = run(&first, a, &[], &Frames::default());
        (published, worker.join().unwrap())
    });
    let published = published.expect("rank 0's server").outputs;
    let worker = worker.expect("the other server").outputs;
    assert_eq!(published["status"], "completed");
    assert!(published["adapter_digest"].as_str().is_some(), "rank 0's server publishes the adapter: {published}");
    assert!(published["adapter_sha256"].as_str().is_some_and(|d| d.starts_with("sha256:")), "{published}");
    assert_eq!(worker, json!({"status": "completed", "objective": "sft", "job": "fleet-test", "ranks": [2, 2], "adapter_sha256": published["adapter_sha256"]}), "the other server answers for its ranks only, with the same adapter");
    assert_reservation_returned(&first);
    assert_reservation_returned(&second);

    // The same three ranks as one server's three cards: the same records per rank, the same rank-ordered sums.
    let three = rig_qwen3_on("fleet-three", 16 << 30, &[0, 1, 2], false);
    let alone = run(&three, params(&three, 3), &[], &Frames::default()).expect("three cards of one server").outputs;
    assert_eq!(alone["adapter_sha256"], published["adapter_sha256"], "one job over two servers is the job three cards of one server run");
}

#[test]
fn a_server_whose_peers_never_join_fails_the_job_within_the_plans_handshake_and_gives_its_cards_back() {
    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let second = server("fleet-alone", &secret, None);
    let started = std::time::Instant::now();
    let failed = run(&second, with_job(params(&second, 2), &plan(4, &coordinator, 2_000), 2, &secret.digest_hex()), &[], &Frames::default()).unwrap_err();
    assert!(failed.contains("coordinator") || failed.contains("join"), "{failed}");
    assert!(started.elapsed() < std::time::Duration::from_secs(60), "bounded by the plan's handshake: {:?}", started.elapsed());
    assert_reservation_returned(&second);
}

#[test]
fn a_job_is_refused_at_the_door_for_a_secret_a_coordinator_or_a_plan_this_server_does_not_hold() {
    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let first = server("fleet-door", &secret, Some(&coordinator));
    let digest = secret.digest_hex();
    let plan4 = plan(4, &coordinator, 60_000);
    let refused = |params: Value| run(&first, params, &[], &Frames::default()).unwrap_err();

    let wrong = refused(with_job(params(&first, 2), &plan4, 2, &brain::JobSecret::random().digest_hex()));
    assert!(wrong.contains("not the digest of this server's job secret"), "{wrong}");
    assert!(!wrong.contains(&secret.to_hex()) && !wrong.contains(&digest), "no secret material is echoed: {wrong}");
    let elsewhere = refused(with_job(params(&first, 2), &plan(4, &free_address(), 60_000), 0, &digest));
    assert!(elsewhere.contains("--train-job-coordinator"), "{elsewhere}");
    let outside = refused(with_job(params(&first, 2), &plan4, 3, &digest));
    assert!(outside.contains("outside the plan's 4 ranks"), "{outside}");
    let mut tp = plan4.clone();
    (tp["tp"], tp["dp"]) = (json!(2), json!(2));
    assert!(refused(with_job(params(&first, 2), &tp, 2, &digest)).contains("tp and pp must be 1"));
    let mut unknown = plan4.clone();
    unknown["extra"] = json!(1);
    assert!(refused(with_job(params(&first, 2), &unknown, 2, &digest)).contains("unknown field"));
    let mut no_seed = with_job(params(&first, 2), &plan4, 2, &digest);
    no_seed.as_object_mut().unwrap().remove("seed");
    assert!(refused(no_seed).contains("job needs seed"));
    let mut huge = params(&first, 2);
    huge["job"] = json!("x".repeat(trainjob::parallel::MAX_JOB_DOCUMENT_BYTES + 1));
    assert!(refused(huge).contains("at most"));

    let none = rig_qwen3_on("fleet-none", 16 << 30, &[0, 1], false);
    let off = run(&none, with_job(params(&none, 2), &plan4, 2, &digest), &[], &Frames::default()).unwrap_err();
    assert!(off.contains("--train-job-secret"), "{off}");
    assert_reservation_returned(&first);
}

/// A resumable job over two servers stopped part way (one server's run cancelled, which fails the job on both)
/// resumes when both servers are sent the same requests again: each server continues its own ranks from the newest
/// generation every rank committed, kept in its own artifact store, and the job trains the uninterrupted job's
/// adapter, byte for byte. Once complete, each server answers the same requests from its record without joining.
#[test]
fn a_stopped_job_over_two_servers_resumes_on_both_to_the_uninterrupted_adapter() {
    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let first = server("fleet-resume-a", &secret, Some(&coordinator));
    let second = server("fleet-resume-b", &secret, None);
    let plan = plan(2, &coordinator, 60_000);
    let digest = secret.digest_hex();
    let request = |rig: &Rig, first_rank: usize, key: Option<&str>| {
        let mut p = with_job(params(rig, 1), &plan, first_rank, &digest);
        p["steps"] = json!(5);
        if let Some(key) = key {
            (p["resume_key"], p["checkpoint_every"]) = (json!(key), json!(1));
        }
        p
    };
    let both = |a: &dyn Fn() -> Result<Value, String>, b: &(dyn Fn() -> Result<Value, String> + Sync)| {
        std::thread::scope(|scope| {
            let worker = scope.spawn(b);
            let published = a();
            (published, worker.join().unwrap())
        })
    };
    let outputs = |rig: &Rig, p: Value| run(rig, p, &[], &Frames::default()).map(|o| o.outputs);
    let (straight, _) = both(&|| outputs(&first, request(&first, 0, None)), &|| outputs(&second, request(&second, 1, None)));
    let straight = straight.expect("the uninterrupted job");

    // Server A's run is cancelled at its first step: the job fails on both servers.
    let stop_a = || crate::cards::run_action_observed(&first, trainjob::ACTION, crate::support::invocation(request(&first, 0, Some("fleet")), &[]).unwrap(), true).result.map(|o| o.outputs);
    let (a, b) = both(&stop_a, &|| outputs(&second, request(&second, 1, Some("fleet"))));
    assert!(a.is_err() && b.is_err(), "the job stopped on both servers: {a:?} {b:?}");
    assert_reservation_returned(&first);
    assert_reservation_returned(&second);

    let (a, b) = both(&|| outputs(&first, request(&first, 0, Some("fleet"))), &|| outputs(&second, request(&second, 1, Some("fleet"))));
    let (a, b) = (a.expect("server A resumes"), b.expect("server B resumes"));
    assert_eq!(a["resume"]["state"], "continued", "{a}");
    assert_eq!(b["resume"], a["resume"], "both servers continued from the same generation: {b}");
    assert_eq!(a["adapter_sha256"], straight["adapter_sha256"], "the resumed job trains the uninterrupted job's adapter");
    assert_eq!(b["adapter_sha256"], straight["adapter_sha256"]);

    // Complete: each server answers from its record, alone.
    let again = outputs(&second, request(&second, 1, Some("fleet"))).expect("server B's recorded answer");
    assert_eq!((&again["adapter_sha256"], &again["ranks"]), (&b["adapter_sha256"], &json!([1, 1])), "{again}");
    let again = outputs(&first, request(&first, 0, Some("fleet"))).expect("server A's recorded result");
    assert_eq!((&again["resume"]["state"], &again["adapter_digest"]), (&json!("completed"), &a["adapter_digest"]), "{again}");
    assert_reservation_returned(&first);
    assert_reservation_returned(&second);
}

/// A full fine-tune (`full_train`) spread over two servers the same way: the plan's sharding is the run's, rank 0's
/// server publishes the checkpoint, the other answers for its ranks, and the weights are those two cards of one
/// server train under the same sharding.
#[test]
fn a_full_fine_tune_over_two_servers_shards_as_its_plan_says_and_trains_what_one_server_trains() {
    use trainjob::consolidate::{full_spec, FULL_ACTION};
    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let first = server("fleet-full-a", &secret, Some(&coordinator));
    let second = server("fleet-full-b", &secret, None);
    let full = |rig: &Rig, cards: u32| {
        let train = data_digest(rig, &plain());
        json!({"base": BASE, "steps": 2, "lr": 1e-3, "seed": 7, "max_block": 128, "cards": cards, "data": json!({"train": {"digest": train.hex()}}).to_string()})
    };
    let mut plan = plan(2, &coordinator, 60_000);
    plan["sharding"] = json!("gradients");
    let digest = secret.digest_hex();
    let (a, b) = (with_job(full(&first, 1), &plan, 0, &digest), with_job(full(&second, 1), &plan, 1, &digest));
    let run_full = |rig: &Rig, params: Value| crate::support::run_action(rig, FULL_ACTION, &full_spec(), params, &[], &Frames::default());
    let (published, worker) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| run_full(&second, b));
        let published = run_full(&first, a);
        (published, worker.join().unwrap())
    });
    let published = published.expect("rank 0's server").outputs;
    let worker = worker.expect("the other server").outputs;
    assert!(published["checkpoint_digest"].as_str().is_some(), "rank 0's server publishes the checkpoint: {published}");
    assert_eq!((&published["parallel"]["world_size"], &published["parallel"]["sharding"]), (&json!(2), &json!("gradients")), "{published}");
    assert_eq!(worker, json!({"status": "completed", "objective": "full_sft", "job": "fleet-test", "ranks": [1, 1]}), "the other server answers for its ranks only");
    assert_reservation_returned(&first);
    assert_reservation_returned(&second);

    let stranger = run_full(&second, with_job(full(&second, 1), &plan, 1, &brain::JobSecret::random().digest_hex())).unwrap_err();
    assert!(stranger.contains("full_train: job: secret_sha256 is not the digest of this server's job secret"), "{stranger}");
    let mut mismatch = with_job(full(&first, 1), &plan, 0, &digest);
    mismatch["sharding"] = json!("optimizer");
    assert!(run_full(&first, mismatch).unwrap_err().contains("plan"), "a request cannot shard other than its job's plan");

    let two = rig_qwen3_on("fleet-full-two", 16 << 30, &[0, 1], false);
    let mut alone = full(&two, 2);
    alone["sharding"] = json!("gradients");
    let alone = run_full(&two, alone).expect("two cards of one server").outputs;
    assert_eq!(alone["weights_sha256"], published["weights_sha256"], "one job over two servers trains what two cards of one server train");
}

