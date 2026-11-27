// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A policy-gradient job fed while it runs (`stream` instead of `data`), end to end on the CPU backend with the toy
//! `qwen3` base served beside the training workload: groups posted to the running job are trained as they arrive, each
//! published adapter becomes the version the server serves, a stale group is dropped, a duplicate is idempotent, a full
//! queue is refused naming its allowance (never while the learner waits for a step), `finish` ends the job, and a run
//! resumed from its saved state takes again exactly the groups it had not committed.

use std::time::{Duration, Instant};

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use futures::StreamExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::support::{rig_stream, run_action, Frames, Rig, BASE};

const KEY: &str = "sk-brain-test";
pub const ADAPTER: &str = "local/toy:lab:policy:v1";

async fn call(app: &Router, method: Method, uri: &str, body: Option<(&str, Vec<u8>)>) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri).header(header::AUTHORIZATION, format!("Bearer {KEY}"));
    let body = match body {
        Some((kind, bytes)) => {
            req = req.header(header::CONTENT_TYPE, kind);
            Body::from(bytes)
        }
        None => Body::empty(),
    };
    let response = tokio::time::timeout(Duration::from_secs(600), app.clone().oneshot(req.body(body).unwrap())).await.expect("the response").unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| json!(String::from_utf8_lossy(&bytes))))
}

/// The job's frames (`frame` events of its SSE stream) after `after`, read as they come until one satisfies `until`
/// (or the stream ends); with the number of the last event read.
async fn frames_until(app: &Router, id: &str, after: u64, until: impl Fn(&Value) -> bool) -> (Vec<Value>, u64) {
    let req = Request::builder().method(Method::GET).uri(format!("/v1/jobs/{id}/events")).header(header::AUTHORIZATION, format!("Bearer {KEY}")).header("last-event-id", after.to_string()).body(Body::empty()).unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let (mut text, mut frames, mut last) = (String::new(), Vec::new(), after);
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        let Ok(Some(Ok(chunk))) = tokio::time::timeout(deadline.saturating_duration_since(Instant::now()), body.next()).await else { return (frames, last) };
        text.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = text.find("\n\n") {
            let event: String = text.drain(..end + 2).collect();
            let field = |name: &str| event.lines().find_map(|l| l.strip_prefix(name).map(|v| v.trim().to_string()));
            if let Some(seq) = field("id:").and_then(|v| v.parse().ok()) {
                last = seq;
            }
            let Some(data) = field("data:").and_then(|d| serde_json::from_str::<Value>(&d).ok()) else { continue };
            let done = field("event:").as_deref() == Some("done");
            let hit = until(&data);
            if !done {
                frames.push(data);
            }
            if hit || done {
                return (frames, last);
            }
        }
    }
}

/// Supervised tokens of each answer as the base renders it, from the `check_dataset` action: a client sizes the
/// sampler's log-probabilities without a tokenizer of its own.
fn answer_tokens(rig: &Rig, answers: &[(&str, &str)]) -> Vec<usize> {
    // As rollouts, so each answer is rendered as the model is asked for it: the tokens a sampler draws.
    let records: Vec<String> = answers.iter().enumerate().map(|(i, (p, a))| json!({"messages": [{"role": "user", "content": p, "train": false}, {"role": "assistant", "content": a, "train": true}], "tools": [], "reward": i as f64, "group": "probe"}).to_string()).collect();
    let report = run_action(rig, "check_dataset", &trainjob::dryrun::spec(), json!({"base": BASE, "per_record": true}), &[("dataset", records.join("\n").into_bytes())], &Frames::default()).expect("the report").outputs;
    report["per_record"].as_array().unwrap().iter().map(|r| r[3].as_u64().unwrap() as usize).collect()
}

/// The answers of a toy group: (prompt, answer, reward).
const PAIRS: [(&str, &str, f64); 2] = [("cab", "bad", 1.0), ("ace", "cab", -1.0)];

/// The supervised tokens of each answer of [`PAIRS`], measured before any job holds the server's one lane.
pub fn sizes(rig: &Rig) -> [usize; 2] {
    let tokens = answer_tokens(rig, &PAIRS.map(|(p, a, _)| (p, a)));
    [tokens[0], tokens[1]]
}

/// A group of two rollouts of the toy task, rewarded 1 and -1, drawn by `version`: JSONL.
pub fn group(sizes: [usize; 2], key: &str, version: u64) -> Vec<u8> {
    PAIRS
        .iter()
        .zip(sizes)
        .map(|((p, a, r), n)| json!({"messages": [{"role": "user", "content": p, "train": false}, {"role": "assistant", "content": a, "train": true, "behavior_logprobs": vec![-12.0; n]}], "tools": [], "reward": r, "group": key, "policy_version": version}).to_string())
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

fn stream_job(stream: Value, extra: Value) -> Value {
    let mut params = json!({"base": BASE, "objective": "policy_gradient", "loss_agg": "token-sum", "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "stream": stream.to_string()});
    params.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    json!({"model": trainjob::MODEL, "action": trainjob::ACTION, "params": params})
}

async fn create(app: &Router, body: Value) -> String {
    let (status, created) = call(app, Method::POST, "/v1/jobs", Some(("application/json", body.to_string().into_bytes()))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{created}");
    created["id"].as_str().unwrap().to_string()
}

async fn post(app: &Router, id: &str, query: &str, records: Vec<u8>) -> (StatusCode, Value) {
    call(app, Method::POST, &format!("/v1/jobs/{id}/records{query}"), Some(("application/x-ndjson", records))).await
}

/// The version `/v1/models` lists the adapter at, waiting (bounded) until it is at least `at_least`.
async fn served_version(app: &Router, at_least: u64) -> u64 {
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let (_, listing) = call(app, Method::GET, "/v1/models", None).await;
        let version = listing["data"].as_array().into_iter().flatten().find(|m| m["id"] == ADAPTER).and_then(|m| m["adapter"]["version"].as_u64()).unwrap_or(0);
        if version >= at_least || Instant::now() > deadline {
            return version;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The policy version that answers a chat request for the published adapter.
async fn reply_version(app: &Router) -> u64 {
    let body = json!({"model": ADAPTER, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2});
    let (status, reply) = call(app, Method::POST, "/v1/chat/completions", Some(("application/json", body.to_string().into_bytes()))).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    reply["brain"]["policy_version"].as_u64().unwrap_or_else(|| panic!("the reply names its policy version: {reply}"))
}

fn of_type<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
    frames.iter().filter(|f| f["type"] == kind).collect()
}

fn keys(frame: &Value) -> Vec<String> {
    frame["groups"].as_array().unwrap().iter().map(|g| g.as_str().unwrap().to_string()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_job_trains_groups_as_they_arrive_and_serves_each_version_it_publishes() {
    let rig = rig_stream("stream-life");
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let sizes = sizes(&rig);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let id = create(&app, stream_job(json!({"publish_adapter": ADAPTER, "publish_every": 1, "max_staleness": 0, "step_tokens": tokens, "queue_tokens": tokens}), json!({}))).await;

    // Before any group, the policy the run starts from is served as version 1: the first rollouts are drawn from it,
    // on the rig's one CPU device, which the run lends while it waits for records. A group is a step; its step
    // publishes version 2, which the server then lists and answers from.
    assert_eq!(served_version(&app, 1).await, 1, "the starting policy is served before any input");
    assert_eq!(reply_version(&app).await, 1, "a request is answered while the run waits on the same device");
    let (status, answer) = post(&app, &id, "", group(sizes, "g1", 1)).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!((answer["accepted"].clone(), answer["duplicates"].clone(), answer["epoch"].clone()), (json!(["g1"]), json!([]), json!(1)));
    assert_eq!(served_version(&app, 2).await, 2, "the first trained version is served");
    assert_eq!(reply_version(&app).await, 2, "a request after the step is answered by the version it published");
    // A group already taken is a duplicate, idempotently; a group drawn from the current version trains and publishes
    // version 3; one drawn from version 0 is behind the bound and is dropped, never trained.
    let (status, again) = post(&app, &id, "", group(sizes, "g1", 1)).await;
    assert_eq!((status, again["accepted"].clone(), again["duplicates"].clone()), (StatusCode::OK, json!([]), json!(["g1"])));
    assert_eq!(post(&app, &id, "", group(sizes, "g2", 2)).await.0, StatusCode::OK);
    assert_eq!(served_version(&app, 3).await, 3);
    let (status, stale) = post(&app, &id, "?epoch=1", group(sizes, "old", 0)).await;
    assert_eq!(status, StatusCode::OK, "{stale}");
    assert_eq!(post(&app, &id, "", b"{\"messages\": []}".to_vec()).await.0, StatusCode::BAD_REQUEST, "a record the contract refuses");

    let (status, finishing) = call(&app, Method::POST, &format!("/v1/jobs/{id}/finish"), None).await;
    assert_eq!((status, finishing["finishing"].clone()), (StatusCode::OK, json!(true)), "{finishing}");
    let (frames, _) = frames_until(&app, &id, 0, |_| false).await;
    let (status, result) = call(&app, Method::GET, &format!("/v1/jobs/{id}/result"), None).await;
    assert_eq!(status, StatusCode::OK, "{result}: {frames:?}");
    assert_eq!(post(&app, &id, "", group(sizes, "late", 2)).await.0, StatusCode::GONE, "a finished job takes no input");

    // The receipts: two steps applied (g1, g2), the stale group dropped, everything committed once the job completed,
    // each version published once its step ran, and the steps carry the stream's telemetry.
    let started = of_type(&frames, "stream");
    assert_eq!((started.len(), started[0]["epoch"].clone(), started[0]["version"].clone()), (1, json!(1), json!(0)));
    let applied: Vec<Vec<String>> = of_type(&frames, "applied").iter().map(|f| keys(f)).collect();
    assert_eq!(applied, [vec!["g1".to_string()], vec!["g2".to_string()]], "{frames:?}");
    assert_eq!(of_type(&frames, "dropped").iter().flat_map(|f| keys(f)).collect::<Vec<_>>(), ["old"]);
    let committed: Vec<String> = of_type(&frames, "committed").iter().flat_map(|f| keys(f)).collect();
    assert_eq!(committed, ["g1", "g2", "old"]);
    let published: Vec<(u64, u64)> = of_type(&frames, "published").iter().map(|f| (f["version"].as_u64().unwrap(), f["step"].as_u64().unwrap())).collect();
    assert_eq!(published, [(1, 0), (2, 1), (3, 2)], "the start, then each step; the last step was published already, so finish publishes nothing new");
    let steps = of_type(&frames, "step");
    assert_eq!(steps.len(), 2);
    for step in &steps {
        let pg = &step["pg"];
        assert_eq!(pg["policy_tokens"].as_f64(), Some(tokens as f64), "{step}");
        for field in ["staleness_mean", "staleness_max", "learner_wait_seconds", "queue_groups", "queue_tokens", "time_in_queue_seconds", "dropped_stale"] {
            assert!(pg[field].is_number(), "{field}: {step}");
        }
    }
    assert_eq!(steps[1]["pg"]["staleness_max"], json!(0.0), "g2 was drawn by the version being trained");
    assert!(result["outputs"]["adapter_digest"].is_string(), "the final adapter is in the artifact store too: {result}");

    // Each version carries the run-level identity of its weights: this run (an id drawn for a run without a resume
    // key, named in its stream frame), its epoch, the update it holds, the base it trained on and its own file.
    let run = started[0]["run"].as_str().expect("the stream frame names the run");
    assert!(started[0]["tokenizer_digest"].as_str().is_some_and(|d| d.starts_with("sha256:")), "the frame names the base's tokenizer: {}", started[0]);
    assert!(run.starts_with("job:"), "{run}");
    for frame in of_type(&frames, "published") {
        let lineage = &frame["lineage"];
        assert_eq!((lineage["run"].as_str(), lineage["epoch"].clone(), lineage["update"].clone(), lineage["adapter_digest"].clone()), (Some(run), json!(1), frame["step"].clone(), frame["digest"].clone()), "{frame}");
        assert!(lineage["base_digest"].as_str().is_some_and(|d| d.starts_with("sha256:")), "{frame}");
    }
    // The served version names it on the model card, in a reply's `brain` object and in its policy trace: exactly the
    // lineage of the last version published.
    let last = of_type(&frames, "published").last().unwrap()["lineage"].clone();
    let (_, listing) = call(&app, Method::GET, "/v1/models", None).await;
    let card = listing["data"].as_array().unwrap().iter().find(|m| m["id"] == ADAPTER).unwrap().clone();
    assert_eq!(card["adapter"]["lineage"], last, "{card}");
    let ask = json!({"model": ADAPTER, "messages": [{"role": "user", "content": "cab"}], "max_tokens": 2, "temperature": 1.0, "policy_trace": true});
    let (status, reply) = call(&app, Method::POST, "/v1/chat/completions", Some(("application/json", ask.to_string().into_bytes()))).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!((reply["brain"]["adapter"]["lineage"].clone(), reply["brain"]["policy_version"].clone()), (last.clone(), json!(3)), "{reply}");
    assert_eq!(reply["policy_trace"]["policy"]["adapter"]["lineage"], last, "{reply}");
}

/// The queue's two promises at the job's own door: an offer is never refused while the queue holds less than one
/// step (the learner is waiting for it), and a full queue refuses the whole offer naming how much it still takes.
#[test]
fn a_full_queue_names_its_allowance_and_never_refuses_below_a_step() {
    let rig = rig_stream("stream-queue");
    let sizes = sizes(&rig);
    let tokens = sizes[0] + sizes[1];
    let base = rig.dir.join("models").join("local").join("toy");
    let config = brain::PolicyGradientConfig { loss: brain::PgLoss { loss_agg: brain::LossAgg::TokenSum, ..brain::PolicyGradientConfig::default().loss }, ..Default::default() };
    // No learner takes from this stream: what is queued stays queued.
    let stream = brain::ChatFineTune::from_pretrained(base.to_str().unwrap()).policy_gradient(config).max_block(128).open_rollout_stream(2 * tokens, 2 * tokens, None, None, None, || true, Default::default()).expect("a stream");
    let answer = |key: &str| trainjob::stream::answer(&stream, 3, "records", &json!({}), &group(sizes, key, 0));
    assert_eq!(answer("a").unwrap()["queued_tokens"], json!(tokens));
    // Below one step the offer is taken even though it passes the queue's bound.
    let mut two = group(sizes, "b", 0);
    two.push(b'\n');
    two.extend(group(sizes, "c", 0));
    assert_eq!(trainjob::stream::answer(&stream, 3, "records", &json!({}), &two).unwrap()["queued_tokens"], json!(3 * tokens));
    let full = answer("d").unwrap_err();
    assert_eq!(full.kind, capability::inbox::InputRefusalKind::Busy, "{full:?}");
    assert_eq!((full.detail["allowance"].clone(), full.detail["queued_tokens"].clone(), full.detail["offered_tokens"].clone()), (json!(0), json!(3 * tokens), json!(tokens)), "{full:?}");
    assert!(full.message.contains("allowance") || full.message.contains("at most"), "{}", full.message);
    assert_eq!(answer("a").unwrap()["duplicates"], json!(["a"]), "a duplicate is answered even when the queue is full");
    let wrong = trainjob::stream::answer(&stream, 3, "records", &json!({"epoch": 2}), &group(sizes, "e", 0)).unwrap_err();
    assert_eq!((wrong.kind, wrong.detail["epoch"].clone()), (capability::inbox::InputRefusalKind::Conflict, json!(3)));
}

/// The gradient of a step is a function of its records and their order alone: the same two groups posted together or
/// one at a time train the same first version, tensor for tensor (the files differ by the run their cards name).
#[tokio::test(flavor = "multi_thread")]
async fn a_step_trains_the_same_whether_its_records_arrived_in_one_post_or_many() {
    let rig = rig_stream("stream-same");
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let sizes = sizes(&rig);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let doc = json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": 2 * tokens, "queue_tokens": 4 * tokens, "max_steps": 1});
    let mut published = Vec::new();
    for split in [false, true] {
        let id = create(&app, stream_job(doc.clone(), json!({}))).await;
        if split {
            assert_eq!(post(&app, &id, "", group(sizes, "g1", 0)).await.0, StatusCode::OK);
            assert_eq!(post(&app, &id, "", group(sizes, "g2", 0)).await.0, StatusCode::OK);
        } else {
            let mut both = group(sizes, "g1", 0);
            both.push(b'\n');
            both.extend(group(sizes, "g2", 0));
            assert_eq!(post(&app, &id, "", both).await.0, StatusCode::OK);
        }
        let (frames, _) = frames_until(&app, &id, 0, |_| false).await;
        // The one step's version is the last published (the first job also published its starting policy; the
        // second, whose id the bank serves already, does not), and the run's adapter in the store is that step's.
        assert!(!of_type(&frames, "published").is_empty(), "{frames:?}");
        let (status, result) = call(&app, Method::GET, &format!("/v1/jobs/{id}/result"), None).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        let digest = trainjob::store::Digest::parse(result["outputs"]["adapter_digest"].as_str().unwrap()).unwrap();
        let path = rig.store.root().join(format!("stream-same-{split}.safetensors"));
        rig.store.fetch(&digest, &path, 1 << 30).unwrap();
        let mut tensors: Vec<(String, Vec<u32>)> = checkpoint::st::load_safetensors(path.to_str().unwrap()).unwrap().tensors.into_iter().map(|(name, data)| (name, data.iter().map(|v| v.to_bits()).collect())).collect();
        tensors.sort();
        std::fs::remove_file(&path).ok();
        published.push(tensors);
    }
    assert!(!published[0].is_empty());
    assert!(published[0] == published[1], "the two runs trained different weights");
}

/// A run resumed from its saved state (the job that ran it is gone, as after a crash) takes again exactly the groups
/// that were applied and not committed; the committed ones are duplicates. The old job's id is unknown to the client's
/// next call, and a call naming the old epoch is refused with the new one.
#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_stream_takes_again_what_was_applied_and_not_committed() {
    let rig = rig_stream("stream-resume");
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let sizes = sizes(&rig);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let body = stream_job(json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": tokens, "queue_tokens": 4 * tokens}), json!({"resume_key": "stream-k1", "checkpoint_every": 2}));
    let id = create(&app, body.clone()).await;
    for key in ["g1", "g2", "g3"] {
        assert_eq!(post(&app, &id, "", group(sizes, key, 0)).await.0, StatusCode::OK);
    }
    // Step 2 wrote the run's state (checkpoint_every 2): g1 and g2 are committed, reported with step 3, whose g3 is
    // applied and not committed.
    let (frames, _) = frames_until(&app, &id, 0, |f| f["type"] == "step" && f["step"] == 3).await;
    assert!(of_type(&frames, "committed").iter().any(|f| keys(f) == ["g1", "g2"] && f["step"] == 2), "{frames:?}");
    assert_eq!(of_type(&frames, "applied").iter().flat_map(|f| keys(f)).collect::<Vec<_>>(), ["g1", "g2", "g3"]);
    let state = rig.store.root().join(trainjob::runs::RUNS_DIR).join(trainjob::store::digest_of(b"stream-k1").hex()).join("state.safetensors");
    let saved = std::fs::read(&state).expect("the state of step 2");
    // The job ends (a cancel writes the state of step 3, holding g3); the state on disk is then put back to step 2's,
    // as a crash after step 3 would leave it.
    assert_eq!(call(&app, Method::DELETE, &format!("/v1/jobs/{id}"), None).await.0, StatusCode::OK);
    crate::support::assert_reservation_returned(&rig);
    assert_ne!(std::fs::read(&state).unwrap(), saved, "the stop wrote its own state");
    std::fs::write(&state, &saved).unwrap();

    // The client's next call to the job it knew is refused; it starts the run again under its key.
    assert_eq!(post(&app, &id, "", group(sizes, "g4", 0)).await.0, StatusCode::GONE);
    let resumed = create(&app, body.clone()).await;
    let (status, wrong) = post(&app, &resumed, "?epoch=1", group(sizes, "g3", 0)).await;
    assert_eq!((status, wrong["detail"]["epoch"].clone()), (StatusCode::CONFLICT, json!(2)), "{wrong}");
    let mut all = Vec::new();
    for key in ["g1", "g2", "g3"] {
        let (status, answer) = post(&app, &resumed, "?epoch=2", group(sizes, key, 0)).await;
        assert_eq!(status, StatusCode::OK, "{answer}");
        all.push((key, answer["accepted"].clone(), answer["duplicates"].clone()));
    }
    assert_eq!(all, [("g1", json!([]), json!(["g1"])), ("g2", json!([]), json!(["g2"])), ("g3", json!(["g3"]), json!([]))], "only the uncommitted group is taken again");
    assert_eq!(call(&app, Method::POST, &format!("/v1/jobs/{resumed}/finish"), None).await.0, StatusCode::OK);
    let (frames, _) = frames_until(&app, &resumed, 0, |_| false).await;
    assert_eq!(of_type(&frames, "stream")[0]["epoch"], 2);
    assert_eq!(of_type(&frames, "applied").iter().flat_map(|f| keys(f)).collect::<Vec<_>>(), ["g3"]);
    // The resumed job's versions are the run's (its key), under the new epoch: update 3 of epoch 2 is not update 3 of
    // epoch 1, which the first job published before the state was put back.
    let lineages: Vec<(Value, Value, Value)> = of_type(&frames, "published").iter().map(|f| (f["lineage"]["run"].clone(), f["lineage"]["epoch"].clone(), f["lineage"]["update"].clone())).collect();
    assert_eq!(lineages, [(json!("stream-k1"), json!(2), json!(3))], "{frames:?}");
    let (status, result) = call(&app, Method::GET, &format!("/v1/jobs/{resumed}/result"), None).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!((result["outputs"]["resume"]["state"].clone(), result["outputs"]["resume"]["resumed_at"].clone(), result["outputs"]["steps_completed"].clone()), (json!("continued"), json!(2), json!(3)), "{result}");
}

/// A stream needs a server that serves an adapter bank and a surface that delivers its records; neither is assumed.
#[test]
fn a_stream_job_is_refused_where_it_could_not_publish_or_be_fed() {
    let rig = crate::support::rig_qwen3("stream-refuse", 16 << 30);
    let stream = json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": 8, "queue_tokens": 8}).to_string();
    let params = json!({"base": BASE, "objective": "policy_gradient", "loss_agg": "token-sum", "max_block": 128, "stream": stream});
    let refused = crate::support::run(&rig, params.clone(), &[], &Frames::default()).unwrap_err();
    assert!(refused.contains("adapter bank"), "{refused}");
    let rig = rig_stream("stream-refuse-unfed");
    let refused = crate::support::run(&rig, params.clone(), &[], &Frames::default()).unwrap_err();
    assert!(refused.contains("surface"), "{refused}");
    for (field, value, why) in [("queue_tokens", json!(4), "queue_tokens"), ("publish_adapter", json!("local/toy"), "publish_adapter"), ("extra", json!(1), "unknown field"), ("support", json!(0), "support must be 1 to 1024"), ("support", json!(1025), "support must be 1 to 1024")] {
        let mut doc: Value = serde_json::from_str(params["stream"].as_str().unwrap()).unwrap();
        doc[field] = value;
        let mut bad = params.clone();
        bad["stream"] = json!(doc.to_string());
        let refused = crate::support::run(&rig, bad, &[], &Frames::default()).unwrap_err();
        assert!(refused.contains(why), "{field}: {refused}");
    }
    let mut both = params.clone();
    both["data"] = json!(json!({"train": {"digest": "0".repeat(64)}}).to_string());
    assert!(crate::support::run(&rig, both, &[], &Frames::default()).unwrap_err().contains("exactly one of data and stream"));
    let mut seq = params.clone();
    seq["loss_agg"] = json!("seq-mean");
    let refused = crate::support::run(&rig, seq, &[], &Frames::default()).unwrap_err();
    assert!(refused.contains("surface") && !refused.contains("loss_agg"), "a stream trains a per-sequence mean; the job is refused for its surface alone: {refused}");
}

/// The stream trains whatever estimator the job's policy-gradient settings select: the score-centred one centres over
/// the sampler's top-K the records carry, of the K the stream names, and a record of another K is refused.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_trains_the_score_centred_estimator_over_the_records_top_k() {
    let rig = rig_stream("stream-centred");
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let sizes = sizes(&rig);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let with_top = |key: &str, k: usize| -> Vec<u8> {
        let lines: Vec<String> = String::from_utf8(group(sizes, key, 0)).unwrap().lines().map(|line| {
            let mut record: Value = serde_json::from_str(line).unwrap();
            let n = record["messages"][1]["behavior_logprobs"].as_array().unwrap().len();
            record["messages"][1]["behavior_top_ids"] = json!(vec![(0..k as u32).collect::<Vec<_>>(); n]);
            record["messages"][1]["behavior_top_logprobs"] = json!(vec![vec![-3.0; k]; n]);
            record.to_string()
        }).collect();
        lines.join("\n").into_bytes()
    };
    let doc = json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": tokens, "queue_tokens": 2 * tokens, "top_k": 2, "max_steps": 1});
    let (status, refused) = call(&app, Method::POST, "/v1/jobs", Some(("application/json", stream_job(json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": tokens, "queue_tokens": tokens}), json!({"estimator": "score-centred"})).to_string().into_bytes()))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "score-centred names its K: {refused}");
    assert!(refused.to_string().contains("top_k"), "{refused}");
    let id = create(&app, stream_job(doc, json!({"estimator": "score-centred"}))).await;
    let (status, wrong) = post(&app, &id, "", with_top("k3", 3)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{wrong}");
    assert!(wrong.to_string().contains("top-3"), "{wrong}");
    assert_eq!(post(&app, &id, "", with_top("g1", 2)).await.0, StatusCode::OK);
    let (frames, _) = frames_until(&app, &id, 0, |_| false).await;
    let steps = of_type(&frames, "step");
    assert_eq!(steps.len(), 1, "{frames:?}");
    assert!(steps[0]["pg"]["sampler_mass"].as_f64().is_some_and(|m| m > 0.0), "{}", steps[0]);
    assert_eq!(of_type(&frames, "published").len(), 2, "the starting policy, then the one step's");
}

/// The policy version that answers a chat request for `model`.
async fn version_of(app: &Router, model: &str) -> Result<u64, Value> {
    let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2});
    let (status, reply) = call(app, Method::POST, "/v1/chat/completions", Some(("application/json", body.to_string().into_bytes()))).await;
    if status != StatusCode::OK {
        return Err(reply);
    }
    Ok(reply["brain"]["policy_version"].as_u64().unwrap_or_else(|| panic!("the reply names its policy version: {reply}")))
}

/// A stream that retains its versions also serves each newest one under an id of its own, so a client that started
/// an episode on a version stays on it after newer ones are published; the oldest beyond the retention are withdrawn.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_retaining_its_versions_serves_each_under_an_id_of_its_own_while_retained() {
    let rig = rig_stream("stream-retain");
    let app = router(AppState::new(rig.exec.clone(), KEY, Provider::OpenAI).with_training(true));
    let sizes = sizes(&rig);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let stream = json!({"publish_adapter": ADAPTER, "publish_every": 1, "max_staleness": 4, "step_tokens": tokens, "queue_tokens": tokens, "retain_versions": 2});
    let id = create(&app, stream_job(stream, json!({}))).await;
    let v = |n: u64| trainjob::bank::version_id(ADAPTER, n);
    assert_eq!(served_version(&app, 1).await, 1);
    assert_eq!(version_of(&app, &v(1)).await, Ok(1), "the starting version under its own id");
    assert_eq!(post(&app, &id, "", group(sizes, "g1", 1)).await.0, StatusCode::OK);
    assert_eq!(served_version(&app, 2).await, 2);
    assert_eq!(version_of(&app, ADAPTER).await, Ok(2), "the published id moves on");
    assert_eq!(version_of(&app, &v(1)).await, Ok(1), "an episode that started on version 1 stays on it");
    assert_eq!(post(&app, &id, "", group(sizes, "g2", 2)).await.0, StatusCode::OK);
    assert_eq!(served_version(&app, 3).await, 3);
    assert_eq!(version_of(&app, &v(2)).await, Ok(2));
    assert!(version_of(&app, &v(1)).await.is_err(), "beyond the retention of two, version 1 is withdrawn");

    let (status, _) = call(&app, Method::POST, &format!("/v1/jobs/{id}/finish"), None).await;
    assert_eq!(status, StatusCode::OK);
    let (frames, _) = frames_until(&app, &id, 0, |_| false).await;
    let ids: Vec<(u64, String)> = of_type(&frames, "published").iter().map(|f| (f["version"].as_u64().unwrap(), f["version_id"].as_str().unwrap().to_string())).collect();
    assert_eq!(ids, [(1, v(1)), (2, v(2)), (3, v(3))], "every published frame names its version id");
    for frame in of_type(&frames, "published") {
        let artifact = frame["artifact_digest"].as_str().unwrap_or_else(|| panic!("a retained version is kept in the store: {frame}"));
        assert!(rig.store.contains(&trainjob::store::Digest::parse(artifact).unwrap()), "the store holds {artifact}");
    }
}

#[test]
fn a_retention_beyond_the_bound_is_refused() {
    let doc = |n: u32| json!({"publish_adapter": ADAPTER, "publish_every": 1, "step_tokens": 8, "queue_tokens": 8, "retain_versions": n}).to_string();
    assert_eq!(trainjob::stream::StreamRefs::parse(&doc(3)).unwrap().retain_versions, 3);
    assert!(trainjob::stream::StreamRefs::parse(&doc(trainjob::stream::MAX_RETAINED_VERSIONS + 1)).unwrap_err().contains("retain_versions"));
}
