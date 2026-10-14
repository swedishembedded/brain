// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The agreement gate: a decision model reached through the SDK and the same
//! model reached over the wire answer the same question the same way.
//!
//! "The wire" is the whole served path: HTTP `/v1/run` and `/v1/systemone`,
//! the scheduler, the residency adapter, the capability and its request
//! parser. The SDK side builds its typed questions BY HAND rather than through
//! that parser, so a disagreement in how a request is understood shows up here
//! instead of cancelling out.
//!
//! The gate runs two ways. On tiny random-weight checkpoints of BOTH
//! architectures it always runs, so it cannot pass by being skipped - the
//! answers are noise, which is fine: the claim is that two paths to one model
//! agree, not that the model is right. On the real `convaiinnovations/laya`
//! checkpoint it runs when the weights are pulled and skips by name when not.
//!
//! The tolerance is 1e-4 on every published number: the two sides build the
//! same graph on the same device, and the SDK builds it trainable while the
//! server builds it for inference only, so the only room for a difference is
//! float reassociation in buffers the inference build does not allocate.
//!
//! Swedish Embedded AB implements verification that two routes to one model
//! cannot drift apart. If your team needs expertise in cross-checking a
//! serving stack against its reference path, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use apiserve::{router, AppState, Provider};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use brain::decision::{Answer, Opt, Question, State};
use catalog::resident_decision::DecisionResident;
use brain::decision_run::{train, TrainRun};
use decisions::fixtures::{synthetic_jsonl, write_decide, write_laya, Scratch, SyntheticTask};
use residency::budget::Budgets;
use residency::{Device, Executor, Policy, ResidentModel};
use serde_json::{json, Value};
use tower::ServiceExt;

const TOLERANCE: f64 = 1e-4;
const STATE: &str = "my payment failed again today";

/// The same three questions, once typed and once as the wire text.
fn typed() -> Vec<Question> {
    vec![
        Question::Choice { instructions: "who handles this".into(), options: vec![Opt::described("billing", "payments"), Opt::new("tech"), Opt::new("sales")] },
        Question::Score { instructions: "how upset".into(), levels: vec!["calm".into(), "annoyed".into(), "furious".into()] },
        Question::Noul { instructions: "is this urgent".into(), yes: Some("someone is blocked".into()), no: Some("it can wait".into()) },
    ]
}

const WIRE: &str = r#"[
  {"id": "team", "type": "choice", "instruction": "who handles this",
   "options": [{"id": "billing", "description": "payments"}, "tech", "sales"]},
  {"id": "anger", "type": "score", "instruction": "how upset", "options": ["calm", "annoyed", "furious"]},
  {"id": "urgent", "type": "noul", "instruction": "is this urgent", "criteria": {"true": "someone is blocked", "false": "it can wait"}}
]"#;

const SYSTEMONE: &str = r#"{"model": "m/x", "state": "my payment failed again today", "questions": {
  "team": {"type": "choice", "instructions": "who handles this",
           "criteria": {"billing": "payments", "tech": "", "sales": null}},
  "anger": {"type": "score", "instructions": "how upset", "criteria": ["calm", "annoyed", "furious"]},
  "urgent": {"type": "noul", "instructions": "is this urgent", "criteria": {"true": "someone is blocked", "false": "it can wait"}}}}"#;

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= TOLERANCE, "{what}: sdk {a} vs served {b}");
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// The SDK's answers for `typed()` about `STATE` from the checkpoint at `dir`.
fn sdk(dir: &Path) -> Vec<Answer> {
    let mut pipe = brain::DecisionPipeline::builder(dir.to_str().unwrap()).load().unwrap_or_else(|e| panic!("sdk load: {e}"));
    pipe.decide(&State::Str(STATE.into()), &typed()).unwrap_or_else(|e| panic!("sdk decide: {e}"))
}

struct Served {
    app: Router,
    exec: Executor,
}

impl Drop for Served {
    /// Release the devices in an orderly way - see the same impl in
    /// `decision_served.rs`.
    fn drop(&mut self) {
        self.exec.shutdown();
    }
}

fn serve(id: &str, dir: &Path) -> Served {
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(DecisionResident::new(id, dir.to_path_buf()))];
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());
    Served { app: router(AppState::new(exec.clone(), "k", Provider::OpenAI)), exec }
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 22).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// One choice answer: the same option selected, the same full distribution.
fn agree_on_choice(sdk: &Answer, served: &Value) {
    match (sdk, served) {
        (Answer::Choice { choice, probabilities, confidence }, served) => {
            assert_eq!(served["selected"], json!(choice), "the same option is selected");
            close(*confidence as f64, f(&served["confidence"]), "choice confidence");
            for (name, p) in probabilities {
                let r = served["ranked"].as_array().unwrap().iter().find(|r| r["id"] == json!(name)).unwrap_or_else(|| panic!("{name} missing from the ranking"));
                close(*p as f64, f(&r["probability"]), &format!("P({name})"));
            }
            assert_eq!(served["ranked"].as_array().unwrap().len(), probabilities.len(), "the FULL distribution");
        }
        other => panic!("{other:?}"),
    }
}

/// Compare the SDK's answers with the capability's response.
fn agree_with_capability(sdk: &[Answer], out: &Value) {
    let answers = out["outputs"]["answers"].as_array().unwrap_or_else(|| panic!("no answers: {out}"));
    assert_eq!(answers.len(), sdk.len());
    agree_on_choice(&sdk[0], &answers[0]);
    match (&sdk[1], &answers[1]) {
        (Answer::Score { score, probabilities, confidence, .. }, served) => {
            close(*score as f64, f(&served["score"]), "score");
            close(*confidence as f64, f(&served["confidence"]), "score confidence");
            for (i, p) in probabilities.iter().enumerate() {
                let r = served["ranked"].as_array().unwrap().iter().find(|r| r["index"] == json!(i)).unwrap();
                close(*p as f64, f(&r["probability"]), &format!("P(level {i})"));
            }
        }
        other => panic!("{other:?}"),
    }
    match (&sdk[2], &answers[2]) {
        (Answer::Noul { noul }, served) => close(*noul as f64, f(&served["noul"]), "noul"),
        other => panic!("{other:?}"),
    }
}

/// Compare the SDK's answers with `/v1/systemone`'s.
fn agree_with_systemone(sdk: &[Answer], doc: &Value) {
    let a = &doc["answers"];
    if let Answer::Choice { choice, probabilities, confidence } = &sdk[0] {
        assert_eq!(a["team"]["choice"], json!(choice));
        close(*confidence as f64, f(&a["team"]["confidence"]), "systemone choice confidence");
        for (name, p) in probabilities {
            close(*p as f64, f(&a["team"]["probabilities"][name]), &format!("systemone P({name})"));
        }
    }
    if let Answer::Score { score, probabilities, .. } = &sdk[1] {
        close(*score as f64, f(&a["anger"]["score"]), "systemone score");
        for (i, p) in probabilities.iter().enumerate() {
            close(*p as f64, f(&a["anger"]["probabilities"][i.to_string()]), &format!("systemone P(level {i})"));
        }
    }
    if let Answer::Noul { noul } = &sdk[2] {
        close(*noul as f64, f(&a["urgent"]["noul"]), "systemone noul");
    }
}

async fn gate(id: &str, dir: &Path) {
    let reference = sdk(dir);
    let served = serve(id, dir);

    let (status, out) = post(&served.app, "/v1/run", json!({"model": id, "action": "decide", "params": {"state": STATE, "questions": WIRE}})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    agree_with_capability(&reference, &out);

    let (status, doc) = post(&served.app, "/v1/systemone", serde_json::from_str::<Value>(&SYSTEMONE.replace("m/x", id)).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{doc}");
    agree_with_systemone(&reference, &doc);
}

#[tokio::test]
async fn laya_over_the_wire_agrees_with_laya_through_the_sdk() {
    let d = Scratch::new("agree-laya");
    write_laya(&d, 48);
    gate("acme/laya", &d).await;
}

#[tokio::test]
async fn decide_over_the_wire_agrees_with_decide_through_the_sdk() {
    let d = Scratch::new("agree-decide");
    write_decide(&d, 11);
    gate("acme/triage", &d).await;
}

/// The head `brain decide train` writes for a Laya base is served exactly as the
/// SDK reads it: the directory holds no trunk, so this fails if either path
/// resolves the referenced backbone differently or serves anything but the
/// trained head.
#[tokio::test]
async fn a_trained_laya_head_over_the_wire_agrees_with_the_sdk() {
    let root = Scratch::new("agree-laya-trained");
    write_laya(&root.join("base"), 48);
    std::fs::write(root.join("data.jsonl"), synthetic_jsonl(&SyntheticTask::new(12, 4, 4))).unwrap();
    let mut run = TrainRun::new(root.join("data.jsonl"), root.join("base").to_str().unwrap(), root.join("tuned"));
    run.steps = 6;
    run.batch = 2;
    train(&run, &mut |_, _| {}).unwrap();
    assert!(!root.join("tuned/model.safetensors").exists(), "the trunk is referenced, not copied");

    let (stock, tuned) = (sdk(&root.join("base")), sdk(&root.join("tuned")));
    let probabilities = |a: &Answer| match a {
        Answer::Choice { probabilities, .. } => probabilities.iter().map(|(_, p)| *p).collect::<Vec<_>>(),
        other => panic!("{other:?}"),
    };
    assert_ne!(probabilities(&stock[0]), probabilities(&tuned[0]), "the trained head, not the base's, is what answers");

    // A training run declares the one question type it trained, so that is
    // the one the served model answers; the others are flagged, not guessed.
    let served = serve("acme/tuned", &root.join("tuned"));
    let team = r#"[{"id": "team", "type": "choice", "instruction": "who handles this", "options": [{"id": "billing", "description": "payments"}, "tech", "sales"]}]"#;
    let (status, out) = post(&served.app, "/v1/run", json!({"model": "acme/tuned", "action": "decide", "params": {"state": STATE, "questions": team}})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    agree_on_choice(&tuned[0], &out["outputs"]["answers"][0]);
}

/// The same claim on the real released checkpoint. Skips by name when it is
/// not pulled; the two tests above are what keep the gate from being vacuous.
#[tokio::test]
async fn the_real_laya_checkpoint_over_the_wire_agrees_with_the_sdk() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }
    gate("convaiinnovations/laya", Path::new(&dir)).await;
}
