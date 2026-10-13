// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `decide` capability: typed questions about one state, answered with a
//! full calibrated distribution, behind the generalized [`capability`]
//! interface - which is what makes `brain decide decide`, the D-Bus `Run` method and the
//! HTTP `/v1/run` route work with no decision-specific plumbing in any
//! transport.
//!
//! # Request
//!
//! | param | |
//! |---|---|
//! | `state` | the situation: text, or a JSON document (see `state_format`) |
//! | `state_format` | `text` (default): taken verbatim; `json`: the state is a JSON document, where a JSON string is text and anything else is structured state kept in the caller's key order |
//! | `questions` | JSON, either spelling [`crate::wire`] documents |
//! | `deadline_ms` | cooperative deadline, from when the model starts serving the call |
//!
//! # Response (`outputs`)
//!
//! ```json
//! {"model": "convaiinnovations/laya", "backend": "laya", "latency_ms": 41.2,
//!  "max_input_tokens": 512,
//!  "flags": {"state_truncated": false, "unsupported_question_types": []},
//!  "answers": [
//!   {"id": "team", "type": "choice", "selected": "billing", "selected_index": 0,
//!    "confidence": 0.91,
//!    "ranked": [{"id": "billing", "index": 0, "probability": 0.98},
//!               {"id": "tech", "index": 1, "probability": 0.02}],
//!    "state": {"tokens": 31, "kept": 31, "truncated": false}},
//!   {"id": "anger", "type": "score", "score": 1.3, "confidence": 0.4,
//!    "legend": ["calm", "annoyed", "furious"],
//!    "ranked": [{"id": "1", "index": 1, "label": "annoyed", "probability": 0.7}],
//!    "state": {"tokens": 31, "kept": 31, "truncated": false}},
//!   {"id": "urgent", "type": "noul", "noul": 0.78,
//!    "ranked": [{"id": "true", "index": 1, "probability": 0.78},
//!               {"id": "false", "index": 0, "probability": 0.22}],
//!    "state": {"tokens": 31, "kept": 31, "truncated": false}}]}
//! ```
//!
//! `ranked` is the FULL distribution, best first (ties by position), every
//! candidate carrying its position in the list the caller sent. A `noul` has
//! no `confidence`: a two-outcome probability already is its own confidence.
//! `state.truncated` is true when the model answered from a prefix of the
//! state (Laya's trained context is a hard bound; `decide` windows the state
//! instead and never drops it). A question whose type the model does not
//! support is answered `{"id", "type", "unsupported": true, "reason"}` and
//! named in `flags.unsupported_question_types`; a request where NO question
//! is supported fails.
//!
//! # Failures
//!
//! Each is tagged with a [`capability::FailureClass`]: a malformed request or
//! one outside the published limits is `invalid-request`; a request no
//! question of which the model supports is `unsupported-capability`; a passed
//! deadline is `timeout`.
//!
//! # Batching
//!
//! Within a request the `decide` architecture encodes the state ONCE for every
//! question. Across requests the instance answers serially (the default
//! `run_batch`): each architecture packs one request's state and questions
//! into one bounded sequence, and a Laya head takes one question's option
//! markers per call, so there is no shared pass to exploit across requests.
//!
//! Swedish Embedded AB implements decision services whose answers carry their
//! own distribution, provenance and limits, so a caller never has to guess how
//! far to trust them. If your team needs expertise in serving calibrated
//! models, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::time::{Duration, Instant};

use capability::{tag, ActionResult, ActionSpec, FailureClass, Invocation, Manifest, Outcome, ParamSpec, ParamType};
use decide::primitives::{Answer, Question};
use serde_json::{json, Value};

use crate::engine::{AnswerError, Answered, Backend, Engine, Reply};
use crate::wire::{self, NamedQuestion, QuestionType, StateFormat};

/// The capability family's model id, for the in-process CLI path (`brain decide`).
pub const FAMILY: &str = "brain/decision";
/// The action every decision model serves.
pub const ACTION: &str = "decide";
/// The deadline applied when a caller names none.
pub const DEFAULT_DEADLINE_MS: i64 = 60_000;
/// The longest deadline a caller may ask for.
pub const MAX_DEADLINE_MS: i64 = 600_000;

/// The `decide` action's schema.
pub fn decide_spec() -> ActionSpec {
    ActionSpec::new(ACTION, "Answer typed questions (choice, score, noul) about one state with a calibrated distribution over options the caller supplies.")
        .param(ParamSpec::new("state", ParamType::Str, "the situation: text, or a JSON document (structured state keeps its key order)").required())
        .param(
            ParamSpec::new("state_format", ParamType::Enum(vec!["text".into(), "json".into()]), "text: taken verbatim; json: a JSON document, where a JSON string is text and anything else is structured state in the caller's key order")
                .default(json!("text")),
        )
        .param(
            ParamSpec::new(
                "questions",
                ParamType::Str,
                "JSON: an array of {id, type, instruction, options|criteria}, or an object of name -> {type, instructions, criteria}",
            )
            .required(),
        )
        .param(
            ParamSpec::new("deadline_ms", ParamType::Int, "cooperative deadline in milliseconds, from when the model starts serving the call")
                .default(json!(DEFAULT_DEADLINE_MS))
                .min(1.0)
                .max(MAX_DEADLINE_MS as f64),
        )
}

/// The capability manifest for the decision model `model`.
pub fn manifest(model: &str) -> Manifest {
    Manifest::new(model, "Decision model: calibrated answers to typed questions over options supplied at request time.", vec![decide_spec()])
}

/// What a loaded model tells callers about itself.
#[derive(Clone, Debug)]
pub struct Served {
    /// The id the model is addressed by.
    pub model: String,
    /// The question types it supports.
    pub question_types: Vec<QuestionType>,
}

impl Served {
    /// A model supporting every question type.
    pub fn new(model: impl Into<String>) -> Served {
        Served { model: model.into(), question_types: QuestionType::ALL.to_vec() }
    }
}

/// One loaded model answering the `decide` action.
pub struct Session {
    served: Served,
    engine: Engine,
}

fn fail(class: FailureClass, message: impl AsRef<str>) -> String {
    tag(class, message)
}

/// The parsed, validated parameters of one call.
struct Call {
    request: wire::Request,
    deadline: Duration,
}

fn parse_call(inv: &Invocation) -> Result<Call, String> {
    let invalid = |m: String| fail(FailureClass::InvalidRequest, m);
    let state = inv.get_str("state").ok_or_else(|| invalid("`state` is required".into()))?;
    let questions = inv.get_str("questions").ok_or_else(|| invalid("`questions` is required".into()))?;
    let format = match inv.get_str("state_format") {
        None => StateFormat::Text,
        Some(f) => StateFormat::parse(&f).ok_or_else(|| invalid(format!("`state_format` must be \"json\" or \"text\", got {f:?}")))?,
    };
    let deadline_ms = inv.get_i64("deadline_ms").unwrap_or(DEFAULT_DEADLINE_MS);
    if !(1..=MAX_DEADLINE_MS).contains(&deadline_ms) {
        return Err(invalid(format!("`deadline_ms` must be between 1 and {MAX_DEADLINE_MS}, got {deadline_ms}")));
    }
    let request = wire::parse_request(&state, format, &questions).map_err(invalid)?;
    Ok(Call { request, deadline: Duration::from_millis(deadline_ms as u64) })
}

impl Session {
    pub fn new(served: Served, engine: Engine) -> Session {
        Session { served, engine }
    }

    pub fn served(&self) -> &Served {
        &self.served
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Run one `decide` call.
    pub fn run(&mut self, inv: &Invocation) -> ActionResult {
        self.run_started(inv, Instant::now())
    }

    /// [`Session::run`] with the deadline measured from `started` - a seam so
    /// a test can say the clock has already run out.
    fn run_started(&mut self, inv: &Invocation, started: Instant) -> ActionResult {
        let Call { request, deadline } = parse_call(inv)?;
        let deadline_at = started + deadline;
        let timeout = || fail(FailureClass::Timeout, format!("the {} ms deadline passed before the model answered", deadline.as_millis()));

        let (supported, unsupported): (Vec<&NamedQuestion>, Vec<&NamedQuestion>) =
            request.questions.iter().partition(|q| self.served.question_types.contains(&QuestionType::of(&q.question)));
        if supported.is_empty() {
            let kinds: Vec<&str> = unsupported.iter().map(|q| QuestionType::of(&q.question).slug()).collect();
            return Err(fail(
                FailureClass::UnsupportedCapability,
                format!("model '{}' supports no question of the types asked ({})", self.served.model, kinds.join(", ")),
            ));
        }
        if inv.cancel.is_cancelled() {
            return Err("cancelled".into());
        }

        let asked: Vec<Question> = supported.iter().map(|q| q.question.clone()).collect();
        let began = Instant::now();
        let reply = self.engine.answer_until(&request.state, &asked, Some(deadline_at)).map_err(|e| match e {
            AnswerError::Refused(m) => fail(FailureClass::InvalidRequest, m),
            AnswerError::DeadlineExceeded => timeout(),
        })?;
        let latency = began.elapsed();
        if Instant::now() > deadline_at {
            return Err(timeout());
        }

        let mut answers = Vec::with_capacity(request.questions.len());
        let mut replies = supported.iter().zip(&reply.answered);
        for q in &request.questions {
            if self.served.question_types.contains(&QuestionType::of(&q.question)) {
                let (named, answered) = replies.next().ok_or_else(|| fail(FailureClass::BackendFailure, "the model returned fewer answers than questions"))?;
                answers.push(render_answer(&named.id, &named.question, answered));
            } else {
                answers.push(render_unsupported(&q.id, QuestionType::of(&q.question), &self.served.model));
            }
        }
        let unsupported_types: Vec<&str> = {
            let mut t: Vec<QuestionType> = unsupported.iter().map(|q| QuestionType::of(&q.question)).collect();
            t.sort();
            t.dedup();
            t.into_iter().map(QuestionType::slug).collect()
        };
        Ok(render_outcome(&self.served.model, self.engine.backend(), self.engine.max_input_tokens(), latency, &reply, answers, &unsupported_types))
    }
}

// ===================== rendering =====================

/// A probability to its shortest `f32` decimal, so `0.3f32` is `0.3` and not
/// `0.30000001192092896`. Non-finite values have no JSON spelling.
fn num(v: f32) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    v.to_string().parse::<f64>().ok().and_then(serde_json::Number::from_f64).map(Value::Number).unwrap_or(Value::Null)
}

/// `(position, probability)` best first, ties by position.
fn ranking(p: &[f32]) -> Vec<(usize, f32)> {
    let mut r: Vec<(usize, f32)> = p.iter().copied().enumerate().collect();
    r.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    r
}

fn state_json(a: &Answered) -> Value {
    json!({"tokens": a.state.total, "kept": a.state.kept, "truncated": a.state.truncated()})
}

fn render_answer(id: &str, q: &Question, a: &Answered) -> Value {
    match (&a.answer, q) {
        (Answer::Choice { choice, probabilities, confidence }, _) => {
            let p: Vec<f32> = probabilities.iter().map(|(_, p)| *p).collect();
            let ranked: Vec<Value> = ranking(&p).into_iter().map(|(i, pi)| json!({"id": probabilities[i].0, "index": i, "probability": num(pi)})).collect();
            let selected_index = ranking(&p).first().map(|(i, _)| *i).unwrap_or(0);
            json!({"id": id, "type": "choice", "selected": choice, "selected_index": selected_index,
                   "confidence": num(*confidence), "ranked": ranked, "state": state_json(a)})
        }
        (Answer::Score { score, legend, probabilities, confidence }, _) => {
            let ranked: Vec<Value> = ranking(probabilities)
                .into_iter()
                .map(|(i, pi)| json!({"id": i.to_string(), "index": i, "label": legend[i], "probability": num(pi)}))
                .collect();
            json!({"id": id, "type": "score", "score": num(*score), "confidence": num(*confidence),
                   "legend": legend, "ranked": ranked, "state": state_json(a)})
        }
        (Answer::Noul { noul }, _) => {
            let ranked: Vec<Value> = ranking(&[1.0 - *noul, *noul])
                .into_iter()
                .map(|(i, pi)| json!({"id": if i == 1 { "true" } else { "false" }, "index": i, "probability": num(pi)}))
                .collect();
            json!({"id": id, "type": "noul", "noul": num(*noul), "ranked": ranked, "state": state_json(a)})
        }
    }
}

fn render_unsupported(id: &str, ty: QuestionType, model: &str) -> Value {
    json!({"id": id, "type": ty.slug(), "unsupported": true, "reason": format!("model '{model}' does not support {} questions", ty.slug())})
}

fn render_outcome(model: &str, backend: Backend, max_input_tokens: usize, latency: Duration, reply: &Reply, answers: Vec<Value>, unsupported_types: &[&str]) -> Outcome {
    let truncated = reply.answered.iter().any(|a| a.state.truncated());
    let mut out = Outcome::new()
        .set("model", json!(model))
        .set("backend", json!(backend.slug()))
        .set("latency_ms", json!((latency.as_secs_f64() * 1e6).round() / 1e3))
        .set("max_input_tokens", json!(max_input_tokens))
        .set("flags", json!({"state_truncated": truncated, "unsupported_question_types": unsupported_types}))
        .set("answers", Value::Array(answers));
    if let Some(w) = reply.windows {
        out = out.set("windows", json!(w));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use decide::primitives::Opt;
    use modernbert::StateFit;

    fn answered(answer: Answer, total: usize, kept: usize) -> Answered {
        Answered { answer, state: StateFit { total, kept } }
    }

    #[test]
    fn a_checkpoint_manifest_declares_one_decide_action_and_no_path() {
        let m = manifest("x/y");
        assert_eq!(m.model, "x/y");
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["decide"]);
        let params: Vec<&str> = m.actions[0].params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(params, ["state", "state_format", "questions", "deadline_ms"]);
    }

    #[test]
    fn a_choice_is_ranked_best_first_with_every_candidate_and_its_position() {
        let q = Question::Choice { instructions: "i".into(), options: vec![Opt::new("a"), Opt::new("b"), Opt::new("c")] };
        let a = answered(Answer::Choice { choice: "b".into(), probabilities: vec![("a".into(), 0.2), ("b".into(), 0.5), ("c".into(), 0.3)], confidence: 0.4 }, 30, 30);
        let v = render_answer("pick", &q, &a);
        assert_eq!(v["selected"], "b");
        assert_eq!(v["selected_index"], 1);
        let ranked: Vec<(&str, u64)> = v["ranked"].as_array().unwrap().iter().map(|r| (r["id"].as_str().unwrap(), r["index"].as_u64().unwrap())).collect();
        assert_eq!(ranked, [("b", 1), ("c", 2), ("a", 0)]);
        assert_eq!(v["ranked"][0]["probability"], 0.5);
        assert_eq!(v["confidence"], 0.4);
        assert_eq!(v["state"], json!({"tokens": 30, "kept": 30, "truncated": false}));
    }

    #[test]
    fn a_tie_ranks_by_the_position_the_caller_listed() {
        let ranked = ranking(&[0.25, 0.5, 0.25]);
        assert_eq!(ranked.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 0, 2]);
    }

    #[test]
    fn a_score_carries_its_scale_and_labels_each_level() {
        let q = Question::Score { instructions: "i".into(), levels: vec!["calm".into(), "angry".into()] };
        let a = answered(Answer::Score { score: 0.7, legend: vec!["calm".into(), "angry".into()], probabilities: vec![0.3, 0.7], confidence: 0.12 }, 5, 5);
        let v = render_answer("anger", &q, &a);
        assert_eq!(v["score"], 0.7);
        assert_eq!(v["legend"], json!(["calm", "angry"]));
        assert_eq!(v["ranked"][0], json!({"id": "1", "index": 1, "label": "angry", "probability": 0.7}));
    }

    #[test]
    fn a_noul_has_no_confidence_and_a_two_candidate_distribution() {
        let q = Question::Noul { instructions: "i".into(), yes: None, no: None };
        let v = render_answer("urgent", &q, &answered(Answer::Noul { noul: 0.75 }, 5, 5));
        assert!(v.get("confidence").is_none());
        assert_eq!(v["noul"], 0.75);
        assert_eq!(v["ranked"], json!([{"id": "true", "index": 1, "probability": 0.75}, {"id": "false", "index": 0, "probability": 0.25}]));
    }

    #[test]
    fn a_truncated_state_is_flagged_on_the_answer_and_on_the_response() {
        let q = Question::Noul { instructions: "i".into(), yes: None, no: None };
        let a = answered(Answer::Noul { noul: 0.5 }, 900, 411);
        let v = render_answer("q", &q, &a);
        assert_eq!(v["state"], json!({"tokens": 900, "kept": 411, "truncated": true}));
        let reply = Reply { answered: vec![a], windows: None };
        let out = render_outcome("m", Backend::Laya, 512, Duration::from_micros(1500), &reply, vec![v], &[]);
        assert_eq!(out.outputs["flags"]["state_truncated"], true);
        assert_eq!(out.outputs["backend"], "laya");
        assert_eq!(out.outputs["max_input_tokens"], 512);
        assert_eq!(out.outputs["latency_ms"], 1.5);
        assert!(out.outputs.get("windows").is_none());
    }

    #[test]
    fn an_unsupported_question_is_an_entry_not_a_silence() {
        let v = render_unsupported("s", QuestionType::Score, "m");
        assert_eq!(v["unsupported"], true);
        assert_eq!(v["type"], "score");
        assert!(v["reason"].as_str().unwrap().contains("score"));
    }

    #[test]
    fn probabilities_print_as_the_f32_the_model_produced() {
        assert_eq!(num(0.3), json!(0.3));
        assert_eq!(num(f32::NAN), Value::Null);
    }

    fn class_of(e: &str) -> FailureClass {
        capability::classify(e).unwrap_or_else(|| panic!("untagged error: {e}")).0
    }

    #[test]
    fn a_call_missing_or_mistyping_its_params_is_an_invalid_request() {
        let ok = Invocation::new().set("state", json!("s")).set("questions", json!(r#"[{"id":"q","type":"noul","instruction":"x"}]"#));
        assert!(parse_call(&ok).is_ok());
        let bad = [
            Invocation::new().set("questions", json!("[]")),
            Invocation::new().set("state", json!("s")),
            ok.clone().set("state_format", json!("xml")),
            ok.clone().set("deadline_ms", json!(0)),
            ok.clone().set("deadline_ms", json!(MAX_DEADLINE_MS + 1)),
            ok.clone().set("questions", json!("[]")),
            ok.clone().set("state_format", json!("json")).set("state", json!("{broken")),
            ok.clone().set("state", json!("x".repeat(wire::MAX_STATE_BYTES + 1))),
        ];
        for inv in bad {
            let err = parse_call(&inv).err().expect("must be refused");
            assert_eq!(class_of(&err), FailureClass::InvalidRequest, "{err}");
        }
    }

    // ---- the capability over real (tiny, random-weight) models ----

    use crate::engine::{DecideWeights, LayaWeights};
    use crate::fixtures::{write_decide, write_laya, Scratch};

    fn laya() -> (Scratch, Session) {
        laya_with_context(48)
    }

    fn laya_with_context(max_len: u32) -> (Scratch, Session) {
        let d = Scratch::new("caps-laya");
        write_laya(&d, max_len);
        let gpu = gpu_core::testgpu::dev(modernbert::kern::PIPELINES);
        let engine = Engine::Laya(Box::new(LayaWeights::read(d.to_str().unwrap(), None).unwrap().build(gpu, modernbert::Training::Off)));
        (d, Session::new(Served::new("t/laya"), engine))
    }

    fn decide_model() -> (Scratch, Session) {
        let d = Scratch::new("caps-decide");
        write_decide(&d, 5);
        let gpu = gpu_core::testgpu::dev(decide::kern::PIPELINES);
        let weights = DecideWeights::read(d.to_str().unwrap(), None, 0).unwrap();
        let engine = Engine::Decide(weights.build(gpu, decide::decide::Limits::default(), false));
        (d, Session::new(Served::new("t/decide"), engine))
    }

    const THREE: &str = r#"[
      {"id": "team", "type": "choice", "instruction": "who handles this",
       "options": [{"id": "billing", "description": "payments"}, "tech", "sales"]},
      {"id": "anger", "type": "score", "instruction": "how upset", "options": ["calm", "annoyed", "furious"]},
      {"id": "urgent", "type": "noul", "instruction": "is this urgent"}
    ]"#;

    fn call(questions: &str) -> Invocation {
        Invocation::new().set("state", json!("my payment failed again today")).set("questions", json!(questions))
    }

    fn sum_ranked(a: &Value) -> f64 {
        a["ranked"].as_array().unwrap().iter().map(|r| r["probability"].as_f64().unwrap()).sum()
    }

    fn check_three_answers(out: &Outcome, backend: &str) {
        assert_eq!(out.outputs["backend"], backend);
        assert!(out.outputs["latency_ms"].as_f64().unwrap() >= 0.0);
        let answers = out.outputs["answers"].as_array().unwrap();
        let ids: Vec<&str> = answers.iter().map(|a| a["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["team", "anger", "urgent"], "answers keep the order asked");
        for a in answers {
            assert!((sum_ranked(a) - 1.0).abs() < 1e-4, "{a}: the FULL distribution is returned and sums to one");
        }
        assert_eq!(answers[0]["ranked"].as_array().unwrap().len(), 3);
        let ranked_ids: Vec<&str> = answers[0]["ranked"].as_array().unwrap().iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert!(ranked_ids.contains(&"billing") && ranked_ids.contains(&"tech") && ranked_ids.contains(&"sales"));
        assert_eq!(answers[0]["selected"], answers[0]["ranked"][0]["id"], "the selection is the top of the ranking");
        assert_eq!(answers[1]["legend"], json!(["calm", "annoyed", "furious"]));
        assert!(answers[2].get("confidence").is_none());
        assert_eq!(out.outputs["flags"]["unsupported_question_types"], json!([]));
    }

    #[test]
    fn the_laya_arm_answers_three_typed_questions_with_full_distributions() {
        let (_d, mut s) = laya();
        let out = s.run(&call(THREE)).unwrap_or_else(|e| panic!("{e}"));
        check_three_answers(&out, "laya");
        assert_eq!(out.outputs["max_input_tokens"], 48);
        assert!(out.outputs.get("windows").is_none());
        assert_eq!(out.outputs["model"], "t/laya");
    }

    #[test]
    fn the_decide_arm_answers_the_same_request_and_reports_its_windows() {
        let (_d, mut s) = decide_model();
        let out = s.run(&call(THREE)).unwrap_or_else(|e| panic!("{e}"));
        check_three_answers(&out, "decide");
        assert_eq!(out.outputs["windows"], 1);
        assert_eq!(out.outputs["flags"]["state_truncated"], false, "decide windows its state and never drops it");
    }

    #[test]
    fn a_state_longer_than_laya_context_is_answered_from_a_prefix_and_says_so() {
        // A context wide enough that the question's own packing leaves room
        // for a short state, so only the long one is cut.
        let (_d, mut s) = laya_with_context(200);
        let long = "payment failing ".repeat(40);
        let inv = Invocation::new().set("state", json!(long)).set("questions", json!(THREE));
        let out = s.run(&inv).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.outputs["flags"]["state_truncated"], true);
        let st = &out.outputs["answers"][2]["state"];
        assert_eq!(st["truncated"], true);
        assert!(st["kept"].as_u64().unwrap() < st["tokens"].as_u64().unwrap());
        // ...and a short one is not.
        let short = s.run(&call(THREE)).unwrap();
        assert_eq!(short.outputs["flags"]["state_truncated"], false);
    }

    #[test]
    fn a_structured_states_key_order_reaches_the_model() {
        let (_d, mut s) = laya();
        let q = r#"[{"id": "u", "type": "noul", "instruction": "is this urgent"}]"#;
        let mut ask = |state: &str| {
            let inv = Invocation::new().set("state", json!(state)).set("state_format", json!("json")).set("questions", json!(q));
            s.run(&inv).unwrap_or_else(|e| panic!("{e}")).outputs["answers"][0]["noul"].as_f64().unwrap()
        };
        let one = ask(r#"{"zebra": "payment failed", "apple": "again"}"#);
        let two = ask(r#"{"apple": "again", "zebra": "payment failed"}"#);
        assert_ne!(one, two, "a sorted re-serialization would have made these the same bytes");
    }

    #[test]
    fn a_question_type_the_model_does_not_support_is_flagged_not_silently_dropped() {
        let (_d, mut s) = laya();
        s.served.question_types = vec![QuestionType::Choice, QuestionType::Noul];
        let out = s.run(&call(THREE)).unwrap_or_else(|e| panic!("{e}"));
        let answers = out.outputs["answers"].as_array().unwrap();
        assert_eq!(answers[1]["unsupported"], true);
        assert_eq!(answers[0]["type"], "choice");
        assert_eq!(answers[2]["type"], "noul");
        assert_eq!(out.outputs["flags"]["unsupported_question_types"], json!(["score"]));
    }

    #[test]
    fn a_request_no_question_of_which_is_supported_is_unsupported_capability() {
        let (_d, mut s) = laya();
        s.served.question_types = vec![QuestionType::Choice];
        let e = s.run(&call(r#"[{"id": "u", "type": "noul", "instruction": "x"}]"#)).err().expect("must fail");
        assert_eq!(class_of(&e), FailureClass::UnsupportedCapability, "{e}");
    }

    #[test]
    fn a_request_outside_the_published_limits_is_an_invalid_request_naming_the_question() {
        let (_d, mut s) = laya();
        let e = s.run(&call(r#"[{"id": "rate", "type": "score", "instruction": "x", "options": ["only"]}]"#)).err().expect("must fail");
        assert_eq!(class_of(&e), FailureClass::InvalidRequest);
        assert!(e.contains("rate"), "{e}");
    }

    #[test]
    fn a_passed_deadline_is_a_timeout_and_returns_no_partial_answers() {
        for (_d, mut s) in [laya(), decide_model()] {
            let long_ago = Instant::now() - Duration::from_secs(10);
            let inv = call(THREE).set("deadline_ms", json!(1));
            let e = s.run_started(&inv, long_ago).err().expect("must time out");
            assert_eq!(class_of(&e), FailureClass::Timeout, "{e}");
        }
    }

    #[test]
    fn a_cancelled_call_stops_before_the_model_runs() {
        let (_d, mut s) = laya();
        let mut inv = call(THREE);
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        assert_eq!(s.run(&inv).err().as_deref(), Some("cancelled"));
    }

    #[test]
    fn asking_twice_gives_the_same_answer() {
        let (_d, mut s) = laya();
        let a = s.run(&call(THREE)).unwrap().outputs["answers"].clone();
        let b = s.run(&call(THREE)).unwrap().outputs["answers"].clone();
        assert_eq!(a, b);
    }
}
