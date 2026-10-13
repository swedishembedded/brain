// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `POST /v1/systemone`: typed decisions in the System One wire shape, as a
//! thin layer over the `decide` capability every decision model serves.
//!
//! ```json
//! {"model": "auto",
//!  "state": {"ticket": "payments failing"},
//!  "questions": {"team":   {"type": "choice", "instructions": "who handles this",
//!                           "criteria": {"billing": "payments, refunds", "tech": "outages"}},
//!                "anger":  {"type": "score",  "instructions": "how upset",
//!                           "criteria": ["calm", "annoyed", "furious"]},
//!                "urgent": {"type": "noul",   "instructions": "is this urgent"}}}
//! ```
//!
//! answers
//!
//! ```json
//! {"model": "convaiinnovations/laya",
//!  "answers": {"team":   {"type": "choice", "choice": "billing", "confidence": 0.91,
//!                         "probabilities": {"billing": 0.98, "tech": 0.02}},
//!              "anger":  {"type": "score", "score": 1.3, "confidence": 0.4,
//!                         "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
//!                         "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}},
//!              "urgent": {"type": "noul", "noul": 0.78}},
//!  "usage": {"output_tokens": 0},
//!  "brain": {"backend": "laya", "latency_ms": 41.2, "max_input_tokens": 512,
//!            "flags": {"state_truncated": false, "unsupported_question_types": []}}}
//! ```
//!
//! * `model` is a decision model id, or `"auto"` (the default), which asks the
//!   `brain/decision` family for its `select`ion - the configured model, else
//!   a validated Laya model, else a validated `decide` model - and fails
//!   rather than answer from a model nobody validated.
//! * Probabilities are keyed in the order the caller listed the options, not
//!   best first; the capability's own response carries the ranked form.
//! * `usage.output_tokens` is always 0: a decision model generates no text.
//! * `brain` is an additive block with what the capability reports beyond the
//!   shared shape; a client written for the shared shape ignores it.
//! * A request outside the published limits (more than 255 options, a score
//!   with fewer than 2 or more than 10 levels, an unknown question type, a
//!   missing field) is a **422** naming the question; a body that is not JSON
//!   is a 400. Every failure body carries the capability's failure class as
//!   its `code`.
//!
//! The request is passed to the capability as the exact text the caller wrote,
//! so key order - the options' positions, the state's fields - survives to the
//! model. No decision-model code is linked here.
//!
//! Swedish Embedded AB implements typed decision endpoints, a model behind a
//! schema a service can rely on rather than free text a caller has to parse.
//! If your team needs one, you can procure our services by sending an email to
//! info@swedishembedded.com.

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use capability::{FailureClass, Invocation};
use serde_json::{json, Value};

use crate::auth::CallerExt;
use crate::bridge;
use crate::error::ApiError;
use crate::state::AppState;

/// The family model that answers "which decision model".
const FAMILY: &str = "brain/decision";
/// The action every decision model serves.
const ACTION: &str = "decide";
/// Largest request body: a state may be a document.
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/systemone", post(systemone)).layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
}

// ------------------------------------------------------------------ raw JSON members

/// The members of a JSON object as `(key, exact source text of its value)`, in
/// source order. Nothing is re-serialized, which is the point: the model reads
/// the caller's own bytes, so option order and state key order survive.
fn members(text: &str) -> Result<Vec<(String, &str)>, String> {
    let b = text.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'{') {
        return Err("the body must be a JSON object".into());
    }
    i += 1;
    let mut out = Vec::new();
    let mut after_comma = false;
    loop {
        i = skip_ws(b, i);
        match b.get(i) {
            // `{}` is an empty object, but `{"a": 1,}` is not valid JSON.
            Some(b'}') if !after_comma => break,
            Some(b'"') => {}
            _ => return Err("expected a member name".into()),
        }
        let key_end = string_end(b, i)?;
        let key: String = serde_json::from_str(&text[i..key_end]).map_err(|e| format!("bad member name: {e}"))?;
        i = skip_ws(b, key_end);
        if b.get(i) != Some(&b':') {
            return Err(format!("expected ':' after \"{key}\""));
        }
        i = skip_ws(b, i + 1);
        let end = value_end(b, i)?;
        out.push((key, &text[i..end]));
        i = skip_ws(b, end);
        match b.get(i) {
            Some(b',') => {
                i += 1;
                after_comma = true;
            }
            Some(b'}') => break,
            _ => return Err("expected ',' or '}'".into()),
        }
    }
    if skip_ws(b, i + 1) != b.len() {
        return Err("unexpected text after the object".into());
    }
    Ok(out)
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        i += 1;
    }
    i
}

/// The index just past the string starting at `i` (which holds the opening quote).
fn string_end(b: &[u8], i: usize) -> Result<usize, String> {
    let mut j = i + 1;
    while let Some(&c) = b.get(j) {
        match c {
            b'\\' => j += 2,
            b'"' => return Ok(j + 1),
            _ => j += 1,
        }
    }
    Err("unterminated string".into())
}

/// The index just past the JSON value starting at `i`.
fn value_end(b: &[u8], i: usize) -> Result<usize, String> {
    match b.get(i) {
        None => Err("expected a value".into()),
        Some(b'"') => string_end(b, i),
        Some(b'{' | b'[') => {
            let mut depth = 0usize;
            let mut j = i;
            while let Some(&c) = b.get(j) {
                match c {
                    b'"' => {
                        j = string_end(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            Err("unterminated object or array".into())
        }
        Some(_) => {
            let mut j = i;
            while let Some(&c) = b.get(j) {
                if matches!(c, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }
                j += 1;
            }
            if j == i {
                return Err("expected a value".into());
            }
            Ok(j)
        }
    }
}

// ------------------------------------------------------------------ the request

/// What the handler needs of a request body.
#[derive(Debug)]
struct Parsed<'a> {
    model: String,
    state: &'a str,
    questions: &'a str,
    /// The question names, in the order written.
    names: Vec<String>,
}

fn parse(text: &str) -> Result<Parsed<'_>, String> {
    let top = members(text)?;
    let find = |name: &str| top.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
    let model = match find("model") {
        None => "auto".to_string(),
        Some(raw) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::String(s)) if !s.is_empty() => s,
            Ok(Value::Null) => "auto".to_string(),
            _ => return Err("`model` must be a non-empty string".into()),
        },
    };
    let state = find("state").ok_or("`state` is required: it is what the questions are about")?;
    let questions = find("questions").ok_or("`questions` is required, and is an object of name -> question")?;
    if !questions.trim_start().starts_with('{') {
        return Err("`questions` must be an object of name -> question".into());
    }
    let names = members(questions).map_err(|e| format!("`questions`: {e}"))?.into_iter().map(|(k, _)| k).collect();
    Ok(Parsed { model, state, questions, names })
}

// ------------------------------------------------------------------ the response

/// A JSON object written in the given order. Values are already JSON text.
fn object(entries: Vec<(String, String)>) -> String {
    let body: Vec<String> = entries.into_iter().map(|(k, v)| format!("{}: {v}", Value::String(k))).collect();
    format!("{{{}}}", body.join(", "))
}

fn text(v: &Value) -> String {
    v.to_string()
}

/// One answer in the shared shape. `None` when the capability's answer is not
/// one a client of the shared shape can read.
fn render_answer(a: &Value) -> Option<String> {
    let ty = a.get("type")?.as_str()?;
    if a.get("unsupported").and_then(Value::as_bool) == Some(true) {
        return Some(object(vec![("type".into(), text(&json!(ty))), ("unsupported".into(), "true".into()), ("reason".into(), text(a.get("reason").unwrap_or(&Value::Null)))]));
    }
    // The capability ranks best-first; the shared shape keys probabilities in
    // the caller's own order, which each candidate's `index` restores.
    let mut ranked: Vec<&Value> = a.get("ranked")?.as_array()?.iter().collect();
    ranked.sort_by_key(|r| r.get("index").and_then(Value::as_u64).unwrap_or(u64::MAX));
    let probabilities = |key: &dyn Fn(&Value) -> Value| {
        object(ranked.iter().map(|r| (key(r).as_str().unwrap_or_default().to_string(), text(&r["probability"]))).collect())
    };
    let mut entries: Vec<(String, String)> = vec![("type".into(), text(&json!(ty)))];
    match ty {
        "choice" => {
            entries.push(("choice".into(), text(&a["selected"])));
            entries.push(("confidence".into(), text(&a["confidence"])));
            entries.push(("probabilities".into(), probabilities(&|r| r["id"].clone())));
        }
        "score" => {
            entries.push(("score".into(), text(&a["score"])));
            entries.push(("confidence".into(), text(&a["confidence"])));
            let legend = a.get("legend")?.as_array()?;
            entries.push(("legend".into(), object(legend.iter().enumerate().map(|(i, l)| (i.to_string(), text(l))).collect())));
            entries.push(("probabilities".into(), probabilities(&|r| r["id"].clone())));
        }
        // No `confidence`: a two-outcome probability already is its own.
        "noul" => entries.push(("noul".into(), text(&a["noul"]))),
        _ => return None,
    }
    Some(object(entries))
}

/// The whole response document.
fn render(model: &str, names: &[String], outputs: &Value) -> Result<String, String> {
    let answers = outputs.get("answers").and_then(Value::as_array).ok_or("the model returned no answers")?;
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let answer = answers
            .iter()
            .find(|a| a.get("id").and_then(Value::as_str) == Some(name))
            .ok_or_else(|| format!("the model returned no answer for question {name:?}"))?;
        entries.push((name.clone(), render_answer(answer).ok_or_else(|| format!("the model's answer to {name:?} has an unreadable shape"))?));
    }
    let brain = object(
        ["backend", "latency_ms", "max_input_tokens", "flags"].iter().filter_map(|k| outputs.get(*k).map(|v| ((*k).to_string(), text(v)))).collect(),
    );
    Ok(object(vec![
        ("model".into(), text(&json!(model))),
        ("answers".into(), object(entries)),
        ("usage".into(), r#"{"output_tokens": 0}"#.into()),
        ("brain".into(), brain),
    ]))
}

fn json_response(status: StatusCode, body: String) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

// ------------------------------------------------------------------ the handler

async fn systemone(State(state): State<AppState>, caller: CallerExt, body: Bytes) -> Response {
    let state = state.scoped(caller);
    let provider = state.provider;
    let Ok(text) = std::str::from_utf8(&body) else {
        return ApiError::invalid_request(provider, "the body must be UTF-8 JSON").into_response();
    };
    let parsed = match parse(text) {
        Ok(p) => p,
        // Malformed JSON is the caller's syntax; a well-formed body missing a
        // field is the semantic 422 the capability would also give.
        Err(e) if e.starts_with('`') => return ApiError::failure(provider, FailureClass::InvalidRequest, e).into_response(),
        Err(e) => return ApiError::invalid_request(provider, e).into_response(),
    };

    let model = match resolve_model(&state, &parsed.model).await {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };

    let mut inv = Invocation::new();
    inv = inv.set("state", json!(parsed.state)).set("state_format", json!("json")).set("questions", json!(parsed.questions));
    // The capability's own spec is the validator, exactly as for `/v1/run`.
    let manifests = state.exec.manifests();
    let spec = manifests.iter().find(|m| m.model == model).and_then(|m| m.actions.iter().find(|a| a.name == ACTION));
    let inv = match spec.map(|s| s.validate_served(inv)) {
        Some(Ok(inv)) => inv,
        Some(Err(e)) => return ApiError::failure(provider, FailureClass::InvalidRequest, e).into_response(),
        None => return ApiError::model_not_found(provider, &model).into_response(),
    };

    match bridge::submit(&state, &model, ACTION, inv).await {
        Ok(outcome) => match render(&model, &parsed.names, &outcome.outputs) {
            Ok(doc) => json_response(StatusCode::OK, doc),
            Err(e) => {
                eprintln!("apiserve: systemone: {e}");
                ApiError::internal(provider, "the model's answer could not be rendered").into_response()
            }
        },
        Err(e) => e.into_response(),
    }
}

/// The decision model `requested` names: an id that serves `decide`, or
/// whatever `"auto"` selects.
async fn resolve_model(state: &AppState, requested: &str) -> Result<String, ApiError> {
    let provider = state.provider;
    if requested != "auto" {
        let serves = state.exec.manifests().iter().any(|m| m.model == requested && m.actions.iter().any(|a| a.name == ACTION));
        return if serves && state.lists(requested) { Ok(requested.to_string()) } else { Err(ApiError::model_not_found(provider, requested)) };
    }
    let mut inv = Invocation::new();
    inv = inv.set("model", json!("auto"));
    let selected = bridge::submit(state, FAMILY, "select", inv).await?;
    match selected.outputs.get("chosen").and_then(Value::as_str) {
        Some(id) => Ok(id.to_string()),
        None => {
            let reason = selected.outputs.get("reason").and_then(Value::as_str).unwrap_or("no decision model is available");
            Err(ApiError::failure(provider, FailureClass::NotInstalled, format!("no decision model was selected: {reason}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn members_keep_source_order_and_exact_value_text() {
        let m = members(r#" {"b": {"z": 1, "a": [1, 2, {"x": "}"}]}, "a": "t\"q", "c": 4.50 } "#).unwrap();
        let keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
        assert_eq!(m[0].1, r#"{"z": 1, "a": [1, 2, {"x": "}"}]}"#, "the caller's own bytes, key order intact");
        assert_eq!(m[1].1, r#""t\"q""#);
        assert_eq!(m[2].1, "4.50");
    }

    #[test]
    fn malformed_bodies_are_refused() {
        for bad in ["", "[]", "{", r#"{"a" 1}"#, r#"{"a": }"#, r#"{"a": 1,}"#, r#"{"a": "x"#, r#"{"a": 1} x"#, r#"{"a": {"b": 1}"#] {
            assert!(members(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn a_request_parses_to_its_model_state_and_question_names_in_order() {
        let p = parse(r#"{"state": {"b": 1, "a": 2}, "questions": {"z": {"type": "noul"}, "m": {"type": "noul"}}}"#).unwrap();
        assert_eq!(p.model, "auto");
        assert_eq!(p.state, r#"{"b": 1, "a": 2}"#);
        assert_eq!(p.names, ["z", "m"]);
        let p = parse(r#"{"model": "x/y", "state": "s", "questions": {"q": {}}}"#).unwrap();
        assert_eq!(p.model, "x/y");
    }

    #[test]
    fn missing_fields_are_named() {
        assert!(parse(r#"{"questions": {}}"#).unwrap_err().contains("`state`"));
        assert!(parse(r#"{"state": "s"}"#).unwrap_err().contains("`questions`"));
        assert!(parse(r#"{"state": "s", "questions": []}"#).unwrap_err().contains("object"));
        assert!(parse(r#"{"state": "s", "questions": {}, "model": 3}"#).unwrap_err().contains("`model`"));
    }

    fn outputs() -> Value {
        json!({
            "backend": "laya", "latency_ms": 1.5, "max_input_tokens": 512,
            "flags": {"state_truncated": false, "unsupported_question_types": []},
            "answers": [
                {"id": "team", "type": "choice", "selected": "tech", "confidence": 0.5,
                 "ranked": [{"id": "tech", "index": 1, "probability": 0.7}, {"id": "billing", "index": 0, "probability": 0.3}]},
                {"id": "anger", "type": "score", "score": 1.5, "confidence": 0.2, "legend": ["calm", "angry"],
                 "ranked": [{"id": "1", "index": 1, "label": "angry", "probability": 0.6}, {"id": "0", "index": 0, "label": "calm", "probability": 0.4}]},
                {"id": "urgent", "type": "noul", "noul": 0.25,
                 "ranked": [{"id": "false", "index": 0, "probability": 0.75}, {"id": "true", "index": 1, "probability": 0.25}]}
            ]
        })
    }

    #[test]
    fn the_response_is_the_shared_shape_with_probabilities_in_the_callers_order() {
        let names = vec!["team".to_string(), "anger".to_string(), "urgent".to_string()];
        let doc = render("m/x", &names, &outputs()).unwrap();
        let back: Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(back["model"], "m/x");
        assert_eq!(back["usage"], json!({"output_tokens": 0}));
        assert_eq!(back["answers"]["team"], json!({"type": "choice", "choice": "tech", "confidence": 0.5, "probabilities": {"billing": 0.3, "tech": 0.7}}));
        assert_eq!(
            back["answers"]["anger"],
            json!({"type": "score", "score": 1.5, "confidence": 0.2, "legend": {"0": "calm", "1": "angry"}, "probabilities": {"0": 0.4, "1": 0.6}})
        );
        assert_eq!(back["answers"]["urgent"], json!({"type": "noul", "noul": 0.25}), "a noul carries no confidence");
        assert_eq!(back["brain"]["backend"], "laya");
        // Order is part of the contract: the caller's option order, then names.
        assert!(doc.find("\"billing\"").unwrap() < doc.find("\"tech\": 0.7").unwrap(), "{doc}");
        assert!(doc.find("\"team\"").unwrap() < doc.find("\"anger\"").unwrap(), "{doc}");
    }

    #[test]
    fn an_unsupported_question_is_reported_not_dropped() {
        let o = json!({"answers": [{"id": "q", "type": "score", "unsupported": true, "reason": "no scores here"}]});
        let doc: Value = serde_json::from_str(&render("m", &["q".to_string()], &o).unwrap()).unwrap();
        assert_eq!(doc["answers"]["q"], json!({"type": "score", "unsupported": true, "reason": "no scores here"}));
    }

    #[test]
    fn a_missing_or_unreadable_answer_is_an_error_not_a_hole() {
        assert!(render("m", &["gone".to_string()], &outputs()).unwrap_err().contains("gone"));
        assert!(render("m", &[], &json!({})).is_err());
    }
}
